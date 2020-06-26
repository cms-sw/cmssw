"""
_|_|_|      _|_|      _|      _|        _|_|_|  _|    _|  _|_|_|  
_|    _|  _|    _|    _|_|  _|_|      _|        _|    _|    _|    
_|    _|  _|  _|_|    _|  _|  _|      _|  _|_|  _|    _|    _|    
_|    _|  _|    _|    _|      _|      _|    _|  _|    _|    _|    
_|_|_|      _|_|  _|  _|      _|        _|_|_|    _|_|    _|_|_|  

This is an entry point to the DQM GUI application. It can be started like this: python3 app.py

This file configures and initializes aiohttp web server and all DQM GUI services. 
Responsibilities of the endpoint methods here are to parse input parameters, call 
the corresponding service methods to get the result and format the output.

Each method is defined twice: for legacy API and for new, v1 API.
If a new version of the API needs to be provided, new /v2/ methods can be provided
and configured here.
"""

from .helpers import get_absolute_path

# Add local python packages dir (if it exists) to python path.
import sys, os
local_packages_dir = get_absolute_path('.python_packages/')
if os.path.isdir(local_packages_dir):
    sys.path.insert(0, local_packages_dir)

# Initialize a process pool before doing anything else.
# This is to make sure that the fork() happens before any imports, and before 
# any threads are created.
processpoolexecutor = None
if __name__ == '__main__':
    import multiprocessing
    # forkserver means that a dummy process (a fork server process) will be forked
    # right now, before any threads/locks are created. Whenever a new process will
    # be requested by the ProcessPoolExecutor, fork server process will be forked
    # instead of the main process.
    multiprocessing.set_start_method('forkserver')
    # from concurrent.futures import ProcessPoolExecutor
    from .helpers import ResilientProcessPoolExecutor
    import signal
    # remove SIGINT handler for the fork'ed children to avoid the stack traces on shutdown.
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    # TODO: make process count configurable? 4 seems to be enough to saturate IO.
    processpoolexecutor = ResilientProcessPoolExecutor(8)
    # concurrent.futures initializes the actual multiprocessing pool lazily. So we
    # need to submit some work here to start the processes.
    fut = processpoolexecutor.submit(print, "Process pool initialized.")
    fut.result()
    signal.signal(signal.SIGINT, original_sigint_handler)
# Now we should be safe.

import asyncio
import logging
import argparse

from aiohttp import web, WSCloseCode
from logging.handlers import TimedRotatingFileHandler

from .service import GUIService
from .storage import GUIDataStore
from .helpers import get_absolute_path, parse_run_lumi
from .rendering import GUIRenderer
from .data_types import RenderingOptions, MEDescription, SampleFull
from .importing.importing import GUIImportManager
from .layouts.layout_manager import LayoutManager

# Services
service = GUIService()
layout_manager = LayoutManager()


# ###################################################################################################### #
# =========================== API endpoint handling methods for all versions =========================== #
# ###################################################################################################### #

async def index(request):
    return web.FileResponse(get_absolute_path('../data/index.html'))


async def samples_legacy(request):
    """Returns a list of matching run/dataset pairs based on provided regex search."""

    run, lumi = parse_run_lumi(request.rel_url.query.get('run'))
    dataset = request.rel_url.query.get('match')

    samples = await service.get_samples(run, dataset, lumi)

    result = {
        'samples': [{
            'type': 'offline_data',
            'items': [{
                'run': str(sample.run) if sample.lumi == 0 else '%s:%s' % (sample.run, sample.lumi),
                'dataset': sample.dataset
            } for sample in samples]
        }]
    }
    return web.json_response(result)


async def samples_v1(request):
    """Returns a list of matching run/dataset pairs based on provided regex search."""

    run = request.rel_url.query.get('run')
    lumi = request.rel_url.query.get('lumi', 0)
    dataset = request.rel_url.query.get('dataset')

    samples = await service.get_samples(run, dataset, lumi)

    result = {
        'data': [{
            'run': sample.run,
            'lumi': sample.lumi,
            'dataset': sample.dataset,
        } for sample in samples]
    }
    return web.json_response(result)


async def archive_legacy(request):
    """Returns a directory listing for provided run:lumi/dataset/path combination."""

    run, lumi = parse_run_lumi(request.match_info['run'])
    full_path = request.match_info['path']
    search = request.rel_url.query.get('search')

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    data = await service.get_archive(run, dataset, path, search, lumi)
    if not data:
        return web.HTTPNotFound()

    result = {'contents': []}
    result['contents'].extend({'subdir': name, 'me_count': me_count} for name, me_count in data.dirs)
    result['contents'].extend({'obj': name, 'path': path, 'layout': layout} for name, path, layout in data.objs)

    return web.json_response(result)


async def archive_v1(request):
    """Returns a directory listing for provided run:lumi/dataset/path combination."""

    run, lumi = parse_run_lumi(request.match_info['run'])
    full_path = request.match_info['path']
    search = request.rel_url.query.get('search')

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    data = await service.get_archive(run, dataset, path, search, lumi)
    if not data:
        return web.HTTPNotFound()

    result = {'data': []}
    result['data'].extend({'subdir': name, 'me_count': me_count} for name, me_count in data.dirs)
    result['data'].extend({'name': name, 'path': path, 'layout': layout} for name, path, layout in data.objs)

    return web.json_response(result)


# This endpoint doesn't exist in legacy API
async def layouts_v1(request):
    """Returns all monitor elements present in the layout of a given name"""

    name = request.rel_url.query.get('name')

    layouts = layout_manager.get_layouts_by_name(name)

    result = {'data':
        [{'source': x.source, 'destination': x.destination} for x in layouts]
    }

    return web.json_response(result)


async def render_legacy(request):
    """Returns a PNG image for provided run:lumi/dataset/path combination"""

    run, lumi = parse_run_lumi(request.match_info['run'])
    full_path = request.match_info['path']
    options = RenderingOptions.from_dict_legacy(request.rel_url.query)

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    me_description = MEDescription(dataset, path, run, lumi)

    data = await service.get_rendered_image([me_description], options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    elif data == b'error':
        return web.HTTPBadRequest()
    return web.Response(body=data, content_type='image/png')


async def render_v1(request):
    """Returns a PNG image for provided run:lumi/dataset/path combination"""

    run, lumi = parse_run_lumi(request.match_info['run'])
    full_path = request.match_info['path']
    options = RenderingOptions.from_dict(request.rel_url.query)

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    me_description = MEDescription(dataset, path, run, lumi)

    data = await service.get_rendered_image([me_description], options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    elif data == b'error':
        return web.HTTPBadRequest()
    return web.Response(body=data, content_type='image/png')


async def render_overlay_legacy(request):
    """Returns a PNG image for provided run:lumi/dataset/path combination"""

    options = RenderingOptions.from_dict_legacy(request.rel_url.query)

    me_descriptions = []
    for obj in request.rel_url.query.getall('obj', []):
        parts = obj.split('/')
        run, lumi = parse_run_lumi(parts[1])
        dataset = '/' + '/'.join(parts[2:5])
        path = '/'.join(parts[5:])

        me_description = MEDescription(dataset, path, run, lumi)
        me_descriptions.append(me_description)

    data = await service.get_rendered_image(me_descriptions, options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    elif data == b'error':
        return web.HTTPBadRequest()
    return web.Response(body=data, content_type='image/png')


async def render_overlay_v1(request):
    """Returns a PNG image for provided run:lumi/dataset/path combination"""

    options = RenderingOptions.from_dict(request.rel_url.query)

    me_descriptions = []
    for obj in request.rel_url.query.getall('obj', []):
        parts = obj.split('/')
        run, lumi = parse_run_lumi(parts[1])
        dataset = '/' + '/'.join(parts[2:5])
        path = '/'.join(parts[5:])

        me_description = MEDescription(dataset, path, run, lumi)
        me_descriptions.append(me_description)

    data = await service.get_rendered_image(me_descriptions, options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    elif data == b'error':
        return web.HTTPBadRequest()
    return web.Response(body=data, content_type='image/png')


async def jsroot_legacy(request):
    """Returns a JSON representation of a ROOT histogram for provided run:lumi/dataset/path combination"""

    run, lumi = parse_run_lumi(request.match_info['run'])
    full_path = request.match_info['path']

    # This is caused by a double slash in the url
    if full_path[0] == '/':
        full_path = full_path[1:]

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    me_description = MEDescription(dataset, path, run, lumi)
    options = RenderingOptions(json=True)

    data = await service.get_rendered_json([me_description], options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    elif data == b'error':
        return web.HTTPBadRequest()
    return web.json_response(data)


async def jsroot_overlay(request):
    """Returns a list of JSON representations of ROOT histograms for provided run:lumi/dataset/path combinations"""

    me_descriptions = []
    for obj in request.rel_url.query.getall('obj', []):
        parts = obj.split('/')
        run, lumi = parse_run_lumi(parts[1])
        dataset = '/' + '/'.join(parts[2:5])
        path = '/'.join(parts[5:])

        me_description = MEDescription(dataset, path, run, lumi)
        me_descriptions.append(me_description)

    options = RenderingOptions(json=True)

    data = await service.get_rendered_json(me_descriptions, options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    elif data == b'error':
        return web.HTTPBadRequest()
    return web.json_response(data)


async def available_lumis_v1(request):
    """Returns a list of available lumisections for provided dataset/run combination."""

    run = request.match_info['run']
    dataset = '/' + request.match_info['dataset']

    data = await service.get_available_lumis(dataset, run)
    return web.json_response(data)


async def register(request):
    """
    Regsiters a sample into a database. 
    A list of samples has to be posted in HTTP body, in JSON format:
    [{"dataset": "/a/b/c", "run": "123456", "lumi": "0", "file": "/a/b/c.root", "fileformat": 1}]'
    """

    samples = await request.json()
    samples = [SampleFull(dataset=x['dataset'], run=int(x['run']), lumi=int(x['lumi']), 
        file=x['file'], fileformat=x['fileformat']) for x in samples]

    await service.register_samples(samples)
    
    return web.HTTPCreated()


# ###################################################################################################### #
# ==================== Server configuration, initialization/destruction of services ==================== #
# ###################################################################################################### #

async def initialize_services(in_memory, files, workers):
    await GUIDataStore.initialize(in_memory=in_memory)
    await GUIImportManager.initialize(files=files, executor=processpoolexecutor)
    await GUIRenderer.initialize(workers=workers)


async def destroy_services():
    await GUIDataStore.destroy()
    await GUIImportManager.destroy()
    await GUIRenderer.destroy()
    processpoolexecutor.shutdown(wait=True)


async def on_shutdown(app):
    print('\nDestroying services...')
    await destroy_services()


def config_and_start_webserver(port):
    app = web.Application(middlewares=[
        web.normalize_path_middleware(append_slash=True, merge_slashes=True),
    ])


    # Legacy routes
    app.add_routes([web.get('/data/json/samples', samples_legacy),
                    web.get(r'/data/json/archive/{run}/{path:.+}', archive_legacy),
                    web.get(r'/plotfairy/archive/{run}/{path:.+}', render_legacy),
                    web.get('/plotfairy/overlay', render_overlay_legacy),
                    web.get(r'/jsrootfairy/archive/{run}/{path:.+}', jsroot_legacy),])

    # Version 1 API routes
    app.add_routes([web.get('/api/v1/samples', samples_v1),
                    web.get('/api/v1/layouts', layouts_v1),
                    web.get(r'/api/v1/archive/{run}/{path:.+}', archive_v1),
                    web.get(r'/api/v1/render/{run}/{path:.+}', render_v1),
                    web.get('/api/v1/render_overlay', render_overlay_v1),
                    web.get(r'/api/v1/json/{run}/{path:.+}', jsroot_legacy),
                    web.get('/api/v1/json_overlay', jsroot_overlay),
                    web.get(r'/api/v1/lumis/{run}/{dataset:.+}', available_lumis_v1),
                    web.post('/api/v1/register', register)])

    # Routes for HTML files
    app.add_routes([web.get('/', index), web.static('/', get_absolute_path('../data/'), show_index=True)])

    app.on_shutdown.append(on_shutdown)

    web.run_app(app, port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQM GUI API')
    parser.add_argument('-f', dest='files', nargs='*', help='DQM files to be imported.')
    parser.add_argument('-p', dest='port', type=int, default=8889, help='Server port.')
    parser.add_argument('-r', dest='renderers', type=int, default=2, help='Number of renderer processes.')
    parser.add_argument('--in-memory', dest='in_memory', default=False, action='store_true', help='If set uses an in memory database.')
    parser.add_argument('--stderr', default=False, action='store_true', help='If set log to stdout instead of log files.')
    args = parser.parse_args()

    # Setup rotating file loggin
    def log_file_namer(filename):
        parts = filename.split('/')
        parts[-1] = f'access_{parts[-1][11:]}.log'
        return '/'.join(parts)
    
    if not args.stderr:
        handler = TimedRotatingFileHandler(get_absolute_path('logs/access.log'), when='midnight', interval=1)
        handler.namer = log_file_namer
    else:
        handler = logging.StreamHandler()
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    asyncio.get_event_loop().run_until_complete(initialize_services(args.in_memory, args.files, args.renderers))
    config_and_start_webserver(args.port)
