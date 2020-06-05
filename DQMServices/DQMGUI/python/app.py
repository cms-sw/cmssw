"""
This is an entry point to the application. It can be started like this: python3 app.py

This file configures and initializes aiohttp web server and all DQM GUI services. 
Responsibilities of the endpoint methods here are to parse input parameters, call 
the corresponding service methods to get the result and format the output.

Each method is defined twice: for legacy API and for new, v1 API.
If a new version of the API needs to be provided, new /v2/ methods can be provided
and configured here.
"""

import time
import asyncio
import logging
import argparse

from logging.handlers import TimedRotatingFileHandler

from data_types import SampleFull
from rendering import GUIRenderer
from service import GUIService
from storage import GUIDataStore
from importing.importing import GUIImportManager
from aiohttp import web, WSCloseCode
from helpers import get_absolute_path

from data_types import RenderingOptions, MEDescription

from layouts.layout_manager import LayoutManager


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

    run = request.rel_url.query.get('run')
    dataset = request.rel_url.query.get('match')

    samples = await service.get_samples(run, dataset)

    result = {
        'samples': [{
            'type': 'offline_data',
            'items': [{
                'run': sample.run,
                'dataset': sample.dataset
            } for sample in samples]
        }]
    }
    return web.json_response(result)


async def samples_v1(request):
    """Returns a list of matching run/dataset pairs based on provided regex search."""

    run = request.rel_url.query.get('run')
    dataset = request.rel_url.query.get('dataset')

    samples = await service.get_samples(run, dataset)

    result = {
        'data': [{
            'run': sample.run,
            'dataset': sample.dataset
        } for sample in samples]
    }
    return web.json_response(result)


async def archive_legacy(request):
    """Returns a directory listing for provided run/dataset/path combination."""

    run = request.match_info['run']
    full_path = request.match_info['path']
    search = request.rel_url.query.get('search')

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    data = await service.get_archive(run, dataset, path, search)

    result = {'contents': []}
    result['contents'].extend({'subdir': name, 'me_count': me_count} for name, me_count in data.dirs)
    result['contents'].extend({'obj': name, 'dir': path, 'layout': layout} for name, path, layout in data.objs)

    return web.json_response(result)


async def archive_v1(request):
    """Returns a directory listing for provided run/dataset/path combination."""

    run = request.match_info['run']
    full_path = request.match_info['path']
    search = request.rel_url.query.get('search')

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    data = await service.get_archive(run, dataset, path, search)

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
    """Returns a PNG image for provided run/dataset/path combination"""

    run = request.match_info['run']
    full_path = request.match_info['path']
    options = RenderingOptions.from_dict_legacy(request.rel_url.query)

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    me_description = MEDescription(dataset, path, run, lumi=0)

    data = await service.get_rendered_image([me_description], options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    return web.Response(body=data, content_type='image/png')


async def render_v1(request):
    """Returns a PNG image for provided run/dataset/path combination"""

    run = request.match_info['run']
    full_path = request.match_info['path']
    options = RenderingOptions.from_dict(request.rel_url.query)

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    me_description = MEDescription(dataset, path, run, lumi=0)

    data = await service.get_rendered_image([me_description], options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    return web.Response(body=data, content_type='image/png')


async def render_overlay_legacy(request):
    """Returns a PNG image for provided run/dataset/path combination"""

    options = RenderingOptions.from_dict_legacy(request.rel_url.query)

    me_descriptions = []
    for obj in request.rel_url.query.getall('obj', []):
        parts = obj.split('/')
        run = int(parts[1])
        dataset = '/' + '/'.join(parts[2:5])
        path = '/'.join(parts[5:])

        me_description = MEDescription(dataset, path, run, lumi=0)
        me_descriptions.append(me_description)

    data = await service.get_rendered_image(me_descriptions, options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    return web.Response(body=data, content_type='image/png')


async def render_overlay_v1(request):
    """Returns a PNG image for provided run/dataset/path combination"""

    options = RenderingOptions.from_dict(request.rel_url.query)

    me_descriptions = []
    for obj in request.rel_url.query.getall('obj', []):
        parts = obj.split('/')
        run = int(parts[1])
        dataset = '/' + '/'.join(parts[2:5])
        path = '/'.join(parts[5:])

        me_description = MEDescription(dataset, path, run, lumi=0)
        me_descriptions.append(me_description)

    data = await service.get_rendered_image(me_descriptions, options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    return web.Response(body=data, content_type='image/png')


async def jsroot_legacy(request):
    """Returns a JSON representation of a ROOT histogram for provided run/dataset/path combination"""

    run = request.match_info['run']
    full_path = request.match_info['path']

    # This is caused by a double slash in the url
    if full_path[0] == '/':
        full_path = full_path[1:]

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    me_description = MEDescription(dataset, path, run, lumi=0)
    options = RenderingOptions(json=True)

    data = await service.get_rendered_json([me_description], options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
    return web.json_response(data)


async def jsroot_overlay(request):
    """Returns a list of JSON representations of ROOT histograms for provided run/dataset/path combinations"""

    me_descriptions = []
    for obj in request.rel_url.query.getall('obj', []):
        parts = obj.split('/')
        run = int(parts[1])
        dataset = '/' + '/'.join(parts[2:5])
        path = '/'.join(parts[5:])

        me_description = MEDescription(dataset, path, run, lumi=0)
        me_descriptions.append(me_description)

    options = RenderingOptions(json=True)

    data = await service.get_rendered_json(me_descriptions, options)

    if data == b'crashed':
        return web.HTTPInternalServerError()
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
    await GUIImportManager.initialize(files=files)
    await GUIRenderer.initialize(workers=workers)


async def destroy_services():
    await GUIDataStore.destroy()
    await GUIRenderer.destroy()


async def on_shutdown(app):
    print('\nDestroying services...')
    await destroy_services()


def config_and_start_webserver(port):
    app = web.Application(middlewares=[
        web.normalize_path_middleware(append_slash=True, merge_slashes=True),
    ])

    # Setup rotating file loggin
    def log_file_namer(filename):
        parts = filename.split('/')
        parts[-1] = f'access_{parts[-1][11:]}.log'
        return '/'.join(parts)
    
    handler = TimedRotatingFileHandler(get_absolute_path('logs/access.log'), when='midnight', interval=1)
    handler.namer = log_file_namer
    logger = logging.getLogger('aiohttp.access')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

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
                    web.post('/api/v1/register', register)])

    # Routes for HTML files
    app.add_routes([web.get('/', index), web.static('/', get_absolute_path('../data/'), show_index=True)])

    app.on_shutdown.append(on_shutdown)

    web.run_app(app, port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQM GUI API')
    parser.add_argument('-f', dest='files', nargs='+', help='DQM files to be imported.')
    parser.add_argument('-p', dest='port', type=int, default=8889, help='Server port.')
    parser.add_argument('-r', dest='renderers', type=int, default=2, help='Number of renderer processes.')
    parser.add_argument('--in-memory', dest='in_memory', default=False, action='store_true', help='If set uses an in memory database.')
    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(initialize_services(args.in_memory, args.files, args.renderers))
    config_and_start_webserver(args.port)
