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

from logging.handlers import TimedRotatingFileHandler

from rendering import GUIRenderer
from service import GUIService
from storage import GUIDataStore
from aiohttp import web, WSCloseCode

from gui_types import RenderingOptions, MEDescription

from layouts.layout_manager import LayoutManager


service = GUIService()
layout_manager = LayoutManager()


# ###################################################################################################### #
# =========================== API endpoint handling methods for all versions =========================== #
# ###################################################################################################### #

async def index(request):
    return web.FileResponse('../data/index.html')


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
    result['contents'].extend({'subdir': x.name} for x in data.dirs)
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
    result['data'].extend({'subdir': x.name} for x in data.dirs)
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

    me_description = MEDescription(run, dataset, path)

    data = await service.get_rendered_image([me_description], options)

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

    me_description = MEDescription(run, dataset, path)

    data = await service.get_rendered_image([me_description], options)

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

        me_description = MEDescription(run, dataset, path)
        me_descriptions.append(me_description)

    data = await service.get_rendered_image(me_descriptions, options)

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

        me_description = MEDescription(run, dataset, path)
        me_descriptions.append(me_description)

    data = await service.get_rendered_image(me_descriptions, options)

    return web.Response(body=data, content_type='image/png')


# ###################################################################################################### #
# ==================== Server configuration, initialization/destruction of services ==================== #
# ###################################################################################################### #

async def initialize_services():
    await GUIDataStore.initialize()
    await GUIRenderer.initialize(workers=2)


async def destroy_services():
    await GUIDataStore.destroy()
    await GUIRenderer.destroy()


async def on_shutdown(app):
    print('\nDestroying services...')
    await destroy_services()


def config_and_start_webserver():
    app = web.Application(middlewares=[
        web.normalize_path_middleware(append_slash=True, merge_slashes=True),
    ])

    # Setup rotating file loggin
    def log_file_namer(filename):
        parts = filename.split('/')
        parts[-1] = f'access_{parts[-1][11:]}.log'
        return '/'.join(parts)
    
    handler = TimedRotatingFileHandler('logs/access.log', when='midnight', interval=1)
    handler.namer = log_file_namer
    logger = logging.getLogger('aiohttp.access')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    # Legacy routes
    app.add_routes([web.get('/data/json/samples', samples_legacy),
                    web.get(r'/data/json/archive/{run}/{path:.+}', archive_legacy),
                    web.get(r'/plotfairy/archive/{run}/{path:.+}', render_legacy),
                    web.get(r'/plotfairy/overlay', render_overlay_legacy)])

    # Version 1 API routes
    app.add_routes([web.get('/api/v1/samples', samples_v1),
                    web.get('/api/v1/layouts', layouts_v1),
                    web.get(r'/api/v1/archive/{run}/{path:.+}', archive_v1),
                    web.get(r'/api/v1/render/{run}/{path:.+}', render_v1),
                    web.get(r'/api/v1/render_overlay', render_overlay_v1)])

    # Routes for HTML files
    app.add_routes([web.get('/', index), web.static('/', '../data/', show_index=True)])

    app.on_shutdown.append(on_shutdown)

    web.run_app(app, port='8889')


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(initialize_services())
    config_and_start_webserver()
