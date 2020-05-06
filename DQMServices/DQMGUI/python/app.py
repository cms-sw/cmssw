import time
import asyncio
import logging

from logging.handlers import TimedRotatingFileHandler

from rendering import GUIRenderer
from helpers import MEDescription
from service import GUIService
from storage import GUIDataStore
from aiohttp import web, WSCloseCode


service = GUIService()


async def index(request):
    return web.FileResponse('../data/index.html')


async def samples(request):
    """Returns a list of matching run/dataset pairs based on provided regex search."""

    run = request.rel_url.query.get('run')
    dataset = request.rel_url.query.get('match')

    data = await service.get_samples(run, dataset)
    return web.json_response(data)


async def archive(request):
    """Returns a directory listing for provided run/dataset/path combination."""

    run = request.match_info['run']
    full_path = request.match_info['path']
    search = request.rel_url.query.get('search')

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    data = await service.get_archive(run, dataset, path, search)
    return web.json_response(data)


async def render(request):
    """Returns a PNG image for provided run/dataset/path combination"""

    run = request.match_info['run']
    full_path = request.match_info['path']
    width = int(request.rel_url.query.get('w', 266))
    height = int(request.rel_url.query.get('h', 200))
    # TODO: Standartize these arguments
    stats = int(request.rel_url.query.get('showstats', 1)) == 1
    normalize = str(request.rel_url.query.get('norm', 'True')) == 'True'
    error_bars = int(request.rel_url.query.get('showerrbars', 0)) == 1

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    me_description = MEDescription(run, dataset, path)

    data = await service.get_rendered_image([me_description], width, height, stats, normalize, error_bars)

    return web.Response(body=data, content_type="image/png")


async def render_overlay(request):
    """Returns a PNG image for provided run/dataset/path combination"""

    width = int(request.rel_url.query.get('w', 200))
    height = int(request.rel_url.query.get('h', 200))
    stats = int(request.rel_url.query.get('showstats', 1)) == 1
    normalize = str(request.rel_url.query.get('norm', 'True')) == 'True'
    error_bars = int(request.rel_url.query.get('showerrbars', 0)) == 1

    me_descriptions = []
    for obj in request.rel_url.query.getall('obj', []):
        parts = obj.split('/')
        run = int(parts[1])
        dataset = '/' + '/'.join(parts[2:5])
        path = '/'.join(parts[5:])

        me_description = MEDescription(run, dataset, path)
        me_descriptions.append(me_description)

    data = await service.get_rendered_image(me_descriptions, width, height, stats, normalize, error_bars)

    return web.Response(body=data, content_type="image/png")


async def initialize_services():
    await GUIDataStore.initialize()
    await GUIRenderer.initialize(workers=2)


async def destroy_services():
    await GUIDataStore.destroy()
    await GUIRenderer.destroy()


def config_and_start_webserver():
    app = web.Application(middlewares=[
        web.normalize_path_middleware(append_slash=True, merge_slashes=True),
    ])

    # Setup rotating file loggin
    def log_file_namer(filename):
        parts = filename.split('/')
        parts[-1] = f'dqmgui_{parts[-1][11:-1]}.log'
        return '/'.join(parts)
    
    handler = TimedRotatingFileHandler('logs/dqmgui.log', when='midnight', interval=1)
    handler.namer = log_file_namer
    logger = logging.getLogger('aiohttp.access')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    app.add_routes([web.get('/', index),
                    web.get('/data/json/samples', samples),
                    web.get(r'/data/json/archive/{run}/{path:.+}', archive),
                    web.get(r'/plotfairy/archive/{run}/{path:.+}', render),
                    web.get(r'/plotfairy/overlay', render_overlay)])
    app.add_routes([web.static('/', '../data/', show_index=True)])

    app.on_shutdown.append(on_shutdown)

    web.run_app(app, port='8889')


async def on_shutdown(app):
    print('\nDestroying services...')
    await destroy_services()


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(initialize_services())
    config_and_start_webserver()
