from rendering import DQMRenderer
from api_service import DQMAPIService
from aiohttp import web

async def index(request):
    return web.FileResponse('../data/index.html')

async def samples(request):
    """Returns a list of matching run/dataset pairs based on provided regex search."""

    run = request.rel_url.query.get('run')
    dataset = request.rel_url.query.get('match')

    service = DQMAPIService()
    data = service.get_samples(run, dataset)
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

    service = DQMAPIService()
    data = service.get_archive(run, dataset, path, search)
    return web.json_response(data)

async def render(request):
    run = request.match_info['run']
    full_path = request.match_info['path']
    width = request.rel_url.query.get('w', 200)
    height = request.rel_url.query.get('h', 200)

    # Separate dataset and a path within the root file
    parts = full_path.split('/')
    dataset = '/' + '/'.join(parts[0:3])
    path = '/'.join(parts[3:])

    service = DQMAPIService()
    data = await service.get_rendered_image(run, dataset, path, width, height)

    return web.Response(body=data, content_type="image/png")


def config_and_start_webserver():
    app = web.Application(middlewares=[
        web.normalize_path_middleware(append_slash=True, merge_slashes=True),
    ])

    app.add_routes([web.get('/', index),
                    web.get('/data/json/samples', samples),
                    web.get(r'/data/json/archive/{run}/{path:.+}', archive),
                    web.get(r'/plotfairy/archive/{run}/{path:.+}', render)])
    app.add_routes([web.static('/', '../data/', show_index=True)])

    # Start aiohttp!
    web.run_app(app, port='8889')

if __name__ == '__main__':
    DQMRenderer.initialize(workers=5)
    config_and_start_webserver()
