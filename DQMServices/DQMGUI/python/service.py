import time
import struct

from functools import lru_cache
from async_lru import alru_cache

from rendering import GUIRenderer
from DQMServices.DQMGUI import nanoroot
from storage import GUIDataStore
from helpers import MERenderingInfo

class GUIService:

    store = GUIDataStore()
    renderer = GUIRenderer()

    @classmethod
    @alru_cache(maxsize=10)
    async def get_samples(cls, run, dataset):
        if run == '':
            run = None
        if dataset == '':
            dataset = None
        
        results = await cls.store.get_samples(run, dataset)

        # TODO: offline_data should probably be removed. No GUI flavor distinction is required.
        samples = {
            'samples': [{
                'type': 'offline_data',
                'items': [{
                    'run': row[0],
                    'dataset': row[1]
                } for row in results]
            }]
        }

        return samples

    @classmethod
    @alru_cache(maxsize=10)
    async def get_archive(cls, run, dataset, path, search):

        def get_subitem(full_path, current_dir):
            """Returns a closest sub item of full_path inside current_dir
            The second value of the response tuple indicates if that closest sub item is final one
            i.e. if it is a file or a directory.
            If full_path is a/b/c/d/file and current_dir is /a/b/c function will return (d, False).
            If full_path is a/b/c/d/file and current_dir is /a/b/c/d function will return (file, True).
            If current_dir is not part of full_path, function will return False.
            """

            # Remove double slashes
            full_path = full_path.replace('//', '/')
            current_dir = current_dir.replace('//', '/')

            if full_path.startswith(current_dir):
                names = full_path.replace(current_dir, '').split('/')
                if len(names) == 1: # This is an ME
                    return (names[0], True)
                else: # This is a folder
                    return (names[0], False)
            else:
                return False

        # Path must end with a slash
        if path and not path.endswith('/'):
            path = path + '/'
        
        # Get a list of all MEs
        lines = await cls.__get_melist(run, dataset)

        regex = re.compile(search) if search else None
        dirs = set()
        objs = set()

        # TODO: This is now a linear search over a sorted list: optimize this!!!
        for x in range(len(lines)):
            line = lines[x].decode("utf-8")
            subitem = get_subitem(line, path)
            if subitem:
                if regex and not regex.match(line.split('/')[-1]):
                    continue # Regex is provided and ME name doesn't match it
                
                if '\0' in line:
                    continue # This is a secondary item, not a main ME name

                if subitem[1]:
                    objs.add(subitem[0])
                else:
                    dirs.add(subitem[0])

        # Remove last slash before returning
        path = path[:-1]

        # Put results to a list to be returned
        data = {'contents': []}
        data['contents'].extend({'subdir': x} for x in dirs)
        data['contents'].extend({'obj': x, 'dir': path} for x in objs)

        return data


    @classmethod
    async def get_rendered_image(cls, me_descriptions, width=266, height=200, stats=True, normalize=True, error_bars=False):
        efficiency = False
        rendering_infos = []

        for me in me_descriptions:
            filename, me_list, me_infos = await cls.__get_filename_melist_offsets(me.run, me.dataset)

            # Find the index of run/dataset/me in me list blob. 
            # The index in me list will correspond to the index in offsets list.
            # TODO: use binary search or smth!!!
            me_info = None
            for i in range(len(me_list)):
                line = me_list[i].decode("utf-8")
                if line == me.path:
                    me_info = me_infos[i]
                    break
            else: # We will end up here if we finish the loop without breaking out
                continue
            
            # If efficiency flag is set for at least one of the MEs, it will be set for an overlay
            if not efficiency:
                efficiency = bytes('%s\0e=1' % me.path, 'utf-8') in me_list

            rendering_infos.append(MERenderingInfo(filename=filename, path=me.path, me_info=me_info))

        if not rendering_infos: # No MEs were found
            return await cls.renderer.render_string('ME not found', width=width, height=height)

        return await cls.renderer.render(rendering_infos, width, height, efficiency, stats, normalize, error_bars)


    @classmethod
    @alru_cache(maxsize=10)
    async def __get_melist(cls, run, dataset):
        lines = await cls.store.get_me_list_blob(run, dataset)
        return lines


    @classmethod
    @alru_cache(maxsize=10)
    async def __get_filename_melist_offsets(cls, run, dataset):
        filename, me_list, me_infos = await cls.store.get_blobs_and_filename(run, dataset)
        return (filename, me_list, me_infos)