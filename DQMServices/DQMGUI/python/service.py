"""
This service provides the logic of the API endpoints. It relies on other services to perform certain tasks.
"""

import time
import struct

from functools import lru_cache
from async_lru import alru_cache
from collections import namedtuple

from rendering import GUIRenderer
from DQMServices.DQMGUI import nanoroot
from storage import GUIDataStore
from helpers import MERenderingInfo, PathUtil

from layouts.layout_manager import LayoutManager


# These named tuples will be returned from the service
Sample = namedtuple('Sample', ['run', 'dataset'])
RootDir = namedtuple('RootDir', ['name'])
RootObj = namedtuple('RootObj', ['name', 'path', 'layout'])
RootDirContent = namedtuple('RootDirContent', ['dirs', 'objs'])


class GUIService:

    store = GUIDataStore()
    renderer = GUIRenderer()
    layouts_manager = LayoutManager()

    @classmethod
    @alru_cache(maxsize=10)
    async def get_samples(cls, run, dataset):
        if run == '':
            run = None
        if dataset == '':
            dataset = None
        
        results = await cls.store.get_samples(run, dataset)
        return [Sample(*x) for x in results]


    @classmethod
    @alru_cache(maxsize=10)
    async def get_archive(cls, run, dataset, path, search):
        """
        Returns a directory listing for run/dataset/path combination.
        Search is performed on ME name if it is provided.
        Layout property is the name of the layout the ME is coming from. If layout property is null, 
        it means that ME is not coming from a layout. 
        """
        # Path must end with a slash
        if path and not path.endswith('/'):
            path = path + '/'
        
        # Get a list of all MEs
        lines = await cls.__get_melist(run, dataset)

        regex = re.compile(search) if search else None
        dirs = set()
        objs = set()

        path_util = PathUtil()

        # TODO: This is now a linear search over a sorted list: optimize this!!!
        for x in range(len(lines)):
            line = lines[x].decode("utf-8")
            path_util.set_path(line)
            subsequent_segment = path_util.subsequent_segment_of(path)
            if subsequent_segment:
                if regex and not regex.match(line.split('/')[-1]):
                    continue # Regex is provided and ME name doesn't match it

                if '\0' in line:
                    continue # This is a secondary item, not a main ME name

                if subsequent_segment.is_file:
                    objs.add(RootObj(name=subsequent_segment.name, path=path + subsequent_segment.name, layout=None))
                else:
                    dirs.add(RootDir(name=subsequent_segment.name))

        # Add MEs from layouts
        # Layouts will be filtered against the search regex on their destination name.
        # Non existant sources will still be attempted to be displayed resulting in 
        # 'ME not found' string to be rendered.
        for layout in cls.layouts_manager.get_layouts():
            path_util.set_path(layout.destination)
            subsequent_segment = path_util.subsequent_segment_of(path)
            if subsequent_segment:
                if regex and not regex.match(layout.destination.split('/')[-1]):
                    continue # Regex is provided and destination ME name doesn't match it

                if subsequent_segment.is_file:
                    objs.add(RootObj(name=subsequent_segment.name, path=layout.source, layout=layout.name))
                else:
                    dirs.add(RootDir(name=subsequent_segment.name))

        # Format results to a named tuple
        data = RootDirContent(dirs, objs)
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