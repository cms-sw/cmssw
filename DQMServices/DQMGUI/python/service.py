"""
This service provides the logic of the API endpoints. It relies on other services to perform certain tasks.
"""

import re
import json
import struct
from collections import defaultdict

from async_lru import alru_cache

from .storage import GUIDataStore
from .helpers import get_api_error, binary_search, logged
from .rendering import GUIRenderer
from .importing.importing import GUIImportManager
from .data_types import Sample, RootDir, RootObj, RootDirContent, RenderingInfo

from .layouts.layout_manager import LayoutManager


class GUIService:

    store = GUIDataStore()
    renderer = GUIRenderer()
    import_manager = GUIImportManager()
    layouts_manager = LayoutManager()

    @classmethod
    @alru_cache(maxsize=10)
    @logged
    async def get_samples(cls, run, dataset, lumi=0):
        if run == '':
            run = None
        if dataset == '':
            dataset = None
        if lumi == '':
            lumi = None
        
        results = await cls.store.get_samples(run, dataset, lumi)
        return [Sample(*x) for x in results]


    @classmethod
    @alru_cache(maxsize=10)
    @logged
    async def get_archive(cls, run, dataset, path, search, lumi=0):
        """
        Returns a directory listing for run/lumi/dataset/path combination.
        Search is performed on ME name if it is provided.
        Layout property is the name of the layout the ME is coming from. If layout property is null, 
        it means that ME is not coming from a layout. 
        """

        # Path must end with a slash
        if path and not path.endswith('/'):
            path = path + '/'
        
        # Get a list of all MEs
        me_names = await cls.__get_me_names_list(dataset, run, lumi)
        if not me_names:
            return None

        # dir is a dict where key is subdir name and value is a count of 
        # how many MEs are inside that subdir (in all deeper levels)
        dirs = defaultdict(int)
        objs = set()

        # Prepare binary string in order not to decode every in linear scan
        path_binary = bytes(path, 'utf-8')
        regex_binary = re.compile(bytes(search, 'utf-8')) if search else None
        regex = re.compile(search) if search else None

        for me_name in me_names:
            # Check if ME name starts with requested path
            if me_name[:len(path_binary)] == path_binary:
                names = me_name[len(path_binary):].split(b'/')
                segment = names[0]
                is_file = len(names) == 1 # Last item in a path is file
                
                if regex_binary and not regex_binary.match(names[-1]):
                    continue # Regex is provided and ME name doesn't match it

                if b'\0' in me_name:
                    continue # This is a secondary item, not a main ME name

                segment = segment.decode('utf-8')

                if is_file:
                    objs.add(RootObj(name=segment, path=path + segment, layout=None))
                else:
                    dirs[segment] += 1

        # Add MEs from layouts
        # Layouts will be filtered against the search regex on their destination name.
        # Non existant sources will still be attempted to be displayed resulting in 
        # 'ME not found' string to be rendered.
        for layout in cls.layouts_manager.get_layouts():
            # Check if ME name starts with requested path
            if layout.destination[:len(path)] == path:
                names = layout.destination[len(path):].split('/')
                segment = names[0]
                is_file = len(names) == 1 # Last item in a path is file
                
                if regex and not regex.match(names[-1]):
                    continue # Regex is provided and ME name doesn't match it

                if is_file:
                    objs.add(RootObj(name=segment, path=layout.source, layout=layout.name))
                else:
                    dirs[segment] += 1

        # Transform dirs into a list of RootDir objects
        dirs = [RootDir(name=key, me_count=dirs[key]) for key in dirs]

        # Format results to a named tuple
        data = RootDirContent(dirs, objs)
        return data


    @classmethod
    @logged
    async def get_rendered_image(cls, me_descriptions, options):
        """options are defined here: data_types.RenderingOptions"""

        options.efficiency = False
        rendering_infos = []

        for me in me_descriptions:
            filename, fileformat, names_list, infos_list = await cls.__get_filename_fileformat_names_infos(me.dataset, me.run, me.lumi)

            if not filename:
                continue

            # Find the index of run/dataset/me in me list blob. 
            # The index in me list will correspond to the index in infos list.
            me_info = None
            index = binary_search(array=names_list, target=bytes(me.path, 'utf-8'))
            if index != -1:
                me_info = infos_list[index]
            else:
                continue
            
            # If efficiency flag is set for at least one of the MEs, it will be set for an overlay
            if not options.efficiency:
                efficiency_line = bytes('%s\0e=1' % me.path, 'utf-8')
                options.efficiency = binary_search(array=names_list, target=efficiency_line) != -1

            rendering_infos.append(RenderingInfo(filename=filename, fileformat=fileformat, path=me.path, me_info=me_info))

        if not rendering_infos: # No MEs were found
            if options.json:
                return None
            else:
                return await cls.renderer.render_string('ME not found', width=options.width, height=options.height)

        return await cls.renderer.render(rendering_infos, options)


    @classmethod
    @logged
    async def get_rendered_json(cls, me_descriptions, options):
        """
        Uses out of process renderer to get JSON representation of a ROOT object.
        Adds additional property to the resulting JSON called dqmProperties
        """

        data = await cls.get_rendered_image(me_descriptions, options)
        if data:
            data = data.decode('utf-8')
            obj = json.loads(data)
            obj['dqmProperties'] = { 'efficiency': options.efficiency }
            return obj
        else:
            return get_api_error(message='ME not found')


    @classmethod
    @logged
    async def get_available_lumis(cls, dataset, run):
        return await cls.store.get_lumis(dataset, run)


    @classmethod
    @logged
    async def register_samples(cls, samples):
        """Register a sample in DB. Samples array if of type SamplesFull."""

        # TODO: validate samples:
        # 1. Check if file format is one of the supported values
        # 2. Check if file and dataset are not empty
        await cls.import_manager.register_samples(samples)


    @classmethod
    @alru_cache(maxsize=10)
    @logged
    async def __get_me_names_list(cls, dataset, run, lumi=0):
        lines = await cls.store.get_me_names_list(dataset, run, lumi)

        if lines == None:
            # Import and retry
            success = await cls.import_manager.import_blobs(dataset, run, lumi)
            if success: 
                # Retry
                lines = await cls.store.get_me_names_list(dataset, run, lumi)

        return lines


    @classmethod
    @alru_cache(maxsize=10)
    @logged
    async def __get_filename_fileformat_names_infos(cls, dataset, run, lumi=0):
        filename_fileformat_names_infos = await cls.store.get_filename_fileformat_names_infos(dataset, run, lumi)

        if filename_fileformat_names_infos == None:
            # Import and retry
            success = await cls.import_manager.import_blobs(dataset, run, lumi)
            if success:
                # Retry
                filename_fileformat_names_infos = await cls.store.get_filename_fileformat_names_infos(dataset, run, lumi)

        if filename_fileformat_names_infos:
            filename, fileformat, names_list, infos_list = filename_fileformat_names_infos
            return (filename, fileformat, names_list, infos_list)
        else:
            return (None, None, None, None)
