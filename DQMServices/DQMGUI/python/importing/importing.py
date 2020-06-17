import glob
import asyncio

from concurrent.futures import ProcessPoolExecutor

from ..storage import GUIDataStore
from ..data_types import FileFormat, SampleFull
from ..compressing import GUIBlobCompressor

from .dqmio_importer import DQMIOImporter
from .dqmclassic_importer import DQMCLASSICImporter
from .protobuf_importer import ProtobufImporter


class GUIImportManager:
    """
    This is a ROOT file importing manager. 
    It picks the correct importer based on a specifc file format that 
    has to be imported and delegates actual importing to it.
    """

    __EOSPATH = '/eos/cms/store/group/comm_dqm/DQMGUI_data/Run*/*/R000*/DQM_*.root'
    __EOSPREFIX = "root://eoscms.cern.ch/"

    store = GUIDataStore()
    compressor = GUIBlobCompressor()
    # this is a global instance, created before loading any modules.
    executor = None

    @classmethod
    async def initialize(cls, files=__EOSPATH, executor=None):
        """
        Imports all samples from given ROOT files if no samples are present in the DB.
        Format is assumed to be TDirectory.
        If files is a string, it is globed to get the list of files.
        If files is a list, all these files gets imported.
        """
        cls.executor = executor

        count = await cls.store.get_samples_count()
        if count == 0:
            if files == None:
                files = cls.__EOSPATH

            if isinstance(files, str):
                print('Listing files for importing, this might take a few minutes...')
                files = glob.glob(files)
            
            print(f'Found {len(files)} files, importing...')
        
            importer = cls.__pick_importer(FileFormat.DQMCLASSIC)
            samples = []

            # Parse filenames to get the metadata
            for file in files:
                run, dataset = importer.parse_filename(file)
                samples.append(SampleFull(dataset=dataset, run=int(run), lumi=0, file=cls.__EOSPREFIX + file, fileformat=FileFormat.DQMCLASSIC))

            await cls.register_samples(samples)

    @classmethod
    async def destroy(cls):
        pass


    @classmethod
    async def register_samples(cls, samples):
        """No need to go to an importer as it's format agnostic. Samples array is of type SamplesFull."""

        await cls.store.register_samples(samples)


    @classmethod
    async def import_blobs(cls, dataset, run, lumi=0):
        """
        Imports ME list and ME info blobs into the database from a ROOT file.
        It is required to first import the blobs before samples can be used.
        """

        filename, fileformat = await cls.store.get_sample_file_info(dataset, run, lumi)
        if not filename: # Sample doesn't exist
            return False
        
        # delegate the hard work to a process pool.
        if cls.executor:
            mes_lists = await asyncio.get_event_loop().run_in_executor(cls.executor,
                cls.import_sync, fileformat, filename, dataset, run, lumi)
        else:
            assert False, "Could run in-process here but we don't need that."

        # It's possible that some samples that exists in this file were not yet
        # registered as samples (through register API endpoint). So we (re)create 
        # all samples that we have found
        samples = [SampleFull(dataset, run=int(key[0]), lumi=int(key[1]), file=filename, fileformat=fileformat) for key in mes_lists]
        await cls.store.register_samples(samples)

        for key in mes_lists:
            mes = mes_lists[key]
            mes.sort()

            # Separate lists
            names_list = b'\n'.join(name for name, _ in mes)
            infos_list = [info for _, info in mes]

            # Compress blobs
            names_blob = await cls.compressor.compress_names_list(names_list)
            infos_blob = await cls.compressor.compress_infos_list(infos_list)

            await cls.store.add_blobs(names_blob, infos_blob, dataset, filename, run=key[0], lumi=key[1])

        return True


    @classmethod
    def __pick_importer(cls, file_format):
        """
        Picks the correct importer based on the file format that's being imported.
        If a new file format are added, an importer has to be registered in this method.
        """

        if file_format == FileFormat.DQMCLASSIC:
            return DQMCLASSICImporter()
        elif file_format == FileFormat.DQMIO:
            return DQMIOImporter()
        elif file_format == FileFormat.PROTOBUF:
            return ProtobufImporter()
        return None

    @classmethod
    def import_sync(cls, fileformat, filename, dataset, run, lumi):
        """
        This function should be called in a different process (via multiprocessing)
        to actually perform the import.
        """
        importer = cls.__pick_importer(fileformat)
        mes_lists = asyncio.run(importer.get_me_lists(filename, dataset, run, lumi))
        return mes_lists
        
