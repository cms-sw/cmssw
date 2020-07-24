import glob
import asyncio

from concurrent.futures import ProcessPoolExecutor

from ..helpers import logged
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

    @classmethod
    async def initialize(cls, files=__EOSPATH):
        """
        Imports all samples from given ROOT files if no samples are present in the DB.
        Format is assumed to be TDirectory.
        If files is a string, it is globed to get the list of files.
        If files is a list, all these files gets imported.
        """
        
        if await cls.store.is_samples_empty():
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
        samples, blob_descriptions = await cls.import_in_worker(fileformat, filename, dataset, run, lumi)

        await cls.store.register_samples(samples)
        for blob_description in blob_descriptions:
            # import_sync prepares everything
            await cls.store.add_blobs(**blob_description)

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
    @logged
    async def import_in_worker(cls, fileformat, filename, dataset, run, lumi):
        """
        This function will call `import_sync` using a process pool.
        """

        # TODO: Semaphore here to not open too many workers at once?
        with ProcessPoolExecutor(1) as executor:
            return await asyncio.get_event_loop().run_in_executor(executor,
                cls.import_sync, fileformat, filename, dataset, run, lumi)


    @classmethod
    def import_sync(cls, fileformat, filename, dataset, run, lumi):
        """
        This function should be called in a different process (via multiprocessing)
        to actually perform the import.
        """
        # This must be a top-level, sync,  public function, so multiprocessing 
        # can import this module on the (clean, forked before initalization)
        # worker process and then call it using the pickle'd arguments.

        # To call the async function, we set up and tear down an event loop just
        # for this one call. The worker is left 'clean', without anything running.
        return asyncio.run(cls.import_async(fileformat, filename, dataset, run, lumi))


    # But we need an async function to call async stuff, so here it is.
    @classmethod
    async def import_async(cls, fileformat, filename, dataset, run, lumi):
        """
        This function performs the actual import work, inside a dedicated main
        loop in a process pool worker.
        """
        importer = cls.__pick_importer(fileformat)
        mes_lists = await importer.get_me_lists(filename, dataset, run, lumi)

        # It's possible that some samples that exists in this file were not yet
        # registered as samples (through register API endpoint). So we (re)create 
        # all samples that we have found
        samples = [SampleFull(dataset, run=int(key[0]), lumi=int(key[1]), file=filename, fileformat=fileformat) for key in mes_lists]
        
        blob_descriptions = []

        for key in mes_lists:
            mes = mes_lists[key]
            mes.sort()

            # Separate lists
            names_list = b'\n'.join(name for name, _ in mes)
            infos_list = [info for _, info in mes]

            # these go straight into GUIDataStore.add_blobs(...), but back in the main process.
            blob_descriptions.append({
                # Compress blobs
                'names_blob': await cls.compressor.compress_names_list(names_list),
                'infos_blob': await cls.compressor.compress_infos_list(infos_list),
                'dataset': dataset,
                'filename': filename,
                'run': key[0],
                'lumi': key[1],
            })
        return samples, blob_descriptions
        
