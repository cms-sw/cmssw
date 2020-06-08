import glob
from storage import GUIDataStore
from compressing import GUIBlobCompressor
from data_types import FileFormat, SampleFull
from importing.dqmclassic_importer import DQMCLASSICImporter
from importing.dqmio_importer import DQMIOImporter


class GUIImportManager:
    """
    This is a ROOT file importing manager. 
    It picks the correct importer based on a specifc file format that 
    has to be imported and delegates actual importing to it.
    """

    __EOSPATH = '/eos/cms/store/group/comm_dqm/DQMGUI_data/Run*/*/R000*/DQM_*.root'

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
                samples.append(SampleFull(dataset=dataset, run=int(run), lumi=0, file=file, fileformat=FileFormat.DQMCLASSIC))

            await cls.register_samples(samples)


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
        
        importer = cls.__pick_importer(fileformat)
        mes_lists = await importer.get_me_lists(filename, dataset, run, lumi)

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
        return None
