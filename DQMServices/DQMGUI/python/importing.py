import re
import mmap
import glob
import zlib

from nanoroot import *
from meinfo import MEInfo
from storage import GUIDataStore
from data_types import SampleFull, EfficiencyFlag, ScalarValue

class GUIImporter:

    # Don't import these (known obsolete/broken stuff)
    # The notorious SiStrip bad component workflow creates are varying number of MEs
    # for each run. We just hardcode-ban them here to help the deduplication
    __BLACKLIST = re.compile(b'By Lumi Section |/Reference/|BadModuleList')
    __EOSPATH = '/eos/cms/store/group/comm_dqm/DQMGUI_data/Run*/*/R000*/DQM_*.root'

    store = GUIDataStore()

    @classmethod
    async def initialize(cls, root_file_location=__EOSPATH):
        """Imports all samples from given location of ROOT files no samples are present in the DB."""

        count = await cls.store.get_samples_count()
        if count == 0:
            await cls.__import_samples(root_file_location)
    

    @classmethod
    async def import_blobs(cls, run, dataset):
        """
        Imports ME list and ME offsets blobs into the database from a ROOT file.
        It is required to first import the blobs before samples can be used.
        """
        
        # Get the filename (if such sample exists)
        filename = await cls.store.get_sample_filename(run, dataset)
        if not filename: # Sample doesn't exist
            return False

        # Remove the folder structure that CMSSW adds
        # TODO: Check run numbers here?
        def normalize(parts):
            if len(parts) < 5 or parts[4] != b'Run summary':
                return b'<broken>' + b'/'.join(parts) + b'/'
            else:
                return b'/'.join((parts[3],) + (parts[5:]) + (b'',))
                
        # Only import these types
        def dqm_classes(name):
            return name in {
                b'TH1D',
                b'TH1F',
                b'TH1S',
                b'TH2D',
                b'TH2F',
                b'TH2S',
                b'TH3F',
                b'TObjString',
                b'TProfile',
                b'TProfile2D',
            }

        me_list = []
        with open(filename, 'rb') as root_file:
            with mmap.mmap(root_file.fileno(), 0, prot=mmap.PROT_READ) as mm:
                tfile = TFile(mm, normalize=normalize, classes=dqm_classes)
                me_list = list(cls.__recursive_list(tfile))
                error = "Problems on import" if tfile.error else None

        me_list_blob, offsets_blob = cls.__melist_to_blob(me_list)

        await cls.store.add_blobs(me_list_blob, offsets_blob, run, dataset, filename)

        return True


    @classmethod
    def __recursive_list(cls, tfile):
        for path, name, cls_name, offset in tfile.fulllist():
            if cls.__BLACKLIST.search(path):
                continue
            if cls_name == b'TObjString':
                # TODO: Move this method it out of MEInfo?
                thing = MEInfo.parsestringentry(name)
                if isinstance(thing, EfficiencyFlag): # efficiency flag on this ME
                    yield (path + thing.name + b'\0e=1', MEInfo(b'Flag'))
                elif isinstance(thing, ScalarValue):
                    # scalar ME.
                    if thing.type == b'i':
                        yield (path + thing.name, MEInfo(b'Int', value = int(thing.value.decode("ascii"))))
                    elif thing.type == b'f':
                        yield (path + thing.name, MEInfo(b'Float', value = float(thing.value.decode("ascii"))))
                    else:
                        yield (path + thing.name, MEInfo(b'XMLString', offset))
                else:
                    # QTest. Only save mename and qtestname, values need to be fetched later.
                    # Separate QTest name with \0 to prevent collisions with ME names.
                    yield (path + thing.name + b'\0.' + thing.qtestname, 
                        MEInfo(b'QTest', offset, qteststatus=int(thing.status.decode("ascii"))))
            else:
                # path position
                yield (path + name, MEInfo(cls_name, offset))


    @classmethod
    def __melist_to_blob(cls, me_list):
        me_list.sort()
        buf = b'\n'.join(key for key, _ in me_list)
        nameblob = zlib.compress(buf)
        infos = [info for _, info in me_list]
        infoblob = MEInfo.listtoblob(infos)
        return nameblob, infoblob


    @classmethod
    async def __import_samples(cls, root_file_location=__EOSPATH):
        """Imports all samples from given location of ROOT files."""

        # TODO: for quicker testing. Remove this!
        # root_file_location='/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2017/Cosmics/R0002946xx/*.root'

        print('Listing files for importing, this might take a few minutes...')
        files = glob.glob(root_file_location)
        print(f'Found {len(files)} files, importing...')

        # TODO: we might want to remove old versions of files here

        samples = []
        for file in files:
            run, dataset = cls.__parse_filename(file)
            samples.append(SampleFull(dataset=dataset, run=int(run), file=file))

        await cls.store.add_samples(samples)

    
    @classmethod
    def __parse_filename(cls, full_path):
        """Splits full path to ROOT file and returns run number and dataset."""

        name = full_path.split('/')[-1]
        run = name[11:20].lstrip('0')
        dataset = '/'.join(name[20:-5].split('__'))

        return run, dataset
