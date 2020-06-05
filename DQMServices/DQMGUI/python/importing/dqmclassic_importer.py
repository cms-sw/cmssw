import re
import mmap
import asyncio
from DQMServices.DQMGUI import nanoroot
from ioservice import IOService
from reading.reading import DQMCLASSICReader
from data_types import MEInfo, ScalarValue, EfficiencyFlag, QTest


class DQMCLASSICImporter:

    # Don't import these (known obsolete/broken stuff)
    # The notorious SiStrip bad component workflow creates are varying number of MEs
    # for each run. We just hardcode-ban them here to help the deduplication
    __BLACKLIST = re.compile(b'By Lumi Section |/Reference/|BadModuleList')

    ioservice = IOService()

    @classmethod
    async def get_me_lists(cls, filename, dataset, run, lumi):
        """
        Returns a list which contains dicts. Keys of the dicts are (run, lumi) 
        tuples and values are lists of tuples (me_path, me_info). Full structure:
        [(run, lumi):[(me_path, me_info)]]
        me_path is normalized and represented as a binary string.
        We can return multiple (run, lumi) pairs because some file formats might 
        contain multiple runs/lumis in ine file.
        me_path, me_info will be saved as separete blobs in the DB.
        """

        buffer = await cls.ioservice.open_url(filename, blockcache=False)
        tfile = await nanoroot.TFile().load(buffer)
        result = await cls.list_mes(tfile)

        return { (run, 0): result }


    @classmethod
    async def list_mes(cls, tfile):
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

        result = []

        async for path, name, class_name, offset in tfile.fulllist(normalize=normalize, classes=dqm_classes):
            if cls.__BLACKLIST.search(path):
                continue
            
            if class_name == b'TObjString':
                parsed = DQMCLASSICReader.parse_string_entry(name)
                if isinstance(parsed, EfficiencyFlag):
                    item = (path + parsed.name + b'\0e=1', MEInfo(b'Flag'))
                elif isinstance(parsed, ScalarValue):
                    if parsed.type == b'i':
                        item = (path + parsed.name, MEInfo(b'Int', value = int(parsed.value.decode("ascii"))))
                    elif parsed.type == b'f':
                        item = (path + parsed.name, MEInfo(b'Float', value = float(parsed.value.decode("ascii"))))
                    else:
                        item = (path + parsed.name, MEInfo(b'XMLString', offset))
                else:
                    # QTest. Only save mename and qtestname, values need to be fetched later.
                    # Separate QTest name with \0 to prevent collisions with ME names.
                    item = (path + parsed.name + b'\0.' + parsed.qtestname, 
                        MEInfo(b'QTest', offset, qteststatus=int(parsed.status.decode("ascii"))))
            else:
                item = (path + name, MEInfo(class_name, offset))

            # Append an item to a final result list
            result.append(item)

        return result

    
    @classmethod
    def parse_filename(cls, full_path):
        """Splits full path to a TDirectory ROOT file and returns run number and dataset."""

        name = full_path.split('/')[-1]
        run = name[11:20].lstrip('0')
        dataset = '/'.join(name[20:-5].split('__'))

        return run, dataset
