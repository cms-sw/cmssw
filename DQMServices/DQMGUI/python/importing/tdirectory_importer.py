
import mmap
import asyncio
from nanoroot import *
from reading.reading import TDirectoryReader
from data_types import MEInfo, ScalarValue, EfficiencyFlag, QTest

class TDirectoryImporter:

    # Don't import these (known obsolete/broken stuff)
    # The notorious SiStrip bad component workflow creates are varying number of MEs
    # for each run. We just hardcode-ban them here to help the deduplication
    __BLACKLIST = re.compile(b'By Lumi Section |/Reference/|BadModuleList')

    @classmethod
    async def get_mes_list(cls, file, dataset, run, lumi):
        """
        Returns a tuple of normalized ME path represented as binary string 
        and a corresponding MEInfo object.
        """

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

        mes = []
        with open(file, 'rb') as root_file:
            with mmap.mmap(root_file.fileno(), 0, prot=mmap.PROT_READ) as mm:
                tfile = TFile(mm, normalize=normalize, classes=dqm_classes)
                mes = cls.list_mes(tfile)
                error = "Problems on import" if tfile.error else None
                
        return mes


    @classmethod
    def list_mes(cls, tfile):
        """
        Returns a list of tuples: (me_path, me_info)
        These lists will be saved as separete blobs in the DB.
        """

        result = []

        for path, name, class_name, offset in tfile.fulllist():
            if cls.__BLACKLIST.search(path):
                continue
            
            if class_name == b'TObjString':
                parsed = TDirectoryReader.parse_string_entry(name)
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
