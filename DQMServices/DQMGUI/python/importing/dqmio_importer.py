import asyncio
from collections import defaultdict
from ioservice import IOService
from compressing import GUIBlobCompressor
from data_types import MEInfo, ScalarValue
from nanoroot import TKey, TFile, TTreeFile, TType


class DQMIOImporter:

    ioservice = IOService()
    compressor = GUIBlobCompressor()

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
        
        dqmioschema = {
            b'Indices': {b'Run': TType.Int32, b'Lumi': TType.Int32, b'Type': TType.Int32,
                         b'FirstIndex': TType.Int64, b'LastIndex': TType.Int64},
            b'Ints':   {b'FullName': TType.String, b'Value': TType.Int64},
            b'Floats': {b'FullName': TType.String, b'Value': TType.Float64},
            b'Strings':{b'FullName': TType.String, b'Value': TType.IndexRange},
            b'TH1Fs':  {b'FullName': TType.String, b'Value': TType.IndexRange},
            b'TH1Ss':  {b'FullName': TType.String, b'Value': TType.IndexRange},
            b'TH1Ds':  {b'FullName': TType.String, b'Value': TType.IndexRange},
            b'TH2Fs':  {b'FullName': TType.String, b'Value': TType.IndexRange},
            b'TH2Ss':  {b'FullName': TType.String, b'Value': TType.IndexRange},
            b'TH2Ds':  {b'FullName': TType.String, b'Value': TType.IndexRange},
            b'TProfiles':   {b'FullName': TType.String, b'Value': TType.IndexRange},
            b'TProfile2Ds': {b'FullName': TType.String, b'Value': TType.IndexRange},
        }

        treenames = {
          0: b"Ints",
          1: b"Floats",
          2: b"Strings",
          3: b"TH1Fs",
          4: b"TH1Ss",
          5: b"TH1Ds",
          6: b"TH2Fs",
          7: b"TH2Ss",
          8: b"TH2Ds",
          9: b"TH3Fs",
          10: b"TProfiles",
          11: b"TProfile2Ds",
        }

        # Open file and preload all data -- we'll need to read everything anyways.
        buffer = await cls.ioservice.open_url(filename, blockcache=False)
        tfile = await TFile().load(buffer)
        trees = await TTreeFile(tfile, dqmioschema)

        # now, we'll iterate over all Indices, and read the MEs for each entry.
        # We sort them into a dict by run/lumi and then return them.
        entries = list(trees[b'Indices'])
        infos = defaultdict(list)

        # create a MEInfo object from whatever the value is.
        def createinfo(value, typeid):
            metype = cls.compressor.id_to_type[typeid]
            if metype in (b'Int', b'Float'):
                return MEInfo(metype, value=value)
            else:
                # value is IndexRange
                return MEInfo(metype, value.fSeekKey, value.start, value.end - value.start)

        for entry in entries:
            if entry[b'Lumi'] == 0:
                # Skip per run histograms
                continue

            # Value TTree for the type of this entry
            tree = trees[treenames[entry[b'Type']]]
            # first, read the names for this entry
            names = tree.branches[b'FullName'][entry[b'FirstIndex'] : entry[b'LastIndex']+1]
            # ...then the values...
            values = [createinfo(v, entry[b'Type']) for v in tree.branches[b'Value'][entry[b'FirstIndex'] : entry[b'LastIndex']+1]]
            # ... then turn all the iterators into a list and add them to the output set.
            infos[(entry[b'Run'], entry[b'Lumi'])] += list(zip(names, values))

        return infos
