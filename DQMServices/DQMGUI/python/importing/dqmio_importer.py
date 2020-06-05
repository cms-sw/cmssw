import asyncio
from ioservice import IOService
from data_types import MEInfo, ScalarValue
from nanoroot import TKey, TFile, TTreeFile, TType
# from nanoroot.TType import Int32, Int64, Float64, String, IndexRange


class DQMIOImporter:

    @classmethod
    async def get_mes_list(cls, file, dataset, run, lumi):
        """
        Returns a tuple of normalized ME path represented as binary string 
        and a corresponding MEInfo object.
        """
        
        dqmioschema = {
            b'Indices': {b'Run': TType.Int32, b'Lumi': TType.Int32, b'Type': TType.Int32,
                         b'FirstIndex': TType.Int64, b'LastIndex': TType.Int64},
            b'Ints':   {b'FullName': String, b'Value': TType.Int64},
            b'Floats': {b'FullName': String, b'Value': TType.Float64},
            b'Strings':{b'FullName': String, b'Value': TType.IndexRange},
            b'TH1Fs':  {b'FullName': String, b'Value': TType.IndexRange},
            b'TH1Ss':  {b'FullName': String, b'Value': TType.IndexRange},
            b'TH1Ds':  {b'FullName': String, b'Value': TType.IndexRange},
            b'TH2Fs':  {b'FullName': String, b'Value': TType.IndexRange},
            b'TH2Ss':  {b'FullName': String, b'Value': TType.IndexRange},
            b'TH2Ds':  {b'FullName': String, b'Value': TType.IndexRange},
            b'TProfiles':   {b'FullName': String, b'Value': TType.IndexRange},
            b'TProfile2Ds': {b'FullName': String, b'Value': TType.IndexRange},
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
        trees = TTreeFile(tfile, dqmioschema)

        # now, we'll iterate over all Indices, and read the MEs for each entry.
        # We sort them into a dict by run/lumi and then return them.
        entries = list(trees[b'Indices'])
        infos = defaultdict(list)
                
        # create a MEInfo object from whatever the value is.
        def createinfo(value, typeid):
            metype = MEInfo.idtotype[typeid]
            if metype in (b'Int', b'Float'):
                return MEInfo(metype, value = value)
            else:
                # value is IndexRange
                return MEInfo(metype, value.fSeekKey, value.start, value.end - value.start)

        for e in entries:
            # Value TTree for the type of this entry
            tree = trees[treenames[e[b'Type']]]
            # first, read the names for this entry
            names = tree.branches[b'FullName'][e[b'FirstIndex'] : e[b'LastIndex']+1]
            # ...then the values...
            values = [createinfo(v, e[b'Type']) for v in tree.branches[b'Value'][e[b'FirstIndex'] : e[b'LastIndex']+1]]
            # ... then turn all the iterators into a list and add them to the output set.
            infos[(e[b'Run'], e[b'Lumi'])] += list(zip(names, values))

        return infos
