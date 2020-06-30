import asyncio

from collections import defaultdict

from ..data_types import MEInfo
from ..compressing import GUIBlobCompressor
from ..nanoroot.io import XRDFile
from ..nanoroot.tfile import TFile
from ..nanoroot.ttree import TTreeFile, TType


class DQMIOImporter:

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

        run, lumi = int(run), int(lumi)
        
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

        # create a MEInfo object from whatever the value is.
        def createinfo(value, typeid):
            metype = cls.compressor.id_to_type[typeid]
            if metype in (b'Int', b'Float'):
                return MEInfo(metype, value=value)
            else:
                # value is IndexRange
                start, end, fSeekKey = value
                return MEInfo(metype, fSeekKey, start, end - start)

        # TODO: figure out proper caching: maybe it is not ideal to put all this
        # data into the main page cache. It is perfectly feasible to use an 
        # (uncached) XRDFile here.
        #buffer = await cls.ioservice.open_url(filename)
        buffer = await XRDFile().load(filename)
        tfile = await TFile().load(buffer)
        t = await TTreeFile().load(tfile, dqmioschema)

        # now, we'll iterate over all Indices, and read the MEs for each entry.
        # We sort them into a dict by run/lumi and then return them.

        infos = defaultdict(list)

        async for entry in await t.trees[b'Indices'][:]:
            if entry[b'Run'] != run or entry[b'Lumi'] != lumi:
                continue
            if entry[b'Type'] == 1000: # 1000 means no data
                continue
            # Value TTree for the type of this entry
            tree = t.trees[treenames[entry[b'Type']]]
            namebranch = tree.branches[b'FullName']
            valuebranch = tree.branches[b'Value']
            firstindex = entry[b'FirstIndex']
            lastindex = entry[b'LastIndex'] + 1 # DQMIO uses *inclusive* upper.
            # first, read the names for this entry
            names = [name async for name in await namebranch[firstindex : lastindex]]
            # ...then the values...
            values = [createinfo(v, entry[b'Type']) async for v in await valuebranch[firstindex : lastindex]]
            # ... finally pair up the results. There may be more than one entry per run/lumi.
            # infos += list(zip(names, values))
            infos[(entry[b'Run'], entry[b'Lumi'])] += list(zip(names, values))

        await buffer.close()
        return infos
