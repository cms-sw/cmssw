import zlib
import struct
import asyncio
from collections import namedtuple

# If uproot is micro-Python ROOT, this is Nano ROOT.
# These classes allow reading the ROOT contianer structures (TFile, TDirectory, TKey),
# but not actually decoding ROOT serialized objects.
# The reason to use this rather than uproot is much higher performance on TDirectory's,
# as well as fewer dependencies.
# Everything is designed around a buffer like that supports indexing to get bytes. 
# Reading from the buffer is rather lazy and copying data is avoided as long as possible.
# This library supports asyncio; that is another reason for its existence. To acheive
# that, many methods are async, and the buffers have an *async* [...:...] operator.
# Though, when buffers read from file already are handled (e.g. all the TTree code), we
# use normal, sync methods and normal `bytes` as a buffer type. This is not very 
# consistent, but pragmatic.
# To work around not having async constructors, the common pattern is await Thing().load(...).

# First, a sample implementation of the 'async buffer' interface. This is not really
# needed, except for testing. Not recommended for practical use, since it is *not*
# actually async.

import mmap
class MMapFile:
    def __init__(self, url):
        self.file = open(url, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
    def __len__(self):
        return len(self.mm)
    async def __getitem__(self, idx):
        # This is quite inefficient: this *will* block the main thread.
        return mm[idx]

# Then, the basic structures to read a ROOT file: TFile and TKey.
# We don't actually need TDirectory, since the directory structure can be inferred
# from the TKeys (though reading the TDirectories *might* be more efficient).

class TFile:
    # Structure from here: https://root.cern.ch/doc/master/classTFile.html
    Fields = namedtuple("TFileFields", ["root", "fVersion", "fBEGIN", "fEND", "fSeekFree", "fNbytesFree", "nfree", "fNbytesName", "fUnits", "fCompress", "fSeekInfo", "fNbytesInfo", "fUUID_low", "fUUID_high"])
    structure_small = struct.Struct(">4sIIIIIIIbIIIQQ")
    structure_big   = struct.Struct(">4sIIQQIIIbIQIQQ")

    
    # These two are default transforms that can be overwritten.
    topath = lambda parts: b'/'.join(parts) + b'/'
    allclasses = lambda name: True

    # The TFile datastructure is pretty boring, the only thing we really need
    # is the address of the first TKey, which is actually hardcode to 100...
    # But this class also provides some caching to make computing full paths
    # fast, and the `fulllist` method that lists all objects using this.
    # Its behaviour can be adjusted with the `normalize` and `classes` 
    # callback functions.
    async def load(self, buf):
        self.buf = buf
        self.fields = TFile.Fields(*TFile.structure_small.unpack(
            await self.buf[0:TFile.structure_small.size]))
        if self.fields.fVersion > 1000000:
            self.fields = TFile.Fields(*TFile.structure_big.unpack(
                await self.buf[0:TFile.structure_big.size]))
        assert self.fields.root == b'root'
        self.error = False
        return self
        
    def __repr__(self):
        return f"TFile({self.buf}, fields = {self.fields})"

    # First TKey in the file. They are sequential, use `next` on the key to
    # get the next key.
    async def first(self):
        return await TKey().load(self.buf, self.fields.fBEGIN, self.end())

    def end(self):
        if len(self.buf) < self.fields.fEND:
            self.error = True
            print(f"TFile corrupted, fEND ({self.fields.fEND}) behind end-of-file ({len(self.buf)})")
            return len(self.buf)
        return self.fields.fEND

    # Returns a generator producing (path, name, class, offset) tuples.
    # The paths are normalized with the `normalize` callback, the classes 
    # filtered with the `classes` callback.
    # use `async for` to iterate this.
    async def fulllist(self, normalize = topath, classes = allclasses):
        # recursive list of path fragments with caching, indexed by fSeekKey
        dircache = dict()
        dircache[0] = ()
        async def fullname(fSeekKey):
            # fast path if in cache
            if fSeekKey in dircache:
                return dircache[fSeekKey]
            # else load the TKey...
            k = await TKey().load(self.buf, fSeekKey)
            parent = k.fields.fSeekPdir
            # ... and recurse to its parent.
            res = await fullname(parent) + (await k.objname(),)
            dircache[fSeekKey] = res
            return res
        
        # normalized dirname for each dir identified by its fSeekKey
        normalizedcache = dict()
        async def normalized(fSeekKey):
            if fSeekKey in normalizedcache:
                return normalizedcache[fSeekKey]
            parts = await fullname(fSeekKey)
            res = normalize(parts)
            normalizedcache[fSeekKey] = res
            return res

        key = await self.first()
        while key:
            c = await key.classname()
            if classes(c):
                yield (await normalized(key.fSeekKey), await key.objname(), c, key.fSeekKey)
            n = await key.next()
            if key.error:
                self.error = True
            key = n

        
class TKey:
    # Structure also here: https://root.cern.ch/doc/master/classTFile.html
    Fields = namedtuple("TKeyFields", ["fNbytes", "fVersion", "fObjLen", "fDatime", "fKeyLen", "fCycle", "fSeekKey", "fSeekPdir"])
    structure_small = struct.Struct(">iHIIHHII")
    structure_big   = struct.Struct(">iHIIHHQQ")
    sizefield = struct.Struct(">i")
    compressedheader = struct.Struct("2sBBBBBBB")

    # Decode key at offset `fSeekKey` in `buf`. `end` can be the file end
    # address if it is less then the buffer end.
    async def load(self, buf, fSeekKey, end = None):
        self.buf = buf
        self.end = end if end != None else len(buf)
        self.fSeekKey = fSeekKey
        self.fields = TKey.Fields(*TKey.structure_small.unpack(
            await self.buf[self.fSeekKey : self.fSeekKey + TKey.structure_small.size]))
        self.headersize = TKey.structure_small.size
        if self.fields.fVersion > 1000:
            self.fields = TKey.Fields(*TKey.structure_big.unpack(
                await self.buf[self.fSeekKey : self.fSeekKey + TKey.structure_big.size]))
            self.headersize = TKey.structure_big.size
        assert self.fields.fSeekKey == self.fSeekKey, f"{self} is corrupted!"
        self.error = False
        return self

    def __repr__(self):
        return f"TKey({self.buf}, {self.fSeekKey}, fields = {self.fields})"

    # Read the TKey following this key. According to the documentation these
    # should be one after the other in the file, but in practice there are
    # sometimes gaps (resized/deleted objects?), which are skipped here.
    # If we still fail to read the next item, we try to find the next key
    # by searching for a familiar class name.
    async def next(self):
        offset = self.fields.fSeekKey + self.fields.fNbytes
        while (offset+TKey.structure_small.size) < self.end:
            # It seems that a negative length indicates an unused block of that size. Skip it.
            # The number of such blocks matches nfree in the TFile.
            size, = TKey.sizefield.unpack(await self.buf[offset:offset+4])
            if size < 0:
                offset += -size
                continue
            k = await TKey().load(self.buf, offset, self.end)
            return k
        return None

    async def _getstr(self, pos):
        size = await self.buf[pos]
        if size == 255: # soultion for when length does not fit one byte
            size, = TKey.sizefield.unpack(await self.buf[pos+1:pos+5])
            pos += 4
        nextpos = pos + size + 1
        value = await self.buf[pos+1:nextpos]
        return value, nextpos

    # Parse the three strings in the TKey (classname, objname, objtitle)
    async def names(self):
        pos = self.fSeekKey + self.headersize
        classname, pos = await self._getstr(pos)
        objname, pos = await self._getstr(pos)
        objtitle, pos = await self._getstr(pos)
        return classname, objname, objtitle

    async def classname(self):
        return (await self._getstr(self.fSeekKey + self.headersize))[0]

    async def objname(self):
        # optimized self.names()[1]
        pos = self.fSeekKey + self.headersize
        pos += await self.buf[pos] + 1
        if await self.buf[pos] == 255:
            size, = TKey.sizefield.unpack(await self.buf[pos+1:pos+5])
            pos += 4
            nextpos = pos + size
        else:
            nextpos = pos + await self.buf[pos]
        return await self.buf[pos+1:nextpos+1]
    
    def compressed(self):
        return self.fields.fNbytes - self.fields.fKeyLen != self.fields.fObjLen

    # Return and potentially decompress object data.
    # Compression is done in thraed pool since it could take more time.
    async def objdata(self):
        start = self.fields.fSeekKey + self.fields.fKeyLen
        end = self.fields.fSeekKey + self.fields.fNbytes
        if not self.compressed():
            return await self.buf[start:end]
        else:
            def decompress(buf, start, end):
                out = []
                while start < end:
                     # Thanks uproot!
                     algo, method, c1, c2, c3, u1, u2, u3 = TKey.compressedheader.unpack(
                         buf[start : start + TKey.compressedheader.size])
                     compressedbytes = c1 + (c2 << 8) + (c3 << 16)
                     uncompressedbytes = u1 + (u2 << 8) + (u3 << 16)
                     start += TKey.compressedheader.size
                     assert algo == b'ZL', "Only Zlib compression supported, not " + repr(comp)
                     uncomp =  zlib.decompress(buf[start:start+compressedbytes])
                     out.append(uncomp)
                     assert len(uncomp) == uncompressedbytes
                     start += compressedbytes
                return b''.join(out)
            buf = await self.buf[start:end]
            return await asyncio.get_event_loop().run_in_executor(None, decompress, buf, 0, end-start)

    # Each key (except the root) has a parent directory.
    # Creates a new TKey pointing there.
    async def parent(self):
        if self.fields.fSeekPdir == 0:
            return None
        return await TKey().load(self.buf, self.fields.fSeekPdir, self.end)

    # Derive the full path of an object by recursing up the parents.
    # slow, primarily for debugging.
    async def fullname(self):
        parent = await self.parent()
        parentname = await parent.fullname() if parent else b''
        return b"%s/%s" % (parentname, await self.objname())

# Next, a helper to format TBufferFile data. All data in ROOT files is serialized
# using TBufferFile serialization, but the headers aded to the data vary. This
# class removes any detected headers and adds the headers needed for a bare object
# that can be read using `ReadObjectAny`. Since TBufferFile data can contain
# references into itself, we need to keep track of where the buffer actually 
# started (`displacement` parameters).

class TBufferFile():
    def __init__(self, objdata, classname, displacement = 0,version = None):
        if objdata[0:1] != b'@' and objdata[1:2] == b'T': 
            # This came out of a branch (TBranchObject?) with class and version header.
            clslen = objdata[0]
            cls = objdata[1:1+clslen]
            assert cls == classname, f"Classname {repr(cls)} from Branch should match {repr(classname)}"
            objdata = objdata[clslen+2:] # strip class and continue.
            displacement -= clslen + 2
        if objdata[0:1] == b'@':
            # @-decode and see if that could be a version header.
            size, = struct.unpack(">I", objdata[0:4])
            size = (size & ~0x40000000) + 4
            if size != len(objdata):
                # this does not look like a version header. Add one.
                totlen = 2 + len(objdata)
                head = struct.pack(">IH", totlen | 0x40000000, version)
                objdata = head + objdata
                displacement += len(head)
        else:
            assert False, "No known header found, TBufferFile wrapping would probably fail."
        # The format is <@length><kNewClassTag=0xFFFFFFFF><classname><nul><@length><2 bytes version><data ...
        # @length is 4byte length of the *entire* remaining object with bit 0x40 (kByteCountMask)
        # set in the first (most significant) byte. This prints as "@" in the dump...
        # the data inside the TKey seems to have the version already.
        totlen = 4 + len(classname) + 1 + len(objdata)
        head = struct.pack(">II", totlen | 0x40000000, 0xFFFFFFFF)
        self.buffer =  head + classname + b'\0' + objdata
        displacement += len(head) + len(classname) + 1

        # The TBufferFile data can contain references into itself, when an 
        # already stored object is used again. This happens especially with
        # class definitions, which can also be re-used. Since these offsets are
        # absolute offsets into the buffer, they will be wrong if we add or
        # remove to/from the beginning of the buffer. To compensate for that,
        # we can use the `SetDisplacement` option, but we need to keep track of
        # the correct displacement here.
        # Getting it wrong usually does not matter, but can lead to missing
        # Objects (axis, labels, ...) inside the histograms and also sometimes
        # ROOT crashes due to infinite recursion.
        self.displacement = displacement


# Then, TTree IO. This is more fragile than the TDirectory code.

# Some minimal TTree IO. The structure of a TTree is not terribly complicated:
# A TTree (table) consists of TBranches (columns). Each TBranch stores its values
# in a list of TBaskets, which contain a variable number of entries each.
# The TBaskets are stored in TKeys in the top level of the directory structure.
# All other info about the TTree (TBranches, TLeafs) is serialized in the
# TTree object in a TKey in the directory.
# Decoding that TTree blob is not easy. But we don't have to: Just reading the
# TBranch keys tells us the TBranch name (name) and TTree name (title) they
# belong to. What remains to be figured out is how many entries there are in
# the TBasket and what the first ID in this TBasket is. For the latter, we will
# just assume that the TBaskets are stored in the file in the order of their
# start IDs; this is not guaranteed but seems to hold. (It seems we could also
# use `fCycle` instead of the file order, not sure if it is better).
# To find how many objects there are in a TBasket, there are two cases: 1. it
# holds fixed size objects, then it's just size/object size, or 2. if holds
# variable size objects. In this case, there is an array at the end of the 
# basket data that provides the offset inside the basket data where the n'th
# object starts. By reading this array, we can know how many entries there are
# and where they are.
# The only problem is that we don't know where this arrray starts (that would
# be encoded somewhere in the TTree object), so we have to read the data from
# the back, heuristically guessing where it ends... but we have a good chance
# that this works: the array header is the length of the array, and we know
# where the first object starts, so we have two consistency checks.
# The other problem is that we don't know the type of the TBaskets (we'd need
# to read the TLeafs from the TTree blob for that), so we require an explicit
# schema from the user.

# First, the schema. All schema types are in the TType namespace.
# We use struct.Struct for fixed size types, and objects with a `size` of None
# and a `unpack` method for variable size types.

class TType:
    Int32 = struct.Struct(">i")
    Int64 = struct.Struct(">q")
    Float64 = struct.Struct(">d")

    class String:
        # This signals "variable size" later.
        size = None

        @staticmethod
        def unpack(buf, start, end, basket):
            size = buf[start]
            start += 1
            if size == 0xFF:
                size, = TType.Int32.unpack(buf[start:start+4])
                start += 4
            s = buf[start:end]
            assert len(s) == size, f"Size of string does not match buffer size: {repr(buf[start:end])}, {size})"
            return s
    
    class IndexRange:
        # This tells the TBranch to not keep the uncompressed data. To actually get
        # the data, the buffer has to be reconstructed using the seekkey and TKey.
        dropdata = True
        size = None

        def __init__(self, start, end, fSeekKey):
            self.start = start
            self.end = end
            self.fSeekKey = fSeekKey # address of the basket this came from

        # To actually retrieve the object, we go back to the original file buffer, unpack the
        # basket and then read the object from there, without needing any of the TTree code.
        async def read(self, buf):
            return (await TKey().load(buf, self.fSeekKey)).objdata[self.start:self.end]

        def __repr__(self):
            return f"IndexRange(start={self.start}, end={self.end}, fSeekKey={self.fSeekKey})"

        @staticmethod
        def unpack(buf, start, end, basket):
            # return object here, so we can later extract pointers pointing *into* the basket.
            # To make that possible, API needs to diverge from Struct.unpack...
            # The basket also passes itself in so we can keep whatever info we need.
            return TType.IndexRange(start, end, basket.fSeekKey)

    class ObjectBlob:
        size = None

        # This simply returns the bytes stored in the tree, without any decoding.
        @staticmethod
        def unpack(buf, start, end, basket):
            return buf[start:end]
    
# A schema for a TTree is a dict mapping branch names to types.
# A schema for a file is a dict mapping TTree names to TTree schemas.

# A TBasket is pretty self-contained, we can read it from a TKey and its type.
# The only thing it does not know is its starting index in the full table.
# We don't save stuff that is not absolutely needed, like names or a ref to the
# file (we do keep a copy of the (decompressed) data, unless dropdata is set).
# `dropdata` is useful since we always need to read full files in the first 
# pass, and a full 2GB file decompressed might not fit in memory. Later, we can
# decode individual baskets and extract objects using the fields of `IndexRange`.
class TBasket:
    Index = struct.Struct(">i")
    
    async def load(self, tkey, ttype, startindex = None):
        assert await tkey.classname() == b'TBasket'
        self.fKeyLen = tkey.fields.fKeyLen
        self.fSeekKey = tkey.fields.fSeekKey # we don't really need that, but keep it for later
        self.ttype = ttype
        self.startindex = startindex
        self.buf = await tkey.objdata() # read and decompress once.
        if ttype.size != None:
            self.initfixed()
        else:
            self.initvariable()
        if hasattr(ttype, 'dropdata') and ttype.dropdata:
            del self.buf
            self.buf = None
        return self

    def initfixed(self):
        # this case is easy -- just cut the buffer into fixed-size records
        self.offsets = [i*self.ttype.size for i in range(len(self.buf)//self.ttype.size + 1)]
        
    def initvariable(self):
        # this case is more complicated: there is a data structure at the end
        # of the buffer that encodes where the objects start and end, but
        # decoding that without the help of the TBranch metadata is complicated.
        buf = self.buf
        pos = len(buf) - 4
        def getint(pos):
            i, = TBasket.Index.unpack(buf[pos:pos+4])
            return i
        assert getint(pos) == 0
        pos -= 4
        prev = getint(pos)
        offs = []
        while True:
            cur = getint(pos)
            # First object should start at fKeyLen, and the last word is the length
            if cur - 1 == len(offs) and offs[-1] == self.fKeyLen:
                break
            offs.append(cur)
            assert cur <= prev, f"objects should have ascending offsets ({cur} < {prev})"
            prev = cur
            pos -= 4
        self.offsets = [x - self.fKeyLen for x in reversed(offs)]
        self.offsets.append(pos)
        
    def __len__(self):
        # offsets always contains the end as well
        return len(self.offsets) - 1
    
    def localindex(self, i):
        if self.ttype.size == None:
            return self.ttype.unpack(self.buf, self.offsets[i], self.offsets[i+1], self)
        else: # The [0] is for 1-element structs. We could support larger ones, but not sure for what...
            return self.ttype.unpack(self.buf[self.offsets[i] : self.offsets[i+1]])[0]
        
    # local iteration
    def __iter__(self):
        for i in range(len(self)):
            yield self.localindex(i)
            
    # global index stuff
    def iterat(self, start):
        for i in range(max(start, self.startindex), self.startindex + len(self)):
            yield i, self.localindex(i - self.startindex)
    def __getitem__(self, i):
        idx, x = next(self.iterat(i))
        assert i == idx
        return x

# Next, a branch holds a list of TBaskets.
# Its main job is managing the starting indices.
# Most of the code is actually to set up a fast way to iterate.
class TBranch:
    def __init__(self, ttype):
        self.ttype = ttype
        self.baskets = [(None, 0)]

    async def addbasket(self, tkey):
        currentend = self.baskets[-1][1]
        basket = await TBasket().load(tkey, self.ttype, currentend)
        self.baskets.append((basket, currentend + len(basket)))

    def iterat(self, start, end = -1):
        # Binary search here would be smart, but we don't use this much anyways.
        for basket, basketend in self.baskets:
            if start < basketend:
                for i, x in basket.iterat(start):
                    if end > 0 and i >= end:
                        return
                    yield i, x

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, end, stride = i.indices(len(self))
            assert stride == 1
            for i, x in self.iterat(start, end):
                yield x
        else:
            idx, x = next(self.iterat(i))
            assert i == idx
            return x

    def __iter__(self):
        return self[:]

    def __len__(self):
        return self.baskets[-1][1]

# The TTree is a pretty stupid container that contains TBranches according to the schema.
class TTree:
    def __init__(self, schema):
        self.branches = {
            name: TBranch(ttype) for name, ttype in schema.items()
        }
        self.branchnames = sorted(schema.keys())

    async def addbasket(self, tkey):
        cls, branch, tree = await tkey.names()
        # ignore branches that we don't have a schema for; this is how we can do partial reads.
        if branch in self.branches:
            await self.branches[branch].addbasket(tkey)

    def __len__(self):
        lens = list(set(len(branch) for branch in self.branches.values()))
        assert len(lens) == 1, "Branches have uneven lenghts! %s" % repr(lens)
        return lens[0]
            
    def __getitem__(self, i):
        # this returns dict's of brnach values.
        # for something more efficient, direct access to the branches can be used.
        if isinstance(i, slice):
            for tup in zip(*[self.branches[name][i] for name in self.branchnames]):
                yield dict(zip(self.branchnames, tup))
        else:
            return {name: branch[i] for name, branch in self.branches.items()}

    def __iter__(self):
        return self[:]

# Finally, set everything up based on a file.
# This is necessarily slow: we read and decompress everything into memory here.
# This is because we need to read each basket to see its length...
# To *not* read everything, reduce the schema to only the things you need.
# But we'll still need to touch each TKey, potentially triggering loading the
# full file to cache.
# The file can be closed after this, we don't keep any references to it.
async def TTreeFile(tfile, schema):
    trees = {
        name: TTree(treeschem) for name, treeschem in schema.items()
    }
    k = await tfile.first()
    while k:
        cls, branch, tree = await k.names()
        if cls == b'TBasket':
            if tree in trees:
                # we could add some parallelism here, but it does not seem to help much.
                await trees[tree].addbasket(k)
        k = await k.next()
    return trees


# Finally, some code for XrootD support. This is not really necessary to read (local)
# ROOT files, but this is why we do asyncio in the first place. This should not be
# used directly, without a layer of caching between us and XrootD. Such a cache is 
# *not* provided here. (Maybe enabling XrootD client caching via env variables is
# enough for some applications:
# https://github.com/xrootd/xrootd/blob/master/src/XrdClient/README_params
# Note that XrootD client *can* read local files (just use a local path as URL), but
# it is much slower then mmap'ing.

# XRootD IO, using pyxrootd. IO latency with xrootd is not as bad as one could 
# think, it takes a few 100ms to open a remote file and a few ten's of ms to
# perform a random read. This is comparable to a normal filesystem on spinning 
# disks and faster than CEPH or mounted EOS (EOS xrootd is maybe 10x faster
# then xrootd from GRID, and slightly faster than mounted EOS, which is faster
# than CEPH standard volumes).
# Throughput can be over 100MBytes/s.

import pyxrootd.client

# PyXrootD uses threads and callbacks.
# We use this interface and translate it into asyncio coroutines.
class XRDFile:
    # All pyxrootd calls go through tis wrapper. It appends a calback= parameter
    # to the call, which releases a asyncio lock in a thread-save way, and then
    # async-waits for this lock to be released.
    async def __async_call(self, function, *args, **kwargs):
        done = asyncio.Event()
        loop = asyncio.get_event_loop()
        async_result = []

        # this must be called from main thread
        def unblock():
            done.set()

        # this can be called from different thread and will call `unblock`
        def callback(*args):
            async_result.append(args)
            loop.call_soon_threadsafe(unblock)

        # the actual call to pyxrootd
        function(*args, **kwargs, callback=callback)
        await done.wait()

        # some minimal error handling: Throw AssertionFailure if *anything* went wrong.
        ok = async_result[0][0]
        assert not ok['error'], repr(ok)
        return async_result[0][1]
        
    async def load(self, url, timeout = 5):
        self.timeout = timeout
        self.file = pyxrootd.client.File()
        await self.__async_call(self.file.open, url, timeout=self.timeout)
        stat = await self.__async_call(self.file.stat)
        self.size = stat['size']
        return self
        
    def __len__(self):
        return self.size
    
    async def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, end, stride = idx.indices(len(self))
            assert stride == 1 and start >= 0 and end >= 0
            buf = await self.__async_call(self.file.read, start, end-start, timeout = self.timeout)
            return buf
        else:
            return (await self[idx:idx+1])[0]
