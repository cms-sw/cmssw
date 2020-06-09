import zlib
import struct
import asyncio
from collections import namedtuple

# These classes allow reading the ROOT contianer structures (TFile, TDirectory, TKey),
# but not actually decoding ROOT serialized objects.
# The reason to use this rather than uproot is much higher performance on TDirectory's,
# as well as fewer dependencies.
# Everything is designed around a buffer like that supports indexing to get bytes. 
# Reading from the buffer is rather lazy and copying data is avoided as long as possible.
# This library supports asyncio; that is another reason for its existence. To acheive
# that, many methods are async, and the buffers have an *async* [...:...] operator.

# The basic structures to read a ROOT file: TFile and TKey.
# We don't actually need TDirectory, since the directory structure can be inferred
# from the TKeys (though reading the TDirectories *might* be more efficient).
# Both of these structures are well documented and should be quite stable over time.

# A ROOT file is a TFile header followed by a sequence of TKey objects. Each TKey object
# starts with a TKey header with some metadata followed by the actual payload object,
# which may be compressed. There can be gaps between TKeys (deleted objects?).

class TFile:
    # Structure from here: https://root.cern.ch/doc/master/classTFile.html
    Fields = namedtuple("TFileFields", ["root", "fVersion", "fBEGIN", "fEND", "fSeekFree", "fNbytesFree", "nfree", "fNbytesName", "fUnits", "fCompress", "fSeekInfo", "fNbytesInfo", "fUUID_low", "fUUID_high"])
    structure_small = struct.Struct(">4sIIIIIIIbIIIQQ")
    structure_big   = struct.Struct(">4sIIQQIIIbIQIQQ")

    # Default behaviour for fulllist()
    TOPATH = lambda parts: b'/'.join(parts) + b'/'
    ALLCLASSES = lambda name: True

    # The TFile datastructure is pretty boring, the only thing we really need
    # is the address of the first TKey, which is actually hardcoded to 100...
    # We provide the fulllist() method for efficient listing of all objects here.
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

    # Returns an async generator producing (path, name, class, offset) tuples.
    # The paths are normalized with the `normalize` callback, the classes 
    # filtered with the `classes` callback.
    # Use `async for` to iterate this.
    async def fulllist(self, normalize = TOPATH, classes = ALLCLASSES):

        # recursive list of path fragments with caching, indexed by fSeekPdir
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
            res = await fullname(parent) + (k.objname(),)
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
            c = key.classname()
            if classes(c):
                yield (await normalized(key.fields.fSeekPdir), key.objname(), c, key.fSeekKey)
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
    # address if it is less than the buffer end.
    async def load(self, buf, fSeekKey, end = None):
        self.buf = buf
        self.end = end if end != None else len(buf)
        self.fSeekKey = fSeekKey

        self.fields = TKey.Fields(*TKey.structure_small.unpack(
            await self.buf[self.fSeekKey : self.fSeekKey + TKey.structure_small.size]))
        headersize = TKey.structure_small.size

        if self.fields.fVersion > 1000:
            self.fields = TKey.Fields(*TKey.structure_big.unpack(
                await self.buf[self.fSeekKey : self.fSeekKey + TKey.structure_big.size]))
            headersize = TKey.structure_big.size

        assert self.fields.fSeekKey == self.fSeekKey, f"{self} is corrupted!"

        # The TKey struct is followed by three strings: class, object name, object title.
        # These consume the sest of the space of the key, unitl, fKeyLen.
        # Read them here eagerly to avoid making to many async read requests later.
        namebuf = await self.buf[self.fSeekKey + headersize : self.fSeekKey + self.fields.fKeyLen]
        self.__classname, pos = self.__readstr(namebuf, 0)
        self.__objname, pos = self.__readstr(namebuf, pos)
        self.__objtitle, pos = self.__readstr(namebuf, pos)

        self.error = False
        return self

    def __readstr(self, buf, pos):
        size = buf[pos]
        if size == 255: # solution for when length does not fit one byte
            size, = TKey.sizefield.unpack(buf[pos+1:pos+5])
            pos += 4
        nextpos = pos + size + 1
        value = buf[pos+1:nextpos]
        return value, nextpos

    def __repr__(self):
        return f"TKey({self.buf}, {self.fSeekKey}, fields = {self.fields})"

    # Read the TKey following this key. According to the documentation these
    # should be one after the other in the file, but in practice there are
    # sometimes gaps (resized/deleted objects?), which are skipped here.
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

    # Parse the three strings in the TKey (classname, objname, objtitle)
    def classname(self):
        return self.__classname

    def objname(self):
        return self.__objname
    
    def compressed(self):
        return self.fields.fNbytes - self.fields.fKeyLen != self.fields.fObjLen

    # Return and potentially decompress object data.
    # Compression is done in thread pool since it could take more time.
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
    # This creates a new TKey pointing there.
    async def parent(self):
        if self.fields.fSeekPdir == 0:
            return None
        return await TKey().load(self.buf, self.fields.fSeekPdir, self.end)

    # Derive the full path of an object by recursing up the parents.
    # slow, primarily for debugging.
    async def fullname(self):
        parent = await self.parent()
        parentname = await parent.fullname() if parent else b''
        return b"%s/%s" % (parentname, self.objname())


