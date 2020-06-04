import re
import zlib
import struct
from collections import namedtuple

# If uproot is micro-Python ROOT, this is Nano ROOT.
# These classes allow reading the ROOT contianer structures (TFile, TDirectory, TKey),
# but not actually decoding ROOT serialized objects.
# The reason to use this rather than uproot is much higher performance on TDirectory's,
# as well as fewer dependencies.
# Everything is designed around a buffer like it is returned by mmap.mmap, that supports
# indexing to get bytes. Reading from the bufer is rather lazy and copying data is
# avoided as long as possible.

class TFile:
    # Structure from here: https://root.cern.ch/doc/master/classTFile.html
    Fields = namedtuple("TFileFields", ["root", "fVersion", "fBEGIN", "fEND", "fSeekFree", "fNbytesFree", "nfree", "fNbytesName", "fUnits", "fCompress", "fSeekInfo", "fNbytesInfo", "fUUID"])
    structure = struct.Struct(">4sIIIIIIIbIIII")

    
    # These two are default transforms that can be overwritten.
    topath = lambda parts: b'/'.join(parts) + b'/'
    allclasses = lambda name: True

    # The TFile datastructure is pretty boring, the only thing we really need
    # is the address of the first TKey, which is actually hardcode to 100...
    # But this class also provides some caching to make computing full paths
    # fast, and the `fulllist` method that lists all objects using this.
    # Its behaviour can be adjusted with the `normalize` and `classes` 
    # callback functions.
    def __init__(self, buf, normalize = topath, classes = allclasses):
        self.buf = buf
        self.normalize = normalize
        self.classes = classes
        self.fields = TFile.Fields(*TFile.structure.unpack(
            self.buf[0 : TFile.structure.size]))
        assert self.fields.root == b'root'
        self.normalizedcache = dict()
        self.dircache = dict()
        self.dircache[0] = ()
        self.error = False
        
    def __repr__(self):
        return f"TFile({self.buf}, fields = {self.fields})"

    # First TKey in the file. They are sequential, use `next` on the key to
    # get the next key.
    def first(self):
        return TKey(self.buf, self.fields.fBEGIN, self.end())

    def end(self):
        if len(self.buf) < self.fields.fEND:
            self.error = True
            print(f"TFile corrupted, fEND ({self.fields.fEND}) behind end-of-file ({len(self.buf)})")
            return len(self.buf)
        return self.fields.fEND

    # Not that useful, but one could build a small file with only the streamers using this.
    def streamerinfo(self):
        return self.buf[tfile.fields.fSeekInfo : tfile.fields.fSeekInfo + tfile.fields.fNbytesInfo]
    
    def fullname(self, key):
        return (self._normalized(key.fields.fSeekPdir), key.objname())
    
    def _fullname(self, fSeekKey):
        if fSeekKey in self.dircache:
            return self.dircache[fSeekKey]
        k = TKey(self.buf, fSeekKey)
        p = k.fields.fSeekPdir
        res = self._fullname(p) + (k.objname(),)
        self.dircache[fSeekKey] = res
        return res
    
    def _normalized(self, fSeekKey):
        if fSeekKey in self.normalizedcache:
            return self.normalizedcache[fSeekKey]
        parts = self._fullname(fSeekKey)
        res = self.normalize(parts)
        self.normalizedcache[fSeekKey] = res
        return res
    
    # Returns a generator producing (path, name, class, offset) tuples.
    # The paths are normalized with the `normalize` callback, the classes 
    # filtered with the `classes` callback.
    def fulllist(self):
        x = self.first()
        while x:
            c = x.classname()
            if self.classes(c):
                yield (*self.fullname(x), c, x.fSeekKey)
            n = x.next()
            if x.error:
                self.error = True
            x = n
        
class TKey:
    # Structure also here: https://root.cern.ch/doc/master/classTFile.html
    Fields = namedtuple("TKeyFields", ["fNbytes", "fVersion", "fObjLen", "fDatime", "fKeyLen", "fCycle", "fSeekKey", "fSeekPdir"])
    structure_small = struct.Struct(">iHIIHHII")
    structure_big   = struct.Struct(">iHIIHHQQ")
    sizefield = struct.Struct(">i")

    # In this case of curruption, we search for the next known class name and try it there.
    # This should never be used in the normal case.
    resync = re.compile(b'TDirectory|TH[123][DFSI]|TProfile|TProfile2D')

    # Decode key at offset `fSeekKey` in `buf`. `end` can be the file end
    # address if it is less then the buffer end.
    def __init__(self, buf, fSeekKey, end = None):
        self.buf = buf
        self.end = end if end != None else len(buf)
        self.fSeekKey = fSeekKey
        self.fields = TKey.Fields(*TKey.structure_small.unpack(
            self.buf[self.fSeekKey : self.fSeekKey + TKey.structure_small.size]))
        self.headersize = TKey.structure_small.size
        if self.fields.fVersion > 1000:
            self.fields = TKey.Fields(*TKey.structure_big.unpack(
                self.buf[self.fSeekKey : self.fSeekKey + TKey.structure_big.size]))
            self.headersize = TKey.structure_big.size
        assert self.fields.fSeekKey == self.fSeekKey, f"{self} is corrupted!"
        self.error = False

    def __repr__(self):
        return f"TKey({self.buf}, {self.fSeekKey}, fields = {self.fields})"

    # Read the TKey following this key. According to the documentation these
    # should be one after the other in the file, but in practice there are
    # sometimes gaps (resized/deleted objects?), which are skipped here.
    # If we still fail to read the next item, we try to find the next key
    # by searching for a familiar class name.
    def next(self):
        offset = self.fields.fSeekKey + self.fields.fNbytes
        while (offset+TKey.structure_small.size) < self.end:
            # It seems that a negative length indicates an unused block of that size. Skip it.
            # The number of such blocks matches nfree in the TFile.
            size, = TKey.sizefield.unpack(self.buf[offset:offset+4])
            if size < 0:
                offset += -size
                continue
            try:
                k = TKey(self.buf, offset, self.end)
                return k
            except AssertionError:
                m = TKey.resync.search(self.buf, offset + TKey.structure_small.size + 2)
                if not m: break
                newoffset = m.start() - 1 - TKey.structure_small.size
                # TODO: also try big header
                print(f"Error: corrupted TKey at {offset}:{repr(self.buf[offset:offset+32])} resyncing (+{newoffset-offset})")
                self.error = True
                offset = newoffset
        return None

    def _getstr(self, pos):
        size = self.buf[pos]
        if size == 255: # soultion for when length does not fit one byte
            size, = TKey.sizefield.unpack(self.buf[pos+1:pos+5])
            pos += 4
        nextpos = pos + size + 1
        value = self.buf[pos+1:nextpos]
        return value, nextpos

    # Parse the three strings in the TKey (classname, objname, objtitle)
    def names(self):
        pos = self.fSeekKey + self.headersize
        classname, pos = self._getstr(pos)
        objname, pos = self._getstr(pos)
        objtitle, pos = self._getstr(pos)
        return classname, objname, objtitle

    def classname(self):
        return self._getstr(self.fSeekKey + self.headersize)[0]

    def objname(self):
        # optimized self.names()[1]
        pos = self.fSeekKey + self.headersize
        pos += self.buf[pos] + 1
        if self.buf[pos] == 255:
            size, = TKey.sizefield.unpack(self.buf[pos+1:pos+5])
            pos += 4
            nextpos = pos + size
        else:
            nextpos = pos + self.buf[pos]
        return self.buf[pos+1:nextpos+1]
    def compressed(self):
        return self.fields.fNbytes - self.fields.fKeyLen != self.fields.fObjLen

    # Return and potentially decompress object data.
    def objdata(self):
        start = self.fields.fSeekKey + self.fields.fKeyLen
        end = self.fields.fSeekKey + self.fields.fNbytes
        if not self.compressed():
            return self.buf[start:end]
        else:
            comp = self.buf[start:start+2]
            assert comp == b'ZL', "Only Zlib compression supported, not " + repr(comp)
            # There are a bunch of header bytes, not exactly sure what they mean.
            return zlib.decompress(self.buf[start+9:end])

    # Each key (except the root) has a parent directory.
    # Creates a new TKey pointing there.
    def parent(self):
        if self.fields.fSeekPdir == 0:
            return None
        return TKey(self.buf, self.fields.fSeekPdir, self.end)

    # Derive the full path of an object by recursing up the parents.
    # slow, primarily for debugging.
    def fullname(self):
        parent = self.parent()
        parentname = parent.fullname() if parent else b''
        return b"%s/%s" % (parentname, self.objname())


# TBufferFile data contains more headers compared to the data in a TKey or
# TTree. This function tries to add the missing headers.
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
        # root crashes due to infinite recursion.
        self.displacement = displacement


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
# start IDs; this is not guaranteed but seems to hold.
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

# First, the schema.
# We use struct.Struct for fixed size types, and objects with a `size` of None
# and a `unpack` method for variable size types.

Int32 = struct.Struct(">i")
Int64 = struct.Struct(">q")
Float64 = struct.Struct(">d")

class String:
    size = None
    @staticmethod
    def unpack(buf, start, end, basket):
        size = buf[start]
        start += 1
        if size == 0xFF:
            size, = Int32.unpack(buf[start:start+4])
            start += 4
        s = buf[start:end]
        assert len(s) == size, f"Size of string does not match buffer size: {repr(buf[start:end])}, {size})"
        return s

class ObjectBlob:
    class Range:
        def __init__(self, buf, start, end, fSeekKey):
            self.buf = buf
            self.start = start
            self.end = end
            self.fSeekKey = fSeekKey # address of the basket this came from
        def data(self):
            return self.buf[self.start:self.end]
        def __repr__(self):
            return f"Range(buf=<len()={len(self.buf)}>, start={self.start}, end={self.end}, fSeekKey={self.fSeekKey})"
    size = None
    @staticmethod
    def unpack(buf, start, end, basket):
        # return object here, so we can later extract pointers pointing *into* the basket.
        # To make that possible, API needs to diverge from Struct.unpack...
        # The basket also passes itself in so we can keep whatever info we need.
        return ObjectBlob.Range(buf, start, end, basket.fSeekKey)
    
# A schema for a TTree is a dict mapping branch names to types.
# A schema for a file is a dict mapping TTree names to TTree schemas.

# A Tbasket is pretty self-contained, we can read it from a TKey and its type.
# The only thing it does not know is its starting index in the full table.
# We don't save stuff that is not absolutely needed, like names or a ref to the
# file (we do keep a copy of the (decompressed) data).
class TBasket:
    def __init__(self, tkey, ttype, startindex = None):
        assert tkey.classname() == b'TBasket'
        self.fKeyLen = tkey.fields.fKeyLen
        self.fSeekKey = tkey.fields.fSeekKey # we don't really need that, but keep it for later
        self.ttype = ttype
        self.startindex = startindex
        self.buf = tkey.objdata() # read and decompress once.
        if ttype.size != None:
            self.initfixed()
        else:
            self.initvariable()
    def initfixed(self):
        self.offsets = [i*self.ttype.size for i in range(len(self.buf)//self.ttype.size + 1)]
    def initvariable(self):
        buf = self.buf
        pos = len(buf) - 4
        def getint(pos):
            i, = Int32.unpack(buf[pos:pos+4])
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
        #  offsets always contains the end as well
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
    def addbasket(self, tkey):
        currentend = self.baskets[-1][1]
        basket = TBasket(tkey, self.ttype, currentend)
        self.baskets.append((basket, currentend + len(basket)))
    def iterat(self, start, end = -1):
        # TODO: we could use this for tbranch[start:end]...
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
    def addbasket(self, tkey):
        cls, branch, tree = tkey.names()
        # ignore branches that we don't have a schema for; this is how we can do partial reads.
        if branch in self.branches:
            self.branches[branch].addbasket(tkey)
    def __getitem__(self, i):
        if isinstance(i, slice):
            for tup in zip(*[self.brnaches[name][i] for name in self.branchnames]):
                yield dict(zip(self.branchnames, tup))
        else:
            return {name: branch[i] for name, branch in self.branches.items()}
    def __len__(self):
        lens = list(set(len(branch) for branch in self.branches.values()))
        assert len(lens) == 1, "Branches have uneven lenghts! %s" % repr(lens)
        return lens[0]
    def __iter__(self):
        return zip(*[self.branches[name] for name in self.branchnames])

# Finally, set everything up based on a file.
# This is necessarily slow: we read and decompress everything into memory here.
# This is because we need to read each basket to see its length...
# To *not* read everything, reduce the schema to only the things you need.
# But we'll still need to touch each TKey, potentially triggering loading the
# full file to cache.
# The file can be closed after this, we don't keep any references to it.
def TTreeFile(tfile, schema):
    trees = {
        name: TTree(treeschem) for name, treeschem in schema.items()
    }
    k = tfile.first()
    while k:
        cls, branch, tree = k.names()
        if cls == b'TBasket':
            if tree in trees:
                trees[tree].addbasket(k)
        k = k.next()
    return trees

