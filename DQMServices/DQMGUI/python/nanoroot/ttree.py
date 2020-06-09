import re
import struct
from async_lru import alru_cache

from .tfile import TKey, TFile

# TTree IO. This is more fragile than the TDirectory code.

# The structure of a TTree is not terribly complicated:
# A TTree (table) consists of TBranches (columns). Each TBranch stores its values
# in a list of TBaskets, which contain a variable number of entries each.
# The TBaskets are stored in TKeys in the top level of the directory structure.
# All other info about the TTree (TBranches, TLeafs) is serialized in the
# TTree object in a TKey in the directory.
# Decoding that TTree blob is not easy. But we don't have to: Just reading the
# TBranch keys tells us the TBranch name (name) and TTree name (title) they
# belong to. What remains to be figured out is how many entries there are in
# the TBasket and what the first ID in this TBasket is. 
# To find how many objects there are in a TBasket, there are two cases: 1. it
# holds fixed size objects, then it's just size/object size, or 2. if holds
# variable size objects. In this case, there is an array at the end of the 
# basket data that provides the offset inside the basket data where the n'th
# object starts. By reading this array, we can know how many entries there are
# and where they are.
# The only problem is that we don't know where this array starts (that would
# be encoded somewhere in the TTree object), so we have to read the data from
# the back, heuristically guessing where it ends... but we have a good chance
# that this works: the array header is the length of the array, and we know
# where the first object starts, so we have two consistency checks.
# The other problem is that we don't know the type of the TBaskets (we'd need
# to read the TLeafs from the TTree blob for that), so we require an explicit
# schema from the user.
# To find the start index of each basket, we can just read the entire file,
# adding the baskets in the order that we find them (which should match the
# fCycle order in their keys). But that is quite slow; we'd need to do a full
# read of the entire file before we can read anything.
# To allow partial reads, we actually do extract some data from the TTree blob:
# The basket addresses inside the file (so we can avoid looking at every single
# TKey) and the entry offsets of the baskets (so we cna avoid decoding them if
# we don't actually need entries of that basket.
# To do so, we also need to locate the TTree blob first, without doing a full
# scan of the file. This means we also need to decode some of the TDirectory
# blobs to find the references to the TTrees.
# All of these are not using proper parsers, but instead rely on heuristics:
# We know roughly what the structure aorund the fields we care about looks 
# like, so we can locate them using a regular expression. This can lead to
# false matches, but since we mostly care about references to TKeys, and TKeys
# have strong consistency checks (mainly a pointer to themselves), this works 
# quite reliably. Nevertheless, one could easily construct a valid ROOT file
# where these heuristics fail.

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
        # the data, the buffer has to be reconstructed using the file and seekkey.
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
            # return object here, so we can later have pointers pointing *into* the basket.
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

# Load TTrees from a file by heuristically searching for them in the Directory.
# Trees missing in the schema are ignored; trees in the schema but missing in 
# the file cause an error.
# The trees can be directly accessed via the `self.trees` dict.
class TTreeFile:
    async def load(self, tfile, schema):
        self.trees = dict()
        filetrees = await self.__findtrees(tfile)
        for treename in schema:
            assert treename in filetrees, f"Tree named {repr(treename)} not found in file {repr(tfile)}"
            tkey = filetrees[treename]
            self.trees[treename] = await TTree().load(tkey, schema[treename])
        return self
    
    async def __findtrees(self, tfile):
        Int32 = struct.Struct(">I")
        Int64 = struct.Struct(">Q")
        # Very quick&dirty directory listing that does *not* need to read the whole file.
        # only desinged to find TTrees in a DQMIO file.
        # Basically we just RegEx-match things that look like TKey addresses and try to
        # load them, then return those that loaded.
        # TODO: this won't work for files larger than 4GB. Just adding one more . does
        # not work immediatley, since then we get overlapping matches.
        rootrx = re.compile(b"(\x00\x00\x00\x00....)", flags=re.DOTALL)
        # This pattern depends on wether it is a "big" (>2GB) or "small" file.
        # It relies on the fact that there is a pointer back to key 100 (b'd', TFile.first())
        # behind each entry.
        if tfile.fields.fVersion >= 1000000:
            Int = Int64
            dirrx = re.compile(b"(\x00\x00\x00.....)\x00\x00\x00\x00\x00\x00\x00d", flags=re.DOTALL)
        else:
            Int = Int32
            dirrx = re.compile(b"(....)\x00\x00\x00d", flags=re.DOTALL)

        # First locate the / dir, which is usually at the end of the file, but
        # there is a reference to it in the first TKey.
        rootdir = None
        firstblock = await (await tfile.first()).objdata()
        for x in rootrx.findall(firstblock):
            key, = Int64.unpack(x)
            if key <= 100 or key > len(tfile.buf):
                continue
            try:
                rootdir = await TKey().load(tfile.buf, key)
            except Exception as ex:
                # The rx is not precise, it might have false matches that fail to read.
                # This is fine, ignore them.
                pass

        assert rootdir != None, "Index to / directory not found in ROOT file!"
        
        buf = await rootdir.objdata()

        # then, list the contents of that dir
        trees = dict()
        for x in dirrx.findall(buf):
            key, = Int.unpack(x)
            try:
                tkey = await TKey().load(tfile.buf, key)
            except:
                # The rx is not precise, it might have false matches that fail to read.
                # This is fine, ignore them.
                continue
            classname, name = tkey.classname(), tkey.objname()
            if classname == b'TTree':
                # Sometimes we find multiple versions of one tree. Then, we want the latest one.
                if not name in trees or trees[name].fields.fCycle < tkey.fields.fCycle:
                    trees[name] = tkey
        return trees

# The TTree is a pretty stupid container that contains TBranches according to the schema.
# It does however contain the heuristic decoding code that reads the arrays of TBasket
# pointers and entry offsets from the TTree object in the ROOT file, while ignoring most
# of the object.
# The data in the TTree can be accessed like an array of dicts, but the [...] is async.
# Access via the TTree is not optimzed, better read directly from the TBranches in the
# `branches` dict.
class TTree:
    async def load(self, tkey, schema):
        self.branches = dict()
        self.branchnames = sorted(schema.keys())
        # read the basket index offset/address arrays.
        branchinfo = await self.__parsebranchinfo(tkey, schema.keys())
        for name in schema:
            fBasketEntry, fBasketSeek = branchinfo[name]
            self.branches[name] = await TBranch().load(tkey.buf, schema[name], fBasketEntry, fBasketSeek)
        return self
        
    async def __parsebranchinfo(self, treekey, expectedbranches = []):
        assert (treekey.classname()) == b'TTree'
        Int64 = struct.Struct(">Q")
        buf = await treekey.objdata()
        filebuf = treekey.buf

        # We use the fact that there are 3 consecutive arrays per branch in the blob,
        # and that these root known-size arrays are separated by a single '\x01'
        # byte (is that true? and why? not sure.)
        # The first is fBasketBytes, 32bit values, and pretty useless.
        # The second is fBasketEntry, 64bit values, the first entry number of each basket.
        # The third is fBasketSeek, 64bit values, the fSeekKey of each basket.
        # All are fMaxBaskets long.
        # https://root.cern.ch/doc/v620/TBranch_8h_source.html#l00138
        # We assume that the file size is below 1TB and there are no more than 1T entries
        # in a branch. This allows the following heuristic pattern:
        arraypattern = re.compile(b'(\x00\x00\x00.....)\x01(\x00\x00\x00.....)', flags=re.DOTALL)
        # (for larger trees, there would be less \x00's)

        # This pattern should always match *between* fBasketEntry and fBasketSeek.
        # But of course it also matches in a ton of other places, and there are lots of
        # special cases (empty tree, different branch types, etc.). 
        # So we need to be very flexible.

        # We have one expensive, but very robust check: the value on the right should
        # be a valid fSeekKey to a TBasket, and the TKey then also tells us the tree and
        # branch names it belongs to.

        # The branch names are also easy to find in the blob if we know them. They 
        # typically appear four times each, then there is a bunch of uninteresting stuff.
        # But this heuristic only works if we know the names of all branches beforehand,
        # so we know when we run into the next branch name. We don't.
        # But let's do a quick sanity check here:
        branchinfo = dict()
        for branchname in expectedbranches:
            x = buf.find(branchname)
            assert x > 0, f"Branch {repr(branchname)} not found in TTree!"
            # pre-populate the result to "empty branch", in case we don't find any data later.
            branchinfo[branchname] = ([], [])

        # Instead, if we actually try reading each potential TKey, we also get everything
        # we need, at the cost of a bit more IO. Apparently, it can still happen that we
        # find multiple valid TKeys for a branch, it seems then the first one is correct.
        roots = dict()
        for m in arraypattern.finditer(buf):
            basketkey, = Int64.unpack(m.group(2))
            # Discard implausible values.
            if basketkey == 0 or basketkey > len(filebuf):
                continue

            try: # if we don't have a valid key, don't bother.
                k = await TKey().load(filebuf, basketkey)    
            except AssertionError:
                continue
            
            # if we made it here, record the offset for later.
            branchname = k.objname()
            if not branchname in roots:
                roots[branchname] = m.start() + 8 # address of the \x01

        # Now, we can start decoding the arrays "from the middle".
        for branchname, root in roots.items():
            
            # read towards the left until we hit a 0, that is the start entry of the first basket.
            fBasketEntry = []
            pos = root - 8
            while True:
                basketentry, = Int64.unpack(buf[pos:pos+8])
                pos -= 8
                fBasketEntry.append(basketentry)
                if basketentry == 0:
                    break
            # now we should have the full array, but in reverse.
            fBasketEntry.reverse()
            fMaxBaskets = len(fBasketEntry)
            
            # The right array is easier.
            # We can simply read it from left to right, with some sanity checking.
            fBasketSeek = []
            pos = root + 1
            while True:
                basketkey, = Int64.unpack(buf[pos:pos+8])
                pos += 8
                assert basketkey < len(filebuf), f"Implausibly high basket key at {pos}, value {basketkey}"
                fBasketSeek.append(basketkey)
                # the last basket is a 0 pointer, we can use that as a termination condition
                # here as well. There is a special case if there is only one basket, it seems.
                if basketkey == 0 or fMaxBaskets == 1:
                    break
                
            # the arrays should have the same length:
            assert len(fBasketSeek) == fMaxBaskets, f"Wrong number of basket keys: {len(fBasketSeek)} instead of {fMaxBaskets} in {branchname} at {root}"

            # Finally, record the arrays for this branch to return them.
            # If the branch contains no entries and therefore no baskets, we never get here
            # (the seek key check never succeeded), so a branch not being in this dict can
            # mean it has no entries. We have set all known branches to ([],[]) before though.
            branchinfo[branchname] = (fBasketEntry, fBasketSeek)
        return branchinfo

    def __len__(self):
        lens = list(set(len(branch) for branch in self.branches.values()))
        assert len(lens) == 1, "Branches have uneven lenghts! %s" % repr(lens)
        return lens[0]
    
    async def __asynciter(self, start, end):
        for i in range(start, end):
            # TODO: maybe use branch iterators here, since [i] might be slow.
            yield {name: await value[i] for name, value in self.branches.items()}

    # async gets a bit excessive here. We either return a value, or an async generator,
    # from an async function. For the latter, we need a helper function to even define
    # that and it needs to be called as `async for x in await tree[:]:`.
    # We don't implement __iter__ and __aiter__ since that would be a huge mess with
    # all the async everywhere, just use [:] instead.
    async def __getitem__(self, i):
        # this returns dict's of branch values.
        # for something more efficient, direct access to the branches can be used.
        if isinstance(i, slice):            
            start, end, stride = i.indices(len(self))
            assert stride == 1
            return self.__asynciter(start, end)
        else:
            return {name: await branch[i] for name, branch in self.branches.items()}
        
# Next, a TBranch holds a list of TBaskets.
# Its main job is managing the starting indices, which were read from the tree blob,
# as well as on-demand loading the baskets from file.
# The data in the brnach can be accessed like an array of values, but [...] is async
# and [...:...] returns an async generator that can only be used with `async for`.
class TBranch:
    async def load(self, filebuf, ttype, fBasketEntry, fBasketSeek, basketcachesize=128):
        self.filebuf = filebuf
        self.ttype = ttype
        self.fBasketEntry = fBasketEntry
        self.fBasketSeek = fBasketSeek
        
        # defined as a local function so the cache is at the correct scope.
        @alru_cache(maxsize=basketcachesize)
        async def loadbasket(basketindex):
            tkey = await TKey().load(self.filebuf, self.fBasketSeek[basketindex])
            return await TBasket().load(tkey, self.ttype)
        self.loadbasket = loadbasket
        
        if len(fBasketSeek) == 1 and fBasketSeek[-1] != 0:
            # There is no sentinel at the end of the arrays, so we don't know how
            # many entries there are in total. This should only happen if there is
            # a single basket only.
            # In this case, we read the basket to know it's size. This is also why we
            # can't pass the basket size down to the basket (which would make decoding
            # it much easier), but that is fine, we *can* derive the basket size from
            # its contents. The alternative would be finding fEntries in the tree blob.
            basket = await self.loadbasket(0)
            fBasketSeek.append(0)
            fBasketEntry.append(len(basket))
        if len(fBasketSeek) == 0:
            # no data about this branch found, probably the tree is empty.
            # add one empty basket and a sentinel so the iteration code works.
            fBasketSeek.append(0)
            fBasketEntry.append(0)
            fBasketSeek.append(0)
            fBasketEntry.append(0)
            
        return self
            
    def __len__(self):
        return self.fBasketEntry[-1]

    async def __asynciter(self, start, end):
        # TODO: Binary search here would be smart.
        current = start
        for basketindex in range(len(self.fBasketEntry)-1):
            basketstart, basketend = self.fBasketEntry[basketindex:basketindex+2]
            if basketstart <= current < basketend:
                basket = await self.loadbasket(basketindex)
                assert len(basket) == basketend - basketstart, "Basket has wrong number of entries!"
                while current < basketend:
                    localindex = current - basketstart
                    yield basket[localindex]
                    current += 1
                    if current >= end:
                        return

    async def __getitem__(self, i):
        if isinstance(i, slice):
            start, end, stride = i.indices(len(self))
            assert stride == 1
            return self.__asynciter(start, end)
        else:
            x = [x async for x in self.__asynciter(i, i+1)]
            return x[0]

# A TBasket is pretty self-contained, we can read it from a TKey and its type.
# The only thing it does not know is its starting index in the full table.
# The TBasket always keeps it's data loaded and decompressed in memory.
# This implementation guesses it's number of entires (if they are variable size)
# heuristically by reading the index structure at the end of the buffer. We do
# know the length in most cases, so this could be simplified, but it also acts
# as a sanity check against the fBasketEntry data and keeps data flow simple.
class TBasket:
    async def load(self, tkey, ttype):
        assert tkey.classname() == b'TBasket'
        # we don't really need these, but keep them for later
        self.fKeyLen = tkey.fields.fKeyLen
        self.fSeekKey = tkey.fields.fSeekKey
        self.ttype = ttype
        self.buf = await tkey.objdata() # read and decompress once.
        if ttype.size != None:
            self.__initfixed()
        else:
            self.__initvariable()
        # if the ttype does not actually care about the data (for IndexRange which
        # only keeps the indices to read the data from file later), we can drop the
        # data here to save some memory.
        if hasattr(ttype, 'dropdata') and ttype.dropdata:
            self.buf = None
        return self

    def __initfixed(self):
        # this case is easy -- just cut the buffer into fixed-size records
        self.fEntryOffset = [i*self.ttype.size for i in range(len(self.buf)//self.ttype.size + 1)]
        
    def __initvariable(self):
        Int32 = struct.Struct(">i")
        # this case is more complicated: there is a data structure at the end
        # of the buffer that encodes where the objects start and end, but
        # decoding that without the help of the TBranch metadata is complicated.
        assert self.buf[-4:] == b'\x00\x00\x00\x00'
        pos = len(self.buf) - 8
        prev = len(self.buf) # just so the sanity check works
        fEntryOffset = []
        while True:
            entryoffset, = Int32.unpack(self.buf[pos:pos+4])
            # First object should start at fKeyLen, and the last word is the length
            if  fEntryOffset and fEntryOffset[-1] == self.fKeyLen and entryoffset-1 == len(fEntryOffset):
                break
            fEntryOffset.append(entryoffset)
            pos -= 4
            # sanity check
            assert entryoffset <= prev, f"objects should have ascending offsets ({entryoffset} < {prev})"
            prev = entryoffset
        # ROOT offsets are always from the beginnign of the tkey -- which does not
        # make much sense when the data is in a compressed buffer. We use offsets 
        # into the buffer.
        self.fEntryOffset = [x - self.fKeyLen for x in reversed(fEntryOffset)]
        # We always keep one offset more, so we know the length of the last element.
        self.fEntryOffset.append(pos)

    def __len__(self):
        # fEntryOffset[-1] is the end of the last entry
        return len(self.fEntryOffset) - 1
    
    def __getitem__(self, i):
        # Would not be hard, but not needed.
        assert not isinstance(i, slice), "Slicing not supported"
        if self.ttype.size == None:
            return self.ttype.unpack(self.buf, self.fEntryOffset[i], self.fEntryOffset[i+1], self)
        else: # The [0] is for 1-element structs. We could support larger ones, but not sure for what...
            return self.ttype.unpack(self.buf[self.fEntryOffset[i] : self.fEntryOffset[i+1]])[0]

