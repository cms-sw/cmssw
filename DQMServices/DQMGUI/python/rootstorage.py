
import sqlite3
import uproot
import struct
import zlib
import time
import re

from collections import namedtuple, defaultdict
from functools import lru_cache

DBNAME = "directory.sqlite"

DBSCHEMA = """
BEGIN;
CREATE TABLE IF NOT EXISTS samples(dataset, run, lumi, filename, menamesid, meoffsetsid);
CREATE INDEX IF NOT EXISTS samplelookup ON samples(dataset, run, lumi);
CREATE UNIQUE INDEX IF NOT EXISTS uniquesamples on samples(dataset, run, lumi, filename);
CREATE TABLE IF NOT EXISTS menames(menamesid INTEGER PRIMARY KEY, menameblob);
CREATE UNIQUE INDEX IF NOT EXISTS menamesdedup ON menames(menameblob);
CREATE TABLE IF NOT EXISTS meoffsets(meoffsetsid INTEGER PRIMARY KEY, meoffsetsblob);
COMMIT;
"""
with sqlite3.connect(DBNAME) as db:
    db.executescript(DBSCHEMA)

# Some types, related to the DB schema.
Sample = namedtuple("Sample", ["dataset", "run", "lumi", "filename", "menamesid", "meoffsetsid"])
MEKey = namedtuple("MEKey", ["name", "secondary", "offset"])
EfficiencyFlag = namedtuple("EfficiencyFlag", ["name"])
ScalarValue = namedtuple("ScalarValue", ["name", "value"]) # could add type here
QTest = namedtuple("QTest", ["name", "qtestname", "status", "result", "algorithm", "message"])
    
# SQLite is at heart single-threaded. However, since we expect a read-heavy
# workload, we can work around that by having multiple connections open at the
# same time. 
class DBHandle:
    def __init__(self, cache):
        self.cache = cache
        self.db = None
    def __enter__(self):
        self.db = sqlite3.connect(DBNAME)
        return self.db
    def __exit__(self, type, value, traceback):
        self.db.close()
dbcache = None # Actually, we don't really need any cache. Makeing sqlite connections is fast.

# TODO: this should go somewhere else, just some glue code for now.
# TODO: this is where Rucio comes in in the future.
def registerfiles(fileurls):
    def tosample(fileurl):
        name = fileurl.split("/")[-1]
        mainpart = name[11:-5] # remove DQM_V00x_R .root
        parts = mainpart.split("__")
        run = int(parts[0])
        dataset = "/" + "/".join(parts[1:])
        lumi = 0
        return Sample(dataset, run, lumi, fileurl, None, None)
    samples = [tosample(f) for f in fileurls]
    with DBHandle(dbcache) as db:
        db.execute("BEGIN;")
        db.executemany("INSERT OR REPLACE INTO samples(dataset, run, lumi, filename) VALUES(?, ?, ?, ?);", 
                   [(sample.dataset, sample.run, sample.lumi, sample.filename) for sample in samples])
        db.execute("COMMIT;")

# TODO: do the slow recursivelist in a separate process, returning the blobs.
def importsample(sample):
    melist = []
    rootf = uproot.open(sample.filename)
    recursivelist(rootf, b"", melist)
    storemelist(sample, melist)
    rootf.close()
    
# This traverses a full TDirectory based file. Slow.
def recursivelist(d, path, out):
    normpath = normalizepath(path)
    if normpath == None: 
        return # don't even bother to look at obsolete stuff
    for k in d._keys:
        if k._fClassName.startswith(b'TDirectory'):
            o = k.get()
            recursivelist(o, path + k._fName + b"/", out)
        elif k._fClassName == b'TObjString':
            # this could be Efficiency flag, scalar ME or QTest
            thing = parsestringentry(k._fName) # name == value!
            if isinstance(thing, EfficiencyFlag): # efficiency flag on this ME
                out.append(MEKey(normpath + thing.name, b'e=1', k._fSeekKey))
            elif isinstance(thing, ScalarValue):
                # scalar ME. Store the name, value needs to be fetched from the file later.
                out.append(MEKey(normpath + thing.name, b'', k._fSeekKey))
            else:
                # QTest. Only save mename and qtestname, values need to be fetched later.
                out.append(MEKey(normpath + thing.name, b'.' + thing.qtestname, k._fSeekKey))
        else:
            # path, name, second name, position
            out.append(MEKey(normpath + k._fName, b'', k._fSeekKey))

def normalizepath(k):
    # CMSSW strucured TDirectory files have some levels of folder structure.
    # Remove that.
    # TODO: we could properly handle multi-run TDirectory files...
    parts = k.split(b"/")
    if len(parts) < 4 or parts[3] != b"Run summary":
        if b'Lumi' in k or b'Reference' in k:
            return None # known obsolete stuff
        else:
            return k # not sure what this could be, better leave as is.
    return parts[2]+ b'/' + b'/'.join(parts[4:-1]) + b'/'

def parsestringentry(val):
    # non-object data is stored in fake-XML stings in the TDirectory.
    # This decodes these strings into an object of correct type.
    assert val[0] == b'<'[0]
    name = val[1:].split(b'>', 1)[0]
    value = val[1+len(name)+1:].split(b'<', 1)[0]
    if value == b"e=1": # efficiency flag on this ME
        return EfficiencyFlag(name)
    elif len(value) >= 2 and value[1] == b'='[0]:
        return ScalarValue(name, value[2:])
    else: # should be a qtest in this case
        assert value.startswith(b'qr=')
        assert b'.' in name
        mename, qtestname = name.split(b'.', 1)
        parts = value[3:].split(b':', 4)
        assert len(parts) == 5, "Expect 5 parts, not " + repr(parts)
        x, status, result, algorithm, message = parts
        assert x == b'st'
        return QTest(mename, qtestname, status, result, algorithm, message)

def storemelist(sample, melist):
    nameblob, offsetblob = melisttoblob(melist)
    # TODO: Sqlite does not like MT-writing. Use a RWLock or sth.?
    # TODO: make this an upsert if the sample already exists.
    with DBHandle(dbcache) as db:
        db.execute("BEGIN;")
        db.execute("INSERT OR IGNORE INTO menames(menameblob) VALUES(?);", (nameblob,))
        cur = db.execute("SELECT menamesid FROM menames WHERE menameblob = ?;",  (nameblob,))
        menamesid = list(cur)[0][0]
        db.execute("INSERT INTO meoffsets(meoffsetsblob) VALUES(?);", (offsetblob,))
        cur = db.execute("SELECT last_insert_rowid();")
        meoffsetsid = list(cur)[0][0]
        db.execute("UPDATE samples SET (menamesid, meoffsetsid) = (?, ?) WHERE (dataset, run, lumi, filename) == (?, ?, ?, ?);", 
                   (menamesid, meoffsetsid, sample.dataset, sample.run, sample.lumi, sample.filename))
        db.execute("COMMIT;")

def melisttoblob(melist):
    # '\n' is a separator that we cna fairly safely use, so we use it for both
    # rows and "colums". One entry is always four rows.
    # Entries are sorted.
    def mekeytorow(key):
        return b'\n'.join(key[:-1])
    # For the list of offsets, Zlib compression is not very effective: there is
    # little repetition for the LZ77 and the Huffmann coder struggles with the
    # large numbers.
    # But, since the list is roughly increasing in order, delta coding makes most
    # values small. Then, the Huffmann coder in Zlib compresses well.
    def deltacode(a):
        prev = 0
        out = []
        for x in a:
            out.append(x - prev)
            prev = x
        return out     
    # The notorious SiStrip bad component workflow creates are varying number of MEs
    # for each run. We just hardcode-ban them here to help the deduplication
    mes = sorted(key for key in melist if not b'BadModuleList' in key.name)
    buf = b'\n'.join(mekeytorow(key) for key in mes)
    nameblob = zlib.compress(buf)
    print (f"Compression {len(buf)/len(nameblob)}, total {len(buf)}")
    offsets = [key.offset for key in mes]
    delta = deltacode(offsets)
    buf = struct.pack("%di" % len(delta), *delta)
    offsetblob = zlib.compress(buf)
    print (f"Compression {len(buf)/len(offsetblob)}, total {len(buf)}")
    return nameblob, offsetblob

# Uncompress the ME names. The offsets will be all -1.
def melistfromblob(nameblob):
    buf = zlib.decompress(nameblob)
    lines = buf.split(b'\n')
    k = len(MEKey._fields) - 1
    out = []
    for x in range(0, len(lines), k):
        parts = list(lines[x:x+k])
        parts.append(-1)
        out.append(MEKey(*parts))
    return out

# Uncompress the ME offsets independently. The result is a bare list of ints,
# matching the names in order.
def meoffsetsfromblob(offsetblob):
    # analog to melisttoblob.
    def deltadecode(d):
        prev = 0
        out = []
        for x in d:
            new = prev + x
            out.append(new)
            prev = new
        return out
    buf = zlib.decompress(offsetblob)
    delta = struct.unpack("%di" % (len(buf)/4), buf)
    return deltadecode(delta)

# This is the main entry point. Given a (partial) Sample tuple and a full name,
# return data for this ME.
# The data will be a list of the main object blob and (potentially) more 
# QTest/Flag/ScalarValue objects.
def readme(sample, fullname):
    fullsample = queryforsample(sample)
    mes = melistfromid(fullsample.menamesid)
    offsets = meoffsetsfromid(fullsample.meoffsetsid)
    keys = locateme(mes, offsets, fullname)
    with FileHandle(fullsample.filename, filecache) as f:
        return [readobject(f, k) for k in keys]
    
# Collect sample info from DB.
# All DB accesses are per-sample and cached.
@lru_cache(maxsize=10)
def queryforsample(sample):
    with DBHandle(dbcache) as db:
        cur = db.execute("SELECT filename, menamesid, meoffsetsid FROM samples WHERE (dataset, run, lumi) == (?, ?, ?);", 
                         (sample.dataset, sample.run, sample.lumi))
        file, menamesid, meoffsetsid = list(cur)[0]
        sample = Sample(sample.dataset, sample.run, sample.lumi, file, menamesid, meoffsetsid)
    if file and menamesid == None or meoffsetsid == None:
        importsample(sample)
        return queryforsample(sample) # retry
    return sample

# These are primarily to have a place for the caches.
# The menames cache should have a high hit rate.
@lru_cache(maxsize=10)
def melistfromid(menamesid):
    with DBHandle(dbcache) as db:
        cur = db.execute("SELECT menameblob FROM menames WHERE menamesid = ?;", (menamesid,))
        blob = list(cur)[0][0]
        return melistfromblob(blob)
    
@lru_cache(maxsize=10)
def meoffsetsfromid(meoffsetsid):
    with DBHandle(dbcache) as db:
        cur = db.execute("SELECT meoffsetsblob FROM meoffsets WHERE meoffsetsid = ?;", (meoffsetsid,))
        blob = list(cur)[0][0]
        return meoffsetsfromblob(blob)

# search for a name in the list of ME names.
# Uses binary search with bound() below.
def locateme(mes, offsets, fullname):
    # This uses a simple order by fullnames, sorted in melisttoblob.
    # This is different from the (path, name) order in DQMIO FullName column.
    # But fine, since we make the list ourselves.
    def access(i):
        return mes[i].name
    idx = bound(access, fullname, 0, len(mes))
    out = []
    while idx < len(mes) and access(idx) == fullname:
        offset = offsets[idx]
        # Attach the offset to the tuple. A bit messy since tuples are immutable...
        out.append(MEKey(*(mes[idx][:-1] + (offset,))))
        idx += 1
    return out 

# Return the index of the first element that is not smaller than key.
# lower is the index of the first element to consider.
# upper is the first element known to be not smaller than key. Must not be touched.
# access is a function used to access elements, and lessthan the comparison functor.
def bound(access, key, lower, upper):
    # nothing more to do, upper is always a valid return value.
    if lower == upper: return upper
    # mid will be smaller than upper, but may be equal to lower.
    mid = lower + int((upper-lower)/2)
    # therefore, it is always save to read a(mid), and we have not done so before.
    if access(mid) < key: 
        # if it is too small, exclude it from the range.
        return bound(access, key, mid+1, upper)
    else:
        # if it is a valid return value, reduce the range and never touch it again.
        return bound(access, key, lower, mid)


# Keep a cache od open xrootd connections.
# Since these do go stale, only reuse them is younger than a minute.
# Maybe the overhead for a connection could be reduced by using the
# uproot.source directly, not using a full file; uproot reads a lot
# of stuff (streamers) in the beginning.
filecache = defaultdict(list)
class FileHandle:
    def __init__(self, filename, cache):
        self.cache = cache
        self.filename = filename
        self.file = None
    def __enter__(self):
        try:
            while not self.file:
                lastused, self.file = self.cache[self.filename].pop()
                if time.time() - lastused > 60:
                    # too old, don't use.
                    self.file.close()
                    self.file = None
        except IndexError:
            # no open files left
            # maybe add more options here
            self.file = uproot.open(self.filename)
        return self.file
    def __exit__(self, type, value, traceback):
        self.cache[self.filename].append((time.time(), self.file))

# Actually fetch a remote object.
def readobject(file, mekey):
    if mekey.secondary == b'e=1': # nothing to read here.
        return EfficiencyFlag(mekey.name)
    # using uproot TKey to read the TKey metadata, and then the value
    # this will typically do only 1 request, since uproot xrootd reads 1M `chunkbytes` at once.
    # going via the TKey gives us ROOT class name and compression handling for free.
    kcur = uproot.source.cursor.Cursor(mekey.offset)
    k = uproot.rootio.TKey.read(file.source, kcur, file._context, None)
    cur = k._cursor.copied()
    buf = bytes(cur.bytes(k._source, k._fObjlen))
    if k._fClassName == b"TObjString":
        thing = parsestringentry(k._fName)
        # could sanity check stuff here...
        return thing
    else:
        # tobject, reconstruct frame for TBufferFile
        # The format is <@length><kNewClassTag=0xFFFFFFFF><classname><nul><@length><2 bytes version><data ...
        # @length is 4byte length of the *entire* remaining object with bit 0x40 (kByteCountMask)
        # set in the first (most significant) byte. This prints as "@" in the dump...
        # the data inside the TKey seems to have the version already.
        classname = k._fClassName
        totlen = 4 + len(classname) + 1 + len(buf)
        head = struct.pack(">II", totlen | 0x40000000, 0xFFFFFFFF)
        return head + classname + b'\0' + buf

# Samples search API
def searchsamples(datasetre = None, runre = None):
    dsmatch = re.compile(datasetre) if datasetre else None
    runmatch = re.compile(runre) if runre else None
    out = []
    with DBHandle(dbcache) as db:
        # TODO: use SQLite with regex extension, pass down re's here.
        cur = db.execute("SELECT DISTINCT dataset, run FROM samples ORDER BY dataset, run;")
        for dataset, run in cur:
            if (not dsmatch or dsmatch.search(dataset)) and (not runmatch or runmatch.search(str(run))):
                out.append(Sample(dataset, run, 0, None, None, None))
        return out

# list API, should be sort-of compatible with old GUI behaviour with recursive = False
def listmes(sample, path, match=None, recursive=True):
    s = queryforsample(sample)
    mes = melistfromid(s.menamesid)
    if match:
        exp = re.compile(match)
        cand = [x.name for x in mes if x.name.startswith(path) and exp.search(x.name)]
    else:
        cand = [x.name for x in mes if x.name.startswith(path)]
    if recursive:
        return set(cand)
    else:
        # return only next-level names
        def relativize(subpath):
            tail = subpath[len(path):]
            if b'/' in tail:
                return tail.split(b'/')[0] + b'/'
            else:
                return tail
        return set(relativize(x) for x in cand)
        

