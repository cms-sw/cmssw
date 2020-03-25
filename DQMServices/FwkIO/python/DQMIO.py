import ROOT
# try to get some better multi-threading support out of root. Needs to happen as early as possible.
ROOT.ROOT.EnableThreadSafety()
ROOT.TFile.Open._threaded = True

import time
from functools import lru_cache
from collections import namedtuple, defaultdict

MonitorElement = namedtuple('MonitorElement', ['run', 'lumi', 'name', 'type', 'data'])
IndexEntry = namedtuple('IndexEntry', ['run', 'lumi', 'type', 'file', 'firstidx', 'lastidx'])
# Some parameters that could be adjusted. We wait a bit longer when opening 
# files for reading than when testing if they are accessible to avoid spurious
# failures.
TESTTIMEOUT = 3
OPENTIMEOUT = 5
CACHEDFILES = 500

TREENAMES = { 
  0: "Ints",
  1: "Floats",
  2: "Strings",
  3: "TH1Fs",
  4: "TH1Ss",
  5: "TH1Ds",
  6: "TH2Fs",
  7: "TH2Ss",
  8: "TH2Ds",
  9: "TH3Fs",
  10: "TProfiles",
  11: "TProfile2Ds",
}

# 
# This first part is low-level IO on DQMIO files. All the dealing with ROOT
# happens here. No threads are created, but all operations are thread-safe.
#

# open ROOT file using the lower level AsyncOpen to get some concurrency.
def asyncopen(name, timeout):
    handle = ROOT.TFile.AsyncOpen(name)
    while timeout > 0 and ROOT.TFile.GetAsyncOpenStatus(handle) == 1: # kAOSInProgress
        time.sleep(1)
        timeout -= 1
    if timeout == 0:
        return None
    try: # not really sure what possible failure modes are, better be careful
        tfile = ROOT.TFile.Open(handle)
        if tfile.IsOpen():
            return tfile
    except:
        return None
    
# open a file using a context manager, using a potentially cached connection.
# also sets up the TTree for the type of IO we need.
class METree(object):
    def __init__(self, cache, indexentry):
        self.name = indexentry.file
        self.treename = TREENAMES[indexentry.type]
        self.cache = cache

    def __enter__(self):
        self.tfile = self.cache.getfromcache(self.name)
        if self.tfile:
            metree = getattr(self.tfile, self.treename)
        else:
            self.tfile = asyncopen(self.name, OPENTIMEOUT)
            if not self.tfile: return None
            for name in TREENAMES.values():
                t = getattr(self.tfile, name)
                t.GetEntry(0)
                t.SetBranchStatus("*",0)
                t.SetBranchStatus("FullName",1)
                # release GIL in long operations. Disable if it causes trouble.
                t.GetEntry._threaded = True
            metree = getattr(self.tfile, self.treename)
            assert(metree.GetEntry._threaded == True)
        return metree

    def __exit__(self, type, value, traceback):
        self.cache.addtocache(self.name, self.tfile)

# the DQMIO files are sorted by (dir, name) pairs. Splits FullName into (dir, name).
def searchkey(fullname):
    return ("/".join(fullname.split("/")[:-1]), fullname.split("/")[-1])

# Return the index of the first element that is not smaller than key.
# lower is the index of the first element to consider.
# upper is the first element known to be not smaller than key. Must not be touched.
# access is a function used to access elements, and lessthan the comparison functor.
def bound(access, lessthan, key, lower, upper):
    # nothing more to do, upper is always a valid return value.
    if lower == upper: return upper
    # mid will be smaller than upper, but may be equal to lower.
    mid = lower + int((upper-lower)/2)
    # therefore, it is always save to read a(mid), and we have not done so before.
    if lessthan(access(mid), key): 
        # if it is too small, exclude it from the range.
        return bound(access, lessthan, key, mid+1, upper)
    else:
        # if it is a valid return value, reduce the range and never touch it again.
        return bound(access, lessthan, key, lower, mid)

# with bound, this finds the first matching element
def normallessthan(a, b):
    return a < b

# with bound, this skips to the next directory
def skiplessthan(a, key):
    if a[0].startswith(key[0]): return True
    else: return a < key

class DQMIOFileIO:
    """
    This class maintains a cache of open ROOT files (or rather XrootD 
    connections and allows low-level read operations on these files.
    Most operations work on `IndexEntry`s, which can be read from the DQMIO
    file's `Indices` Tree using `readindex`.
    
    Apart form the caching, all operations are stateless and safe to be called 
    from multiple threads.
    """
    def __init__(self):
        # open file cache that operations can "borrow" open files from
        self.openfiles = defaultdict(list)
        # create a per-instance cache of FullName entries.
        self.cached_getsearchkey = lru_cache(maxsize=1024*1024)(self.getsearchkey)

    def getfromcache(self, name):
        if self.openfiles[name]:
            tfile, _ = self.openfiles[name].pop()
            return tfile
        return None
        
    def addtocache(self, name, tfile):
        if tfile:
            self.openfiles[name].append((tfile, time.time()))
            
    # TODO: this is not as threadsafe as it should be.
    def cleancache(self):
        nowopen = dict(self.openfiles)
        if sum(len(l) for l in nowopen.values()) > CACHEDFILES:
            pairs = [(age, tfile, name) for name in nowopen for tfile, age in nowopen[name]]
            pairs = sorted(pairs)
            self.openfiles = defaultdict(list)
            for age, tfile, name in pairs[-CACHEDFILES:]:
                self.openfiles[name].append((tfile, age))
            
    def checkreadable(self, filename):
        """
        Check if `file` can be opened, and return a (filename, bool) pair.

        This will typically wait for a timeout if the cannot be read. Should
        be called multi-threaded for high throughput.
        """
        tfile = asyncopen(filename, TESTTIMEOUT)
        self.addtocache(filename, tfile)
        return filename, tfile != None
    
    def readindex(self, file):
        """
        Read the index table out of a DQMIO file.

        Returns the contents as a list of `IndexEntry`s.
        """
        tfile = self.getfromcache(file)
        if not tfile:
            tfile = asyncopen(file, OPENTIMEOUT)
        if not tfile: return []
        index = []
        idxtree = getattr(tfile, "Indices")
        idxtree.GetEntry._threaded = True
        knownlumis = set()
        for i in range(idxtree.GetEntries()):
            idxtree.GetEntry(i)
            run, lumi, metype = idxtree.Run, idxtree.Lumi, idxtree.Type
            # inclusive range -- for 0 entries, row is left out
            firstidx, lastidx = idxtree.FirstIndex, idxtree.LastIndex
            e = IndexEntry(run, lumi, metype, file, firstidx, lastidx)
            index.append(e)
        self.addtocache(file, tfile)
        return index

    # function just to get caching here: read one FullName from a file.
    # the cache is set up in __init__ so it is per-instance.
    # calling this later *will* open every file twice, but that might be worth it.
    def getsearchkey(self, indexentry, idx):
        with METree(self, indexentry) as metree:
            metree.GetEntry(idx)
            return searchkey(str(metree.FullName))
        
    def listentry(self, indexentry, path):
        """
        List direct children (folders or objects) of the given path in the set
        of MonitorElements identified by `indexentry`.

        Returns a list of relative names, ending in "/" for subdirectories.

        This is implemented as a skip scan using binary search over the FullName
        branch in the DQMIO data; it will not read the full list of names.
        """
        assert path == "" or path[-1] == "/", "Can only list directories!"
        with METree(self, indexentry) as metree:
            if not metree: return []

            def getentry(idx):
                return self.cached_getsearchkey(indexentry, idx)

            first, last = indexentry.firstidx, indexentry.lastidx+1
            key = searchkey(path)
            # start by skipping to the first item in the directory
            start = bound(getentry, normallessthan, key, first, last)
            pos = start
            elements = []
            # first, list the objects.
            while pos != last:
                dirname, objname = getentry(pos)
                if not dirname == key[0]:
                    # we are done, hit subdirectories
                    break
                #print("obj", dirname, objname)
                elements.append(objname)
                pos += 1

            # then, skip over directories.
            while pos != last:
                dirname, objname = getentry(pos)
                if not dirname.startswith(key[0]):
                    # we are done, left the directory.
                    break
                relative = dirname[len(path):]
                reldirname = relative.split("/")[0]
                elements.append(reldirname + "/")
                #print("dir", dirname, objname, relative, reldirname)
                pos = bound(getentry, skiplessthan, (path + reldirname, ""), pos, last)

            return elements

    def filterentry(self, indexentry, pathfilter, first=0, last=1e9):
        """
        List MonitorElements matching the `pathfilter` re object in the set of
        MEs indentified by the given indexentry. The search range can be 
        reduced with the first/last parameters to allow for a parallel search.

        Returns a list of matching ME fullnames.

        This is implemented as a linear scan.
        """
        with METree(self, indexentry) as metree:
            if not metree: return []
            begin = max(first, indexentry.firstidx)
            end = min(last, indexentry.lastidx+1)
            result = []
            for i in range(begin, end):
                metree.GetEntry(i)
                fullname = str(metree.FullName)
                m = pathfilter.match(fullname)
                if m:
                    result.append(fullname)
            return result
    
    def readoneME(self, indexentry, fullname):
        """
        Read a MonitorElement with the given fullname from the set of MEs
        ientified by indexentry. Returns a `MonitorElement` object with a ROOT
        object as `value`, or None if no matching ME is found.
        """
        with METree(self, indexentry) as metree:
            if not metree: return None

            first, last = indexentry.firstidx, indexentry.lastidx+1
            key = searchkey(fullname)
            # start by skipping to the first item in the directory
            pos = bound(lambda idx: self.cached_getsearchkey(indexentry, idx), normallessthan, key, first, last)
            if pos == last:
                return None

            metree.GetEntry(pos, 1) # read full row
            if str(metree.FullName) != fullname:
                return None
            value = metree.Value.Clone()
            return MonitorElement(indexentry.run, indexentry.lumi, fullname, indexentry.type, value)

#
# This second part is a higher level abstraction over full datasets consisting
# of many DQMIO files. A large thread pool is used to make the high latency 
# remote reads acceptably fast. A SQLite database is kept for metadata.
#
import re
import sqlite3
import fnmatch
import subprocess
from multiprocessing.pool import ThreadPool

DASFILEPREFIX = "root://cms-xrd-global.cern.ch/"
DBSCHEMA = """
    CREATE TABLE IF NOT EXISTS dataset(id INTEGER PRIMARY KEY, datasetname, lastchecked);
    CREATE UNIQUE INDEX IF NOT EXISTS datasets ON dataset(datasetname);
    CREATE TABLE IF NOT EXISTS file(id INTEGER PRIMARY KEY, datasetid, name, readable, lastchecked, indexed);
    CREATE INDEX IF NOT EXISTS datasettofile ON file(datasetid);
    CREATE UNIQUE INDEX IF NOT EXISTS filetoid ON file(name);
    CREATE TABLE IF NOT EXISTS indexentry(id INTEGER PRIMARY KEY, run, lumi, type, file, firstidx, lastidx);
    CREATE UNIQUE INDEX IF NOT EXISTS entries ON indexentry(run, lumi, type, file, firstidx, lastidx);
    CREATE INDEX IF NOT EXISTS fileentries ON indexentry(file);
    CREATE INDEX IF NOT EXISTS runlumi ON indexentry(run, lumi);
"""

# helper function for building simple task managers.
def splitdone(tasks):
    running = []
    done = []
    for t in tasks:
        if t.ready():
            done.append(t)
        else:
            running.append(t)
    return running, done

class DQMIOReader:
    """
    This class manages data from a set of DQMIO files. This can be a full
    dataset read from DAS. Operations are internally multi-threaded.
    All state is kept in a SQLite database, caches are managed by `DQMIOFileIO`.
    """
    def __init__(self, dbname="", nthreads=32):
        self.db = sqlite3.connect(dbname)
        self.db.executescript(DBSCHEMA)
        self.pool = ThreadPool(nthreads)
        self.io = DQMIOFileIO()
          
    def importdatasets(self, datasetpattern):
        """
        Query DAS for datasets matching `datasetpattern` and add their files.
        """
        dqmiodatasets = subprocess.check_output(['dasgoclient', '-query', 'dataset dataset=/*/*/DQMIO'])
        datasets = []
        for name in dqmiodatasets.splitlines():
            datasetname = name.decode("utf-8")
            if fnmatch.fnmatch(datasetname, datasetpattern):
                datasets.append(datasetname)
        self.importdataset(datasets) 
    
    def importdataset(self, datasetnames):
        """
        Query DAS for files from the given datasets and add them.
        """
        if isinstance(datasetnames, str):
            return self.importdataset([datasetnames])
        
        # helper function for parallel running
        def getfiles(datasetname):
            filelist = subprocess.check_output(['dasgoclient', '-query', 'file dataset=%s' % datasetname])
            return datasetname, [DASFILEPREFIX + f.decode("utf8") for f in filelist.splitlines()]

        tasks = [self.pool.map_async(getfiles, [d,]) for d in datasetnames] 
        while tasks:
            time.sleep(1)
            tasks, done = splitdone(tasks)
            for t in done:
                if not t.successful():
                    print("DAS file listing task failed.")
                    continue
                for it in t.get(): # should only be one
                    datasetname, filelist = it
                    self.addfiles(datasetname, filelist)
        print(str(len(tasks)) + " tasks remaining")
        
    def addfiles(self, datasetname, filelist):
        """
        Add ROOT files to the metadata database, under the given dataset name.
        """
        self.db.execute("BEGIN;")
        self.db.execute(f"INSERT OR REPLACE INTO dataset(datasetname, lastchecked) VALUES (?, datetime('now'));", (datasetname,))
        datasetid = self.db.execute(f"SELECT id FROM dataset WHERE datasetname = ?;", (datasetname,))
        datasetid = list(datasetid)[0][0]
        self.db.executemany("INSERT OR IGNORE INTO file(datasetid, name) VALUES (?, ?)", [(datasetid, name) for name in filelist])  
        self.db.execute("COMMIT;")
          
    def datasets(self):
        """
        Return a list of known dataset names.
        """
        return [row[0] for row in self.db.execute("SELECT datasetname FROM dataset;")]

    def checkfiles(self, since="+1 day"):
        """
        Update metadata for all known files. Checks if files are readable and
        reads the Index trees, re-checks files last checked more than `since` ago.
        """
        cur = self.db.cursor()
        newfiles = list(cur.execute("""
           SELECT name FROM file 
           WHERE (readable and indexed is null)
              or (  (readable is null or not readable) 
                and (lastchecked is null or datetime(lastchecked, ?) <  datetime('now')));
        """, (since,)))
        cur.close()
        checktasks = [self.pool.map_async(self.io.checkreadable, [f,]) for f, in newfiles] 
        readtasks = []
        
        while checktasks or readtasks:
            time.sleep(1)
            self.io.cleancache()
            print(str(len(checktasks)) + " readablility check tasks, " + str(len(readtasks)) + " index read tasks remaining")
            checktasks, done = splitdone(checktasks)
            results = []
            for t in done:
                if not t.successful():
                    print("Readability check task failed.")
                for file, readable in t.get(): # should only be one
                    results.append((file, readable))
                    if readable:
                        readtasks.append(self.pool.map_async(self.io.readindex, [file,]))
            if results:
                self.db.execute("BEGIN;")
                self.db.executemany("UPDATE file SET readable = ? WHERE name = ?;", 
                                    [(readable, name) for name, readable in results])
                self.db.executemany("UPDATE file SET lastchecked = datetime('now') WHERE name = ?;", 
                                    [(name,) for name, _ in results])
                self.db.execute("COMMIT;")
            
            readtasks, done = splitdone(readtasks)
            entries = []
            for t in done:
                if not t.successful():
                    print("Index read task failed.")
                for index in t.get(): # should only be one
                    entries += index
            if entries:
                self.db.execute("BEGIN;")
                self.db.executemany("""INSERT OR REPLACE INTO indexentry(run, lumi, type, file, firstidx, lastidx) 
                    VALUES (?, ?, ?, (SELECT id FROM file WHERE name = ?), ?, ?);""", entries)
                files = set(e.file for e in entries)
                self.db.executemany('UPDATE file SET indexed = 1 WHERE id = (SELECT id FROM file WHERE name = ?);', [(f, ) for f in files])
                self.db.execute("COMMIT;")
    
    def samples(self):
        """
        Return a list of known samples ((datasetname, run, lumi) tuples).
        """
        return list(self.db.execute("""
            SELECT DISTINCT datasetname, run, lumi FROM indexentry 
            INNER JOIN file ON indexentry.file = file.id 
            INNER JOIN dataset ON file.datasetid = dataset.id;"""))

    def entriesforsample(self, datasetname, run, lumi, onefileonly=False):
        query = f"""
            SELECT run, lumi, type, file.name, firstidx, lastidx
            FROM dataset
            INNER JOIN file ON file.datasetid = dataset.id 
            INNER JOIN indexentry on indexentry.file = file.id 
            WHERE datasetname = ? AND run = ? AND lumi = ?
        """
        if onefileonly:
            query += "GROUP BY type" #we only need one entry per type, no need to read all files.
        entries = self.db.execute(query, (datasetname, run, lumi))
        return [IndexEntry(*row) for row in entries]
        
    def filtersample(self, datasetname, run, lumi, pathfilter):
        """
        Return ME names matching the regular expression `pathfilter` in the
        given sample.
        """
        entries = self.entriesforsample(datasetname, run, lumi, onefileonly=True)
        tasks = []
        exp = re.compile(pathfilter)
        for e in entries:
            for i in range(e.firstidx, e.lastidx+1, 5000):
                tasks.append((e, exp, i, i + 5000))
        pathsets = self.pool.map(lambda t: self.io.filterentry(*t), tasks)
        paths = set().union(*pathsets)
        self.io.cleancache()
        return paths

    def listsample(self, datasetname, run, lumi, path):
        """
        Return object names that are children of `path`  in the
        given sample.
        """
        entries = self.entriesforsample(datasetname, run, lumi, onefileonly=True)
        pathsets = self.pool.map(lambda e: self.io.listentry(e, path), entries)
        paths = set().union(*pathsets)
        self.io.cleancache()
        return paths
    
    def readsampleme(self, datasetname, run, lumi, fullnames):
        """
        Return `MonitorElement`s matching the given names in the given sample.

        For RUN MEs (lumi = 0), there may be multiple objects in different files
        that need to be merged; these are returned as individual objects.
        """
        if isinstance(fullnames, str):
            return self.readsampleme(datasetname, run, lumi, [fullnames])
        entries = self.entriesforsample(datasetname, run, lumi)
        # TODO: maybe try the most common `type`s first, to make the common case fast.
        mes = self.pool.starmap(self.io.readoneME, [(e, fullname) for e in entries for fullname in fullnames])
        self.io.cleancache()
        return [me for me in mes if me]

    def readlumimes(self, datasetname, run, fullnames):
        """
        Return `MonitorElement`s matching the given names in all lumis of the
        given dataset/run. This might not be all lumis of the run, since some
        files might not be readable.

        """
        if isinstance(fullnames, str):
            return self.readlumimes(datasetname, run, [fullnames])
        # a custom SQL query would be much more efficient but that does not really matter here.
        lumis = set(lumi for d, r, lumi in self.samples() if d == datasetname and r == run)
        entries = sum([self.entriesforsample(datasetname, run, l) for l in lumis], []) 
        mes = self.pool.starmap(self.io.readoneME, [(e, fullname) for e in entries for fullname in fullnames])
        self.io.cleancache()
        return [me for me in mes if me]
    
