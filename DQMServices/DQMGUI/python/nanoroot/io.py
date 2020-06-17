import os
import mmap
import asyncio
os.environ["XRD_RUNFORKHANDLER"] = "1"   # To make xrootd + multiprocessing work (Thanks uproot!)
import pyxrootd.client

# First, a sample implementation of the 'async buffer' interface. This is not really
# needed, except for testing. Not recommended for practical use, since it is *not*
# actually async.

class MMapFile:
    def __init__(self, url):
        self.file = open(url, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
    def __len__(self):
        return len(self.mm)
    async def __getitem__(self, idx):
        # This is quite inefficient: this *will* block the main thread.
        # But since mmap is very fast in general, it can still be very fast...
        return self.mm[idx]

# Finally, some code for XrootD support. This is not really necessary to read (local)
# ROOT files, but this is why we do asyncio in the first place. This should not be
# used directly, without a layer of caching between us and XrootD. Such a cache is 
# *not* provided here. There is minimal caching for small reads, which is enough to
# read TTrees efficiently.
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

# PyXrootD uses threads and callbacks.
# We use this interface and translate it into asyncio coroutines.
class XRDFile:
    # All pyxrootd calls go through tis wrapper. It appends a calback= parameter
    # to the call, which releases a asyncio lock in a thread-save way, and then
    # async-waits for this lock to be released.
    async def __async_call(self, function, *args, **kwargs):
        done = asyncio.Event()
        loop = asyncio.get_running_loop()
        async_result = []

        # this must be called from main thread
        def unblock():
            done.set()

        # this can be called from different thread and will call `unblock`
        def callback(*args):
            async_result.append(args)
            if loop.is_running(): # the loop may be shut down by the time we come back
                loop.call_soon_threadsafe(unblock)

        # the actual call to pyxrootd
        function(*args, **kwargs, callback=callback)
        await done.wait()

        # some minimal error handling: Throw AssertionFailure if *anything* went wrong.
        # TODO: better produce some more specific errors...
        ok = async_result[0][0]
        assert not ok['error'], f"Problem with {function} {args} {kwargs} on {self.url}: {repr(ok)}"
        return async_result[0][1]
        
    async def load(self, url, timeout = 20):
        self.timeout = timeout
        self.url = url
        self.file = pyxrootd.client.File()
        await self.__async_call(self.file.open, url, timeout=self.timeout)
        stat = await self.__async_call(self.file.stat)
        self.size = stat['size']
        self.cache = []
        self.cachestart = 0
        return self

    def __repr__(self):
        return f"XRDFile(url={repr(self.url)})"

    async def close(self):
        # no timeout here, better block eternally here than to return and then
        # have GC block forever.
        await self.__async_call(self.file.close)
        
    def __len__(self):
        return self.size
    
    async def __getitem__(self, idx):
        CACHESIZE = 1024*100
        if isinstance(idx, slice):
            start, end, stride = idx.indices(len(self))
            assert stride == 1 and start >= 0 and end >= 0
            if self.cachestart <= start <= end <= self.cachestart + len(self.cache):
                cachehit = self.cache[start - self.cachestart : end - self.cachestart]
                return cachehit
            if end - start < CACHESIZE:
                self.cache = await self.__async_call(self.file.read, start, min(CACHESIZE, self.size-start), timeout = self.timeout)
                self.cachestart = start
                return self.cache[0:end-start]
            # else: uncached read    
            buf = await self.__async_call(self.file.read, start, end-start, timeout = self.timeout)
            #print(f"XRDFile.read  {(end-start)/1024:.2f}kB at {start} {endt-startt:.4f}s {(end-start)/(endt-startt)/1024/1024:.3f}MB/s from {startt} to {endt}")
            return buf
        else:
            return (await self[idx:idx+1])[0]
