import asyncio
from async_lru import alru_cache


class IOService:
    BLOCKSIZE = 128*1024 # number of bytes to read at once
    READAHEAD = 8        # number of blocks to read at once
    CACHEBLOCKS = 1000   # total number of block to keep in cache
    CONNECTIONS = 100    # maximum number of open connections
    OPENTIMEOUT = 5      # maximum time to wait while opening file (seconds)
    MAXOPENTIME = 60     # maximum time that a connection stays open (seconds)
    
    @classmethod
    async def openurl(cls, url, blockcache=True):
        if blockcache:
            f = BlockCachedFile(url, cls.BLOCKSIZE)
        else:
            f = FullFile(url, timeout = cls.OPENTIMEOUT)
        await f.preload()
        return f
    
    @classmethod
    async def readahead(cls, url, firstblock):
        size = await cls.readlen(url)
        for blockid in range(firstblock, firstblock+cls.READAHEAD):
            if size < cls.BLOCKSIZE*blockid:
                break
            await IOService.readblock(url, blockid) 
    
    @classmethod
    @alru_cache(maxsize=CACHEBLOCKS)
    async def readblock(cls, url, blockid):
        print(f"readblock {url} {blockid}")
        # Start reading some blocks. This will read more blocks than we actually
        # need in the background, to pre-populate the cache.
        # TODO: This does not work: calling it here causes an infinite recursion,
        # reading the entire file. But we also don't want to call it from user code,
        # to avoid the overhead in the case of a cache *hit*.
        #asyncio.Task(cls.readahead(url, blockid))
        
        file = await cls.connect(url)
        return await file[blockid*cls.BLOCKSIZE : (blockid+1)*cls.BLOCKSIZE]
 
    @classmethod
    @alru_cache(maxsize=CONNECTIONS)
    async def readlen(cls, url):
        print(f"readlen {url}")
        file = await cls.connect(url)
        return len(file)
    
    @classmethod
    @alru_cache(maxsize=CONNECTIONS)
    async def connect(cls, url):
        print(f"connect {url}")
        def closefile():
            # XRD connections tend to break after a while.
            # To prevent this, we preventively close all connections after a certain time.
            await asyncio.sleep(cls.MAXOPENTIME)
            # This is done by removing the file from the cache, GC takes care of the rest.
            cls.connect.invalidate(cls, url)
        file = XRDFile()
        await file.connect(url, timeout=cls.OPENTIMEOUT)
        asyncio.Task(closefile())
        return file
    
class BlockCachedFile:
    def __init__(self, url, blocksize):
        self.url = url 
        self.blocksize = blocksize
        
    async def preload(self):
        # since len() can't be async, we read the length here.
        self.size = await IOService.readlen(self.url)
        
    def __len__(self):
        return self.size
    
    async def __getblocks(self, idxslice):
        # Process the __getitem__ parameters.
        start, end, stride = idxslice.indices(len(self))
        assert stride == 1 and start >= 0 and end >= 0
        
        firstblock, lastblock = start//self.blocksize, end//self.blocksize
        
        # For the blocks we actually need (rarely more than one), we start parallel requests.
        blockids = list(range(firstblock, lastblock+1))
        tasks = [IOService.readblock(self.url, blockid) for blockid in blockids]
        blocks = await asyncio.gather(*tasks)
        
        # finally, we assemble the result. There are some unnecessary copies here.
        parts = []
        for blockid, block in zip(blockids, blocks): 
            offset = blockid*self.blocksize
            parts.append(block[max(0, start-offset) : max(0, end-offset)])
        return b''.join(parts)
            
    async def __getitem__(self, idx):
        if isinstance(idx, slice):
            return await self.__getblocks(idx)
        else:
            return await self[idx:idx+1][0]
    
class FullFile:
    def __init__(self, url, timeout):
        self.url = url
        self.timeout = timeout
    async def preload(self):
        # in this mode, we just preload the full file in the beginning, into a per-file cache.
        f = XRDFile()
        await f.connect(url, timeout=self.timeout)
        self.buf = await f[:]
    async def __getitem__(self, idx):
        return self.buf[idx]

import mmap
# A very basic implementation of the "async buffer" interface.
class MMapFile:
    def __init__(self, url):
        self.file = open(url, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
    def __len__(self):
        return len(self.mm)
    async def __getitem__(self, idx):
        return mm[idx]
    
import pyxrootd.client
# asyncio interface to PyXrootD. PyXrootD uses threades and callbacks.
# We use this interface and translate it into asyncio coroutines.
class XRDFile:
    async def __async_call(self, function, *args, **kwargs):
        done = asyncio.Event()
        loop = asyncio.get_event_loop()
        async_result = []
        def unblock():
            done.set()
        def callback(*args):
            async_result.append(args)
            loop.call_soon_threadsafe(unblock)
        function(*args, **kwargs, callback=callback)
        await done.wait()
        ok = async_result[0][0]
        assert not ok['error'], repr(ok)
        return async_result[0][1]
        
    async def connect(self, url, timeout = 5):
        self.timeout = timeout
        self.file = pyxrootd.client.File()
        await self.__async_call(self.file.open, url, timeout=self.timeout)
        stat = await self.__async_call(self.file.stat)
        self.size = stat['size']
        
    def __len__(self):
        return self.size
    
    async def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, end, stride = idx.indices(len(self))
            assert stride == 1 and start >= 0 and end >= 0
            buf = await self.__async_call(self.file.read, start, end-start, timeout = self.timeout)
            return buf
        else:
            return self[idx:idx+1][0]

