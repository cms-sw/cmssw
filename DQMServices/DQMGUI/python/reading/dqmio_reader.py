import asyncio

from async_lru import alru_cache

from ..helpers import logged
from ..ioservice import IOService
from ..data_types import ScalarValue
from ..nanoroot.tfile import TKey
from ..nanoroot.ttree import TType
from ..nanoroot.tbufferfile import TBufferFile


class DQMIOReader:

    ioservice = IOService()
    
    # Baskets are quite big (MB's compressd, 10's od MBs decompressed as cached
    # here, so we want this cache. But it will take a few hand full of baskets
    # for one sample, so 1 is not sufficient.
    @classmethod
    @alru_cache(maxsize=200)
    @logged
    async def read_basket(cls, filename, seekkey):
         buffer = await cls.ioservice.open_url(filename, blockcache=True)
         key = await TKey().load(buffer, seekkey)
         data = await key.objdata()
         # keylen is need to compute displacement later.
         return data, key.fields.fKeyLen


    @classmethod
    async def read(cls, filename, me_info):
        """
        Possible return values: ScalarValue, EfficiencyFlag, QTest, bytes
        """

        # To read int and float we don't need to go to the file
        if me_info.value != None:
            return ScalarValue(b'', b'', me_info.value) # TODO: do sth. better.

        data, keylen = await cls.read_basket(filename, me_info.seekkey)
        obj = data[me_info.offset : me_info.offset + me_info.size]
        if me_info.type == b'String':
            s = TType.String.unpack(obj, 0, len(obj), None)
            return ScalarValue(b'', b's', s)
        
        # Usually this value is unused since the class version is already in
        # the buffer. Only needed for some Trees in DQMIO.
        #TODO: we need to have a better guess here...
        classversion = 3 
        # The buffers in a TKey based file start with the TKey. Since we only 
        # send the object to the renderer, we need to compensate for that using
        # the displacement.
        displacement = - keylen - me_info.offset
        # metype doubles as root class name here.
        return TBufferFile(obj, me_info.type, displacement, classversion)

