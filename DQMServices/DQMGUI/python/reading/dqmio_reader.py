import asyncio
from nanoroot.tfile import TKey
from nanoroot.tbufferfile import TBufferFile
from nanoroot.ttree import TType
from data_types import ScalarValue
from ioservice import IOService


class DQMIOReader:

    ioservice = IOService()

    @classmethod
    async def read(cls, filename, me_info):
        """
        Possible return values: ScalarValue, EfficiencyFlag, QTest, bytes
        """

        # To read int and float we don't need to go to the file
        if me_info.value != None:
            return ScalarValue(b'', b'', me_info.value) # TODO: do sth. better.

        buffer = await cls.ioservice.open_url(filename, blockcache=True)
        key = await TKey().load(buffer, me_info.seekkey)
        data = await key.objdata()
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
        # TODO: not sure if this does in fact work for TTrees.
        displacement = - key.fields.fKeyLen - me_info.offset
        # metype doubles as root class name here.
        return TBufferFile(obj, me_info.type, displacement, classversion)

