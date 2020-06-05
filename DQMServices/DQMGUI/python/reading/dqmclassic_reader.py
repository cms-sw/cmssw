import mmap
import asyncio
from DQMServices.DQMGUI import nanoroot
from data_types import ScalarValue, EfficiencyFlag, QTest
from ioservice import IOService


class DQMCLASSICReader:

    ioservice = IOService()

    @classmethod
    async def read(cls, filename, me_info):
        """
        Possible return values: ScalarValue, EfficiencyFlag, QTest, bytes
        """

        # To read int and float we don't need to go to the file
        if me_info.value != None:
            return ScalarValue(b'', b'', me_info.value) # TODO: do sth. better.

        buffer = await cls.ioservice.open_url(filename)
        key = await nanoroot.TKey().load(buffer, me_info.seekkey)
        data = await key.objdata()
        if me_info.type == b'QTest':
            return cls.parse_string_entry(await key.objname())
        if me_info.type == b'XMLString':
            return cls.parse_string_entry(await key.objname())
        if me_info.offset == 0 and me_info.size == -1:
            obj = data
        else:
            obj = data[me_info.offset : me_info.offset + me_info.size]
        if me_info.type == b'String':
            s = nanoroot.String.unpack(obj, 0, len(obj), None)
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
        return nanoroot.TBufferFile(obj, me_info.type, displacement, classversion)


    @classmethod
    def parse_string_entry(cls, string):
        """
        Non-object data is stored in fake-XML stings in the TDirectory.
        This decodes these strings into an object of correct type.
        Possible return types: EfficiencyFlag, ScalarValue, QTest
        """

        assert string[0] == b'<'[0]
        name = string[1:].split(b'>', 1)[0]
        value = string[1 + len(name)+1:].split(b'<', 1)[0]

        if value == b"e=1": 
            # Efficiency flag on this ME
            return EfficiencyFlag(name)
        elif len(value) >= 2 and value[1] == b'='[0]:
            return ScalarValue(name, type=value[0:1], value=value[2:])
        else: 
            # Should be a qtest in this case
            assert value.startswith(b'qr=')
            assert b'.' in name
            mename, qtestname = name.split(b'.', 1)
            parts = value[3:].split(b':', 4)
            assert len(parts) == 5, "Expect 5 parts, not " + repr(parts)
            x, status, result, algorithm, message = parts
            assert x == b'st'
            return QTest(mename, qtestname, status, result, algorithm, message)