import os
import asyncio
from ..ioservice import IOService
from ..data_types import ScalarValue
from ..nanoroot.tbufferfile import TBufferFile
from ..reading.reading import DQMCLASSICReader
from ..protobuf.protobuf_parser import ProtobufParser


class ProtobufReader:

    protobuf_parser = ProtobufParser()
    ioservice = IOService()

    @classmethod
    async def read(cls, filename, me_info):
        """
        Possible return values: ScalarValue, EfficiencyFlag, QTest, bytes
        """

        # To read int and float we don't need to go to the file
        if me_info.value != None:
            return ScalarValue(b'', b'', me_info.value) # TODO: do sth. better.

        buffer = await cls.ioservice.open_url(filename, True)
        buffer.seek(me_info.offset, os.SEEK_CUR)
        histo_message = await cls.protobuf_parser.read_histo_message(buffer, me_info.size, read_histogram_bytes=True, uncompress_histogram_bytes=True)

        if me_info.type == b'String':
            string_value = cls.get_tobjstring_content(histo_message.full_pathname, histo_message.streamed_histo)
            parsed = DQMCLASSICReader.parse_string_entry(string_value)
            value = parsed.value.decode('ascii')
            return ScalarValue(b'', b's', value)

        return TBufferFile(histo_message.streamed_histo, '', is_raw=True)


    @classmethod
    def get_tobjstring_content(cls, me_path, buffer):
        """There is a ROOT format prefix that we just drop to get the raw string content."""

        last_slash_index = me_path.rfind(b'/') + 1
        me_name = b'<' + me_path[last_slash_index:] + b'>'
        index = buffer.index(me_name)
        return buffer[index:]
