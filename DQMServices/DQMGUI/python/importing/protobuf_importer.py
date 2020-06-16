import asyncio
from ..data_types import MEInfo
from ..reading.reading import DQMCLASSICReader, ProtobufReader
from ..protobuf.protobuf_parser import ProtobufParser


class ProtobufImporter:

    # Flags encode the type of the ME
    FLAG_TO_TYPE = {
        0x00000001: b'Int',
        0x00000002: b'Float',
        0x00000003: b'String',
        0x00000010: b'TH1F',
        0x00000011: b'TH1S',
        0x00000012: b'TH1D',
        0x00000020: b'TH2F',
        0x00000021: b'TH2S',
        0x00000022: b'TH2D',
        0x00000030: b'TH3F',
        0x00000040: b'TProfile',
        0x00000041: b'TProfile2D',
    }
    TYPE_FLAGS = FLAG_TO_TYPE.keys()
    EFFICIENCY_FLAG = 0x00200000

    protobuf_parser = ProtobufParser()

    @classmethod
    async def get_me_lists(cls, filename, dataset, run, lumi):
        me_paths = [] 
        me_infos = []

        histo_messages = cls.protobuf_parser.deserialize_file(filename, read_histogram_bytes=True)

        for histo_message in histo_messages:
            me_type = cls.get_me_type(histo_message.flags)

            if me_type in (b'Int', b'Float'):
                string_value = ProtobufReader.get_tobjstring_content(histo_message.full_pathname, histo_message.streamed_histo)
                parsed = DQMCLASSICReader.parse_string_entry(string_value)
                value = parsed.value.decode('ascii')

                if me_type == b'Int':
                    value = int(value)
                elif me_type == b'Float':
                    value = float(value)

                me_info = MEInfo(me_type, value=value)
            else:
                me_info = MEInfo(me_type, 0, histo_message.offset, histo_message.message_size, None, 0)

            me_paths.append(histo_message.full_pathname)
            me_infos.append(me_info)

        result = list(zip(me_paths, me_infos))
        return { (run, 0): result }

        
    @classmethod
    def get_me_type(cls, flags):
        """Type is encoded in a least significant byte."""

        type_byte = flags & 255
        return cls.FLAG_TO_TYPE[type_byte]
