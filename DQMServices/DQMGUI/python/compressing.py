import zlib
import struct
import asyncio
from data_types import MEInfo

class GUIBlobCompressor():
    """This class is responsible for compressing/uncompressing ME names and infos blobs."""

    # These match the DQMIO encoding where possible. But they don't really need to.
    id_to_type = {
        0: b"Int",
        1: b"Float",
        2: b"String",
        3: b"TH1F",
        4: b"TH1S",
        5: b"TH1D",
        6: b"TH2F",
        7: b"TH2S",
        8: b"TH2D",
        9: b"TH3F",
        10: b"TProfile",
        11: b"TProfile2D",
        20: b"Flag",
        21: b"QTest",
        22: b"XMLString", # For string type in TDirectory
    }
    type_to_id = {v: k for k, v in id_to_type.items()}

    # These are used to store the MEInfo into a blob. The scalar
    # version stores the value directly. They are all the
    # same size to allow direct indexing.
    normal_format = struct.Struct("<qiihh")
    # This is a bit of a hack: the deltaencode below needs all topmost bits to be unused.
    # so we spread the double and int64 values for scalars with a bit of padding to keep them free.
    scalar_format = struct.Struct("<xBBBBBBxxBBxxxxxHxx") 
    int_format    = struct.Struct("<q")
    float_format  = struct.Struct("<d")


    @classmethod
    async def compress_names_list(cls, names):
        def compress_buffer_sync():
            return zlib.compress(names)
        names_blob = await asyncio.get_event_loop().run_in_executor(None, compress_buffer_sync)
        return names_blob


    @classmethod
    async def compress_infos_list(cls, infos):
        """
        For the ME infos, Zlib compression is not very effective: there is
        little repetition for the LZ77 and the Huffmann coder struggles with the
        large numbers.
        But, since the list is roughly increasing in order, delta coding makes most
        values small. Then, the Huffmann coder in Zlib compresses well.
        Decreases output size about 4x.
        """

        words = cls.normal_format
        def delta_encode(a):
            prev = [0, 0, 0, 0, 0]
            for x in a:
                new = words.unpack(x)
                yield words.pack(*[a-b for a, b in zip(new, prev)])
                prev = new
        delta = delta_encode(cls.__pack(info) for info in infos)
        buffer = b''.join(delta)
        def compress_buffer_sync():
            return zlib.compress(buffer)
        infos_blob = await asyncio.get_event_loop().run_in_executor(None, compress_buffer_sync)
        return infos_blob


    @classmethod
    async def uncompress_names_blob(cls, names_blob):
        def uncompress_sync():
            return zlib.decompress(names_blob).splitlines()
        return await asyncio.get_event_loop().run_in_executor(None, uncompress_sync)


    @classmethod
    async def uncompress_infos_blob(cls, infos_blob):
        words = cls.normal_format
        def delta_decode(d):
            prev = [0, 0, 0, 0, 0]
            for x in d:
                new = [a+b for a, b in zip(prev, words.unpack(x))]
                yield words.pack(*new)
                prev = new
        buffer = zlib.decompress(infos_blob)
        packed = delta_decode(buffer[i:i+words.size] for i in range(0, len(buffer), words.size))
        return [cls.__unpack(x) for x in list(packed)]

    
    @classmethod
    def __pack(cls, me_info):
        if me_info.type == b'Int':
            buffer = cls.int_format.pack(me_info.value)
            return cls.scalar_format.pack(*buffer, cls.type_to_id[me_info.type])
        elif me_info.type == b'Float':
            buffer = cls.float_format.pack(me_info.value)
            return cls.scalar_format.pack(*buffer, cls.type_to_id[me_info.type])
        
        return cls.normal_format.pack(me_info.seekkey, me_info.offset, me_info.size, 
                                        cls.type_to_id[me_info.type], me_info.qteststatus)


    @classmethod
    def __unpack(cls, buffer):
        seekkey, offset, size, metype, qteststatus = cls.normal_format.unpack(buffer)
        metype = cls.id_to_type[metype]

        if metype == b'Int':
            buffer = cls.scalar_format.unpack(buffer)
            value, = cls.int_format.unpack(bytes(buffer[:-1])) # last is metype again
            return MEInfo(metype, value=value)
        elif metype == b'Float':
            buffer = cls.scalar_format.unpack(buffer)
            value, = cls.float_format.unpack(bytes(buffer[:-1])) # last is metype again
            return MEInfo(metype, value=value)
        
        return MEInfo(metype, seekkey=seekkey, offset=offset, size=size, qteststatus=qteststatus)
