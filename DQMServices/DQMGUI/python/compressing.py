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
        # we do a conversion from "array of structs" to "struct of arrays".
        # for that, we need the format characters (without the "<")
        fstring = cls.normal_format.format[1:]
        normal_format = cls.normal_format
        # SoA
        lists = [[] for _ in fstring]
        # current value fot the delta-encoding
        cur = [0 for _ in fstring]
        for info in infos:
            # convert into bare words (explicitly ignoring the special cases)
            b = cls.pack_meinfo(info)
            words = normal_format.unpack(b)
            # delta-encode
            for i in range(len(fstring)):
                lists[i].append(words[i] - cur[i])
                cur[i] = words[i]
        # now we pack the SoA into blobs (again).
        blobs = []
        for i, l in enumerate(lists):
            blobs.append(struct.pack(f"<{len(l)}{fstring[i]}", *l))
        blob = b"".join(blobs)
        # and finally compress it. This takes negligible time compared to the pure Python logic above.
        ziped = zlib.compress(blob)
        return ziped

    @classmethod
    async def uncompress_names_blob(cls, names_blob):
        def uncompress_sync():
            return zlib.decompress(names_blob).splitlines()
        return await asyncio.get_event_loop().run_in_executor(None, uncompress_sync)


    @classmethod
    async def uncompress_infos_blob(cls, infos_blob):
        # Here, we undo the transform from `compress_infos_list`.
        fstring = cls.normal_format.format[1:]
        normal_format = struct.Struct(fstring)
        # agagin, this is very fast -- could be in threadpool, but does no really need to.
        buf = zlib.decompress(infos_blob)
        n = len(buf) // normal_format.size
        bufs = []
        sizes = []
        start = 0
        for t in fstring:
            # Unpack SoA
            s = struct.Struct(f"<{n}{t}")
            sizes.append(s.size // n)
            l = list(s.unpack_from(buf, offset=start))
            start += s.size
            # delta decode
            cur = 0
            for i in range(n):
                cur += l[i]
                l[i] = cur
            # and repack SoA! We will use `unpack_meinfo` later, which expects bytes.
            bufs.append(s.pack(*l))
        # All further steps can be done on-demand. We only need to do the 
        # delta-decode here, the reads can be done per-element (and most likely
        # we will only read very few elements).
        # Inner class defined here so we can access cls. This allows us to 
        # also access all the other state via the closure.
        class LazyMEInfoList:
            """
            This represents a list of MEInfo objects. However, the objects are only
            created form their binary representation lazyliy as they are accessed.
            """
            def __getitem__(self, idx):
                parts = []
                # read one entry out of each of the arrays
                for i, buf in enumerate(bufs):
                    parts.append(buf[idx*sizes[i] : (idx+1)*sizes[i]])
                # then merge them into one struct, and decode the object from that.
                return  cls.unpack_meinfo(b''.join(parts))
            def __len__(self):
                return n
        return LazyMEInfoList()


    
    @classmethod
    def pack_meinfo(cls, me_info):
        if me_info.type == b'Int':
            buffer = cls.int_format.pack(me_info.value)
            return cls.scalar_format.pack(*buffer, cls.type_to_id[me_info.type])
        elif me_info.type == b'Float':
            buffer = cls.float_format.pack(me_info.value)
            return cls.scalar_format.pack(*buffer, cls.type_to_id[me_info.type])
        
        return cls.normal_format.pack(me_info.seekkey, me_info.offset, me_info.size, 
                                        cls.type_to_id[me_info.type], me_info.qteststatus)


    @classmethod
    def unpack_meinfo(cls, buffer):
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
