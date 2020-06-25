import struct

# A helper to format TBufferFile data. All data in ROOT files is serialized
# using TBufferFile serialization, but the headers added to the data vary. This
# class removes any detected headers and adds the headers needed for a bare object
# that can be read using `ReadObjectAny`. Since TBufferFile data can contain
# references into itself, we need to keep track of where the buffer actually 
# started (`displacement` parameters).

# Usually data in ROOT files comes with at least the version header attached.
# If it does not, we will add the version number here, but then the correct
# class version needs to be passed in here. But there is no way in nanoroot to
# detect what the version is, in that case. It turns out, that for many TH1
# derived classes, the correct version is 3.

class TBufferFile():
    def __init__(self, objdata, classname, displacement=0, version=None, is_raw=False):
        """
        If is_raw is True, buffer will be set to objdata without other parsig or validation. This is
        used when creating TBufferFile from data read from protobuf file.
        """

        if is_raw:
            self.buffer = objdata
            self.displacement = 0
        else:
            if objdata[0:1] != b'@' and objdata[1:2] == b'T': 
                # This came out of a branch (TBranchObject?) with class and version header.
                clslen = objdata[0]
                cls = objdata[1:1+clslen]
                assert cls == classname, f"Classname {repr(cls)} from Branch should match {repr(classname)}"
                objdata = objdata[clslen+2:] # strip class and continue.
                displacement -= clslen + 2

            # @-decode and see if that could be a version header.
            size, = struct.unpack(">I", objdata[0:4])
            assert (size & 0x40000000) > 0, "That does not look like a ROOT object."
            size = (size & ~0x40000000) + 4
            assert size <= len(objdata), "Sub-object seems too big."
            if size != len(objdata):
                # this does not look like a version header. Add one.
                totlen = 2 + len(objdata)
                head = struct.pack(">IH", totlen | 0x40000000, version)
                objdata = head + objdata
                displacement += len(head)

            # The format is <@length><kNewClassTag=0xFFFFFFFF><classname><nul><@length><2 bytes version><data ...
            # @length is 4byte length of the *entire* remaining object with bit 0x40 (kByteCountMask)
            # set in the first (most significant) byte. This prints as "@" in the dump...
            # the data inside the TKey seems to have the version already.
            totlen = 4 + len(classname) + 1 + len(objdata)
            head = struct.pack(">II", totlen | 0x40000000, 0xFFFFFFFF)
            self.buffer =  head + classname + b'\0' + objdata
            displacement += len(head) + len(classname) + 1

            # The TBufferFile data can contain references into itself, when an 
            # already stored object is used again. This happens especially with
            # class definitions, which can also be re-used. Since these offsets are
            # absolute offsets into the buffer, they will be wrong if we add or
            # remove to/from the beginning of the buffer. To compensate for that,
            # we can use the `SetDisplacement` option, but we need to keep track of
            # the correct displacement here.
            # Getting it wrong usually does not matter, but can lead to missing
            # Objects (axis, labels, ...) inside the histograms and also sometimes
            # ROOT crashes due to infinite recursion.
            self.displacement = displacement

