
import os
import zlib
from ..ioservice import IOService
from collections import namedtuple


class ProtobufParser:
    """
    Protocol buffers is binary serialization format from Google. Python library for deserializing protobuf 
    encoded messages exist, however, it doesn't support partial reads - if we want to get just a single 
    histogram out, we have to read and deserialize the entire file. This is an implementation of protocol 
    buffers format deserializer with limited functionality as we only need to deserialize fields used in our 
    messages. To see the supported message format, please have a look at ROOTFilePB.proto. This deserializer 
    supports reading any number of histograms from any location in the file without reading/deserializing 
    anything else.

    In the main scope of the message there is only one repeated Histo field. Repeated fields are like lists
    that can have 0 to infinitely many elements. Histo message contains 4 required fields that hold data
    and metadata about the ME.

    Bellow I will discuss the implemented parts of the format. For a complete format specification please 
    have a look here: https://developers.google.com/protocol-buffers/docs/encoding

    Data Formats:

    Full format specifies more data formats but we'll be using only two: variants and length-delimited values.

    Variant is a variable length number. For more info on how to deserialize it please look at the doc string
    of read_variant_value() function.

    Length-delimited value is an arbitrary length blob. For more information on how to deserialize it please 
    look at the doc string of read_length_delimited_value() function.

    Wire type reveals us the type of the value and tells us how to read it. Continue reading the Message 
    format section for more details.

    Message format:

    Messages are stored in key value pairs. Key is always of variant type. 3 least significant bits of this
    variant encode the wire type and remaining preceding bits encode the field number of the following value.
    For more information on how to extract wire type and field number from a variant please look at the doc
    string of read_field_number_and_wire_type() method.

    Every odd numbered element (1, 3, 5, ...) in the main message, as well as in every embedded message, is 
    a variant that splits into wire type and field number - a key. 
    Every even numbered element in the main message, as well as in every embedded message, is a value that we 
    read differently based on wire type.

    Wire type tells us the length of the value and instructions on how to read it. 
    Field number is just a number that was assigned to every field in the message so we could identify what
    field we have just deserialized. You can see what field corresponds to what field number in 
    ROOTFilePB.proto file.

    Since we only have a few data types in this implementation, we only care about two wire types: 
    0 - variant
    2 - Length-delimited value

    Embedded message, string and bytes will use wire type 2 and uint32 will use variant.

    Backwards compatibility:

    Protobuf format ensures that updated messages can be read with old serializers provided no breaking
    changes occur. All fields that we don't know about are just skipped. 

    Partial reading:

    The crux of this code is to be able to read only one histogram without having to read and deserialize 
    the entire file.

    read_histo_message() function does just that. Give it a buffer that is seeked to the correct position
    and a size of the message and it will return only one histo message without reading more bytes than
    required.
    """


    # This is a type that hold a decoded message. seek_key and offset are not within the message,
    # they are added by the parser.
    # A list of this tuple will be returned when reading from file.
    HistoMessage = namedtuple('HistoMessage', ['full_pathname', 'size', 'streamed_histo', 'flags', 'offset', 'message_size'])

    # Wire type tells us the length of the value and instructions on how to read it
    WIRE_TYPE_LENGTHS = { 0: 'variant', 1: 8, 2: 'length-delimited', 3: 0, 4: 0, 5: 4 }

    # Wire types that are suppored by protobuf format.
    # If we encounter a wire type that is not from this list, we are doing something wrong or the file is corrupted.
    WIRE_TYPES = WIRE_TYPE_LENGTHS.keys()
    
    ioservice = IOService()


    @classmethod
    async def deserialize_file(cls, filename, read_histogram_bytes=False, uncompress_histogram_bytes=True):
        """
        Parses non-gzipped protobuf file and returns a list of HistoMessage tuples.
        If read_histogram_bytes is True, binary data of actual histograms will be read, otherwise it won't.
        If uncompress_histogram_bytes is True, binary data of the histograms will be zlib uncompressed.
        If you need only metadata, set read_histogram_bytes to False.
        """

        histos = []

        buffer = await cls.ioservice.open_url(filename, blockcache=False)
        while True:
            field_number, wire_type = await cls.read_field_number_and_wire_type(buffer)
            
            if field_number == 1 and wire_type == 2:
                # Found a value of the repeated Histo field, parse it!
                message_size = await cls.read_variant_value(buffer)
                histo = await cls.read_histo_message(buffer, message_size, read_histogram_bytes, uncompress_histogram_bytes)
                histos.append(histo)
            else:
                await cls.consume_unknown_field(buffer, wire_type)

            # Break out if file is over
            if await buffer.peek(1) == b'':
                break

        return histos


    @classmethod
    async def read_histo_message(cls, buffer, message_size, read_histogram_bytes=False, uncompress_histogram_bytes=True):
        """Read Histo message and parse its fields"""

        # Values that will be returned in HistoMessage tuple
        full_pathname = ''
        size = 0
        streamed_histo = None
        flags = 0
        offset = buffer.seek(0, os.SEEK_CUR)

        buffer = AsyncBufferView(buffer, message_size)
        histo = {}

        while True:
            field_number, wire_type = await cls.read_field_number_and_wire_type(buffer)

            if field_number == 1 and wire_type == 2:
                full_pathname = await cls.read_length_delimited_value(buffer)
            elif field_number == 2 and wire_type == 0:
                size = await cls.read_variant_value(buffer)
            elif field_number == 3 and wire_type == 2:
                if read_histogram_bytes:
                    streamed_histo = await cls.read_length_delimited_value(buffer)
                    if uncompress_histogram_bytes:
                        streamed_histo = zlib.decompress(streamed_histo)
                else: 
                    # If we don't need the histogram, just seek through it
                    await cls.consume_unknown_field(buffer, wire_type)
            elif field_number == 4 and wire_type == 0:
                flags = await cls.read_variant_value(buffer)
            else:
                await cls.consume_unknown_field(buffer, wire_type)

            if await buffer.peek(1) == b'':
                break

        return cls.HistoMessage(full_pathname, size, streamed_histo, flags, offset, message_size)


    @classmethod
    async def read_variant_value(cls, buffer):
        """
        Variants are variable length integers. Their wire type is 0.
        We read one byte at a time and check the most significant bit (msb) of each byte.
        If msb is 1, this is not the last byte of an integer.
        If msb is 0, this is the last byte of an integer (we still include it).

        Msb is just to tell us if there are more bytes to come or not. When decoding we drop it!!!

        Variants store numbers with the least significant group first, so when decoding, 
        groups of 7 bits have to be combined in a reversed order.

        Example:
        We know that the following message has a wire_type of 0 (it's a variant) and we want to decode it:

        1010 1100 0000 0010

        As you can see, msb of the first byte (8 bits) is 1, so we read the next byte too.
        Msb of the second byte is 0 meaning that it's the final byte we need to read.
        Now we drop both msbs:

        010 1100 000 0010

        And combine the groups in reverse order:

        000 0010 010 1100

        This is the binary representation of the decoded number:

        0000 0001 0010 1100

        Or 300 in decimal.
        """

        value = 0
        number_of_bytes = 0
        msb = 1 # We read bytes one by one until most significant bit is 0.
        while msb == 1:
            byte = await buffer.read(1)
            data = int.from_bytes(byte, byteorder='little', signed=False)
            msb = (data >> 7) & 1

            # Set msb to 0:
            data &= ~(1 << 7)

            # Variants are stored with least significant group of 7 bits first
            value |= data << (7 * number_of_bytes)
            number_of_bytes += 1

        return value


    @classmethod
    async def read_length_delimited_value(cls, buffer):
        """
        Length-delimited value is an arbitrary length blob (string, bytes or embedded message). Its wire type is 2.
        Length-delimited value consists of two parts: a variant denoting it's length followed by a blob of that length.

        To decode a length-delimited value we first read a variant to figure out its size and read that many bytes 
        from the stream.
        """

        size = await cls.read_variant_value(buffer)
        value = await buffer.read(size)
        return value


    @classmethod
    async def read_field_number_and_wire_type(cls, buffer):
        """
        Wire type is encoded in the last (least significant) 3 bits of a variant.
        Field number is encoded in all other preceding bits.
        """

        variant = await cls.read_variant_value(buffer)

        field_number = variant >> 3 # will drop last 3 bits
        wire_type = variant & 0b111 # will get last 3 bits

        assert wire_type in cls.WIRE_TYPES, 'Incorrect wire_type: %s' % wire_type

        return field_number, wire_type


    @classmethod
    async def consume_unknown_field(cls, buffer, wire_type):
        """If we encounter a field number that we don't know how to or don't want to deserialize, we seek through it."""

        size = cls.WIRE_TYPE_LENGTHS[wire_type]
        if isinstance(size, int):
            buffer.seek(size, os.SEEK_CUR)
        elif size == 'variant':
            # We have to actually read and inspect msb of every byte to know when to stop reading
            await cls.read_variant_value(buffer)
        elif size == 'length-delimited':
            message_length = await cls.read_variant_value(buffer)
            buffer.seek(message_length, os.SEEK_CUR)


class AsyncBufferView:
    """
    Provides an async view of an async buffer. 
    Will return b'' when view_size is exceeded even if the underlying buffer is still not over.
    """

    def __init__(self, buffer, view_size):
        self.buffer = buffer
        self.view_size = view_size
        self.position = 0
        self.underlying_buffer_position = buffer.seek(0, os.SEEK_CUR)


    def seek(self, offset, whence=os.SEEK_SET):
        """
        Changes current position pointer.
        os.SEEK_SET or 0 - start of the stream (the default); offset should be zero or positive
        os.SEEK_CUR or 1 - current stream position; offset may be negative
        """

        if whence == os.SEEK_SET:
            self.position = offset
            self.underlying_buffer_position = self.buffer.seek(self.underlying_buffer_position + offset, os.SEEK_SET)
        elif whence == os.SEEK_CUR:
            self.position += offset
            self.underlying_buffer_position = self.buffer.seek(offset, os.SEEK_CUR)

        return self.position


    async def peek(self, size):
        """Return bytes from the stream without advancing the position."""
        
        size, eof = self.__get_size_safe(size)
        data = await self.buffer.peek(size)
        return data + b'' if eof else data
    

    async def read(self, size):
        """Read up to size bytes from the object and return them."""

        size, eof = self.__get_size_safe(size)
        data = await self.buffer.read(size)
        self.position += size
        return data + b'' if eof else data


    def __get_size_safe(self, requested_size):
        """
        Returns how many bytes we can read without exceeding view size. 
        If second value is True, b'' has to be returned.
        """

        eof = False
        size = requested_size

        # Make sure we don't exceed the view size
        if self.position + size > self.view_size:
            size = self.view_size - self.position
            eof = True

        return (size, eof)
