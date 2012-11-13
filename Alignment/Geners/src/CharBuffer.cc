#include <cstring>
#include <climits>
#include <cassert>
#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/CharBuffer.hh"

namespace gs {
    bool CharBuffer::write(std::ostream& os) const
    {
        unsigned long long tmp = 0ULL;
        const char* buf = buf_.getPutBuffer(&tmp);
        write_pod(os, tmp);
        os.write(buf, tmp);
        return !os.fail();
    }

    unsigned long CharBuffer::size() const
    {
        unsigned long long tmp = 0ULL;
        buf_.getPutBuffer(&tmp);
        if (tmp > ULONG_MAX) 
            throw gs::IOLengthError("In CharBuffer::size: buffer is too large");
        return static_cast<unsigned long>(tmp);
    }

    void CharBuffer::restore(const ClassId& id, std::istream& in,
                             CharBuffer* buf)
    {
        static const ClassId current(ClassId::makeId<CharBuffer>());

        assert(buf);
        current.ensureSameId(id);

        unsigned long long tmp = 0ULL;
        read_pod(in, &tmp);
        buf->clear();
        buf->seekp(0);
        
        const unsigned locLen = 4096;
        char local[locLen];
        std::streambuf* inbuf = in.rdbuf();
        std::streambuf* outbuf = buf->rdbuf();

        while (tmp > locLen)
        {
            inbuf->sgetn(local, locLen);
            outbuf->sputn(local, locLen);
            tmp -= locLen;
        }
        if (tmp)
        {
            inbuf->sgetn(local, tmp);
            outbuf->sputn(local, tmp);
        }
        if (in.fail()) throw IOReadFailure(
            "In gs::CharBuffer::restore: input stream failure");
        if (buf->fail()) throw IOWriteFailure(
            "In gs::CharBuffer::restore: buffer stream failure");
    }

    bool CharBuffer::operator==(const CharBuffer& r) const
    {
        unsigned long long tmp = 0ULL;
        const char* buf = buf_.getPutBuffer(&tmp);
        unsigned long long tmp2 = 0ULL;
        const char* buf2 = r.buf_.getPutBuffer(&tmp2);
        if (tmp != tmp2)
            return false;
        return memcmp(buf, buf2, tmp) == 0;
    }
}
