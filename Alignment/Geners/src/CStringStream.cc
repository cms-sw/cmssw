#include <cassert>
#include <cstring>

#include "zlib.h"

#include "Alignment/Geners/interface/BZ2Handle.hh"
#include "Alignment/Geners/interface/CStringStream.hh"
#include "Alignment/Geners/interface/IOException.hh"

static void doZlibCompression(const char* data, const unsigned long long len,
                              const bool defl, z_stream_s& strm,
                              char* buffer, const unsigned long long bufLen,
                              std::ostream& sink)
{
    assert(buffer);
    assert(bufLen);

    int status = Z_OK;
    strm.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data));
    strm.avail_in = len;
    do 
    {
        strm.next_out = reinterpret_cast<Bytef*>(buffer);
        strm.avail_out = bufLen;
        status = defl ? deflate(&strm, Z_FINISH) :
                        inflate(&strm, Z_NO_FLUSH);
        assert(status == Z_OK || status == Z_STREAM_END);
        const unsigned have = bufLen - strm.avail_out;
        sink.write(buffer, have);
        if (sink.fail()) throw gs::IOWriteFailure(
            "In gs::doZlibCompression: sink stream failure");
    } while (strm.avail_out == 0);

    if (defl)
    {
        assert(strm.avail_in == 0);
        assert(status == Z_STREAM_END);
        assert(deflateReset(&strm) == Z_OK);
    }
    else
        assert(inflateReset(&strm) == Z_OK);
}


static void doBZ2Compression(const char* data, const unsigned long long len,
                             const bool defl, bz_stream& strm,
                             char* buffer, const unsigned long long bufLen,
                             std::ostream& sink)
{
    assert(buffer);
    assert(bufLen);

    int status = BZ_OK;
    strm.next_in = const_cast<char*>(data);
    strm.avail_in = len;
    do 
    {
        strm.next_out = buffer;
        strm.avail_out = bufLen;
        status = defl ? BZ2_bzCompress(&strm, BZ_FINISH) :
                        BZ2_bzDecompress(&strm);
        assert(status == BZ_OK || status == BZ_STREAM_END);
        const unsigned have = bufLen - strm.avail_out;
        sink.write(buffer, have);
        if (sink.fail()) throw gs::IOWriteFailure(
            "In gs::doBZ2Compression: sink stream failure");
    } while (status != BZ_STREAM_END);
}


namespace gs {
    CStringStream::CStringStream(const CompressionMode m,
                                 const int compressionLevel,
                                 const unsigned minSizeToCompress,
                                 const unsigned bufSize)
        : mode_(m),
          compressionLevel_(compressionLevel),
          minSizeToCompress_(minSizeToCompress),
          // Have a reasonable minimum buffer size to maintain
          // performance even if the user wants to shoot himself
          // in a foot
          comprBuf_(bufSize > 1024U ? bufSize : 1024U),
          sink_(0)
    {
        this->init(&buf_);
    }

    void CStringStream::setCompressionMode(const CompressionMode newmode)
    {
        reset();
        mode_ = newmode;
    }

    void CStringStream::reset()
    {
        clear();
        seekp(0);
        seekg(0);
    }

    void CStringStream::readCompressed(std::istream& in,
                                       const unsigned compressionCode,
                                       const unsigned long long len)
    {
        reset();
        if (!len)
            return;

        // Decompress and dump to this string stream
        if (len > readBuf_.size())
            readBuf_.resize(len);
        in.read(&readBuf_[0], len);

        switch (static_cast<CompressionMode>(compressionCode))
        {
        case NOT_COMPRESSED:
            this->write(&readBuf_[0], len);
            return;

        case ZLIB:
        {
            if (!inflator_.get())
                inflator_ = CPP11_auto_ptr<ZlibInflateHandle>(
                    new ZlibInflateHandle());
            doZlibCompression(&readBuf_[0], len, false, inflator_->strm(),
                              &comprBuf_[0], comprBuf_.size(), *this);
        }
        break;

        case BZIP2:
        {
            // bzlib2 can not be reset, so we have to make
            // a new inflator every time
            bz_stream strm;
            BZ2InflateHandle h(strm);
            doBZ2Compression(&readBuf_[0], len, false, strm,
                             &comprBuf_[0], comprBuf_.size(), *this);
        }
        break;

        default:
            assert(!"Unhandled switch case in "
                   "CStringStream::readCompressed. "
                   "This is a bug. Please report.");
        }
    }

    CStringStream::CompressionMode CStringStream::writeCompressed()
    {
        // Compress and dump to sink
        assert(sink_);

        unsigned long long len = 0;
        const char* data = buf_.getPutBuffer(&len);
        if (len == 0)
            return NOT_COMPRESSED;

        if (mode_ == NOT_COMPRESSED || len < minSizeToCompress_)
        {
            sink_->write(data, len);
            return NOT_COMPRESSED;
        }

        switch (mode_)
        {
        case ZLIB:
        {
            if (!deflator_.get())
                deflator_ = CPP11_auto_ptr<ZlibDeflateHandle>(
                    new ZlibDeflateHandle(compressionLevel_));
            doZlibCompression(data, len, true, deflator_->strm(),
                              &comprBuf_[0], comprBuf_.size(), *sink_);
        }        
        break;

        case BZIP2:
        {
            // bzlib2 can not be reset, so we have to make
            // a new deflator every time
            bz_stream strm;
            BZ2DeflateHandle h(strm);
            doBZ2Compression(data, len, true, strm,
                             &comprBuf_[0], comprBuf_.size(), *sink_);
        }
        break;

        default:
            assert(!"Unhandled switch case in "
                   "CStringStream::writeCompressed. "
                   "This is a bug. Please report.");
        }

        seekp(0);
        return mode_;
    }

    bool CStringStream::getCompressionModeByName(const char* name,
                                                 CompressionMode* m)
    {
        static const char* names[] = {
            "n",
            "z",
            "b"
        };
        if (!name || !m)
            return false;
        for (unsigned i=0; i<sizeof(names)/sizeof(names[0]); ++i)
            if (strcasecmp(name, names[i]) == 0)
            {
                *m = static_cast<CompressionMode>(i);
                return true;
            }
        return false;
    }

    std::string CStringStream::compressionModeName(const CompressionMode m,
                                                   const bool useShortName)
    {
        std::string mode;
        switch (m)
        {
        case NOT_COMPRESSED:
            mode = useShortName ? "n" : "not compressed";
            break;
        case ZLIB:
            mode = useShortName ? "z" : "zlib";
            break;
        case BZIP2:
            mode = useShortName ? "b" : "bzip2";
            break;
        default:
            assert(!"Unhandled switch case in "
                   "CStringStream::compressionModeName. "
                   "This is a bug. Please report.");
        }
        return mode;
    }
}
