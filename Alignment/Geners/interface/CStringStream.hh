// Compressed string stream.
//
// Properties we need to have in this class:
//
// 1. Ability to use it as std::ostringstream for uncompressed writes.
//
// 2. Ability to use it as std::istringstream for uncompressed reads.
//
// 3. Ability to fill the compression buffer from an istream.
//
// 4. Ability to dump the compression buffer to an ostream.
//
// 5. Ability to convert data between compressed and uncompressed buffers.

#ifndef GENERS_CSTRINGSTREAM_HH_
#define GENERS_CSTRINGSTREAM_HH_

#include <vector>
#include <iostream>

#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/CStringBuf.hh"
#include "Alignment/Geners/interface/ZlibHandle.hh"

namespace gs {
    class CStringStream : public std::basic_iostream<char>
    {
    public:
        enum CompressionMode
        {
            NOT_COMPRESSED = 0,
            ZLIB,
            BZIP2
        };

        CStringStream(CompressionMode m, int compressionLevel,
                      unsigned minSizeToCompress, unsigned bufSize);

        // Basic inspectors
        inline CompressionMode compressionMode() const {return mode_;}
        inline int compressionLevel() const {return compressionLevel_;}
        inline unsigned minSizeToCompress() const {return minSizeToCompress_;}
        inline std::size_t bufferSize() const {return comprBuf_.size();}

        // "setCompressionMode" calls "reset" internally.
        // All unprocessed data will be lost.
        void setCompressionMode(CompressionMode m);

        // The sink must be set before calling "writeCompressed".
        // This is where the compressed data will be dumped.
        inline void setSink(std::ostream& os) {sink_ = &os;}

        // "writeCompressed" compresses the content of this stream
        // and dumps them to sink. The write pointer is repositioned
        // at the beginning of the stream. The compression mode is
        // returned (always NOT_COMPRESSED if the amount of data
        // was below "minSizeToCompress").
        CompressionMode writeCompressed();

        // Fill this stream from compressed data. Could be called
        // repetitively. Uncompressed data is appended at the end
        // (internal pointers are not reset).
        void readCompressed(std::istream& in, unsigned compressionCode,
                            unsigned long long len);

        // Reposition both read and write pointers of the stream
        // at the beginning of the stream
        void reset();

        // Parse the compression mode. Returns "true" on success.
        static bool getCompressionModeByName(const char* name,
                                             CompressionMode* m);

        // String representation of the compression mode
        static std::string compressionModeName(CompressionMode m,
                                               bool useShortName=true);
    private:
        CStringStream(const CStringStream&);
        CStringStream& operator=(const CStringStream&);

        CStringBuf buf_;
        CompressionMode mode_;
        int compressionLevel_;
        unsigned minSizeToCompress_;
        std::vector<char> comprBuf_;
        std::vector<char> readBuf_;
        std::ostream* sink_;
        CPP11_auto_ptr<ZlibInflateHandle> inflator_;
        CPP11_auto_ptr<ZlibDeflateHandle> deflator_;
    };
}

#endif // GENERS_CSTRINGSTREAM_HH_

