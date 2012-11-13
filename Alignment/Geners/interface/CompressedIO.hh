// This is a simple high-level driver to write items into streams
// in a compressed form. Useful if, due to some reason, we do not
// want to write directly into a compressed archive.
//
// Note that this code is rather slow, and is not recommended for
// saving a lot of small objects (use a compressed archive instead).

#ifndef GENERS_COMPRESSEDIO_HH_
#define GENERS_COMPRESSEDIO_HH_

#include "Alignment/Geners/interface/GenericIO.hh"
#include "Alignment/Geners/interface/CStringStream.hh"

namespace gs {
    // The following function returns "true" on success, "false" on failure
    template <class Item>
    bool write_compressed_item(std::ostream& os, const Item& item,
             CStringStream::CompressionMode m = CStringStream::ZLIB,
             int compressionLevel = -1, unsigned minSizeToCompress = 1024U,
             unsigned bufSize = 1048576U);

    template <class Item>
    void restore_compressed_item(std::istream& in, Item* item);

    template <class Item>
    CPP11_auto_ptr<Item> read_compressed_item(std::istream& in);
}

namespace gs {
    template <class Item>
    inline bool write_compressed_item(std::ostream& os, const Item& item,
             const CStringStream::CompressionMode m,
             const int compressionLevel, const unsigned minSizeToCompress,
             const unsigned bufSize)
    {
        CStringStream cs(m, compressionLevel, minSizeToCompress, bufSize);
        unsigned compressionCode = 0;
        long long len = 0;
        const std::streampos base = os.tellp();
        write_pod(os, len);
        write_pod(os, compressionCode);
        if (os.fail() || os.bad())
            return false;
        const std::streampos start = os.tellp();
        cs.setSink(os);
        if (!write_item(cs, item))
            return false;
        cs.flush();
        compressionCode = static_cast<unsigned>(cs.writeCompressed());
        const std::streampos now = os.tellp();
        const std::streamoff off = now - start;
        len = off;
        os.seekp(base);
        write_pod(os, len);
        write_pod(os, compressionCode);
        os.seekp(now);
        return !(cs.fail() || cs.bad() || os.fail() || os.bad());
    }

    template <class Item>
    inline void restore_compressed_item(std::istream& is, Item* item)
    {
        long long len;
        read_pod(is, &len);
        unsigned compressionCode;
        read_pod(is, &compressionCode);
        CStringStream::CompressionMode m = 
            static_cast<CStringStream::CompressionMode>(compressionCode);
        CStringStream cs(m, -1, 1024U, 1048576U);
        cs.readCompressed(is, compressionCode, len);
        if (!is.good())
            throw IOReadFailure("In restore_compressed_item: "
                                "input stream failure");
        restore_item(cs, item);
    }

    template <class Item>
    inline CPP11_auto_ptr<Item> read_compressed_item(std::istream& is)
    {
        long long len;
        read_pod(is, &len);
        unsigned compressionCode;
        read_pod(is, &compressionCode);
        CStringStream::CompressionMode m = 
            static_cast<CStringStream::CompressionMode>(compressionCode);
        CStringStream cs(m, -1, 1024U, 1048576U);
        cs.readCompressed(is, compressionCode, len);
        if (!is.good())
            throw IOReadFailure("In read_compressed_item: "
                                "input stream failure");
        return read_item<Item,std::istream>(cs);
    }
}


#endif // GENERS_COMPRESSEDIO_HH_

