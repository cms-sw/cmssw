#include <cassert>
#include <cstring>
#include <fstream>
#include <sstream>
#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/stringArchiveIO.hh"
#include "Alignment/Geners/interface/CompressedIO.hh"
#include "Alignment/Geners/interface/IOException.hh"
#include "Alignment/Geners/interface/Reference.hh"


static bool suffix_matches(const char* filename, const char* suffix)
{
    static const char default_suffix[] = ".gssaz";
    assert(filename);
    if (suffix == 0)
        suffix = default_suffix;
    const std::size_t lenSuffix = strlen(suffix);
    const std::size_t len = strlen(filename);
    bool suffixMatches = len >= lenSuffix;
    for (std::size_t i=0; i<lenSuffix && suffixMatches; ++i)
        suffixMatches = suffix[i] == filename[len-lenSuffix+i];
    return suffixMatches;
}


namespace gs {
    bool writeStringArchive(const StringArchive& ar, const char* filename)
    {
        assert(filename);
        bool status = false;
        {
            std::ofstream of(filename, std::ios_base::binary);
            if (of.is_open())
            {
                const_cast<StringArchive&>(ar).flush();
                status = write_item(of, ar);
            }
        }
        return status;
    }

    StringArchive* readStringArchive(const char* filename)
    {
        assert(filename);
        std::ifstream is(filename, std::ios_base::binary);
        if (!is.is_open())
            throw IOOpeningFailure("gs::readStringArchive", filename);
        CPP11_auto_ptr<StringArchive> ar = read_item<StringArchive>(is);
        return ar.release();
    }

    bool writeCompressedStringArchive(
        const StringArchive& ar, const char* filename,
        const unsigned inCompressionMode, const int compressionLevel,
        const unsigned minSizeToCompress, const unsigned bufSize)
    {
        assert(filename);
        if (inCompressionMode > CStringStream::BZIP2)
            throw gs::IOInvalidArgument(
                "In gs::writeCompressedStringArchive: "
                "compression mode argument out of range");
        const CStringStream::CompressionMode m = 
            static_cast<CStringStream::CompressionMode>(inCompressionMode);
        bool status = false;
        {
            std::ofstream of(filename, std::ios_base::binary);
            if (of.is_open())
            {
                const_cast<StringArchive&>(ar).flush();
                status = write_compressed_item(of, ar, m, compressionLevel,
                                               minSizeToCompress, bufSize);
            }
        }
        return status;        
    }

    StringArchive* readCompressedStringArchive(const char* filename)
    {
        assert(filename);
        std::ifstream is(filename, std::ios_base::binary);
        if (!is.is_open())
            throw IOOpeningFailure("gs::readCompressedStringArchive", filename);
        CPP11_auto_ptr<StringArchive> ar =
            read_compressed_item<StringArchive>(is);
        return ar.release();
    }

    StringArchive* loadStringArchiveFromArchive(AbsArchive& arch,
                                                const unsigned long long id)
    {
        Reference<StringArchive> ref(arch, id);
        if (!ref.unique())
        {
            std::ostringstream os;
            os << "In gs::loadStringArchiveFromArchive: "
               << "StringArchive item with id " << id << " not found";
            throw gs::IOInvalidArgument(os.str());
        }
        CPP11_auto_ptr<StringArchive> p = ref.get(0);
        return p.release();
    }

    StringArchive* readCompressedStringArchiveExt(const char* filename,
                                                  const char* suffix)
    {
        if (suffix_matches(filename, suffix))
            return readCompressedStringArchive(filename);
        else
            return readStringArchive(filename);
    }

    bool writeCompressedStringArchiveExt(const StringArchive& ar,
                                         const char* filename,
                                         const char* suffix)
    {
        if (suffix_matches(filename, suffix))
            return writeCompressedStringArchive(ar, filename);
        else
            return writeStringArchive(ar, filename);
    }
}
