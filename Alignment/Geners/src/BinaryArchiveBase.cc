#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cassert>

#include "Alignment/Geners/interface/BinaryArchiveBase.hh"
#include "Alignment/Geners/interface/CatalogIO.hh"
#include "Alignment/Geners/interface/binaryIO.hh"

#ifdef GENERS_BINARY_ARCHIVE_FORMAT_ID
#undef GENERS_BINARY_ARCHIVE_FORMAT_ID
#endif
#define GENERS_BINARY_ARCHIVE_FORMAT_ID (0x1f2e3d4c)

static bool parse_unsigned(std::ostringstream& err,
                           const char *c, unsigned *result)
{
    char *endptr;
    errno = 0;
    const unsigned long value = strtoul(c, &endptr, 0);
    if (errno || *endptr != '\0')
    {
        err << "expected an unsigned integer, got \"" << c << '"';
        if (errno) err << ", " << strerror(errno);
        return false;
    }
    if (value > UINT_MAX)
    {
        err << "unsigned value \"" << c << "\" is out of range";
        return false;
    }
    *result = value;
    return true;
}

static bool parse_int(std::ostringstream& err,
                      const char *c, int *result)
{
    char *endptr;
    errno = 0;
    const long value = strtol(c, &endptr, 0);
    if (errno || *endptr != '\0')
    {
        err << "expected an integer, got \"" << c << '"';
        if (errno) err << ", " << strerror(errno);
        return false;
    }
    if (value < INT_MIN || value > INT_MAX)
    {
        err << "integer value \"" << c << "\" is out of range";
        return false;
    }
    *result = value;
    return true;
}

namespace gs {
    BinaryArchiveBase::BinaryArchiveBase(const char* name, const char* mode)
        : AbsArchive(name),
          mode_(parseMode(mode)),
          errorStream_(0),
          cStream_(0),
          catalog_(0),
          storedEntryId_(0),
          storedLocationId_(0),
          catalogIsSet_(false),
          addCatalogToData_(false)
    {
        CStringStream::CompressionMode m = CStringStream::NOT_COMPRESSED;
        int compressionLevel = -1;
        unsigned minSizeToCompress = 1024U;
        unsigned bufSize = 1048576U; // 1024*1024

        std::ostringstream err;
        modeIsValid_ = parseArchiveOptions(err, mode, &m, &compressionLevel,
                                           &minSizeToCompress, &bufSize,
                                           &addCatalogToData_);
        if (modeIsValid_)
            cStream_ = new CStringStream(m, compressionLevel,
                                         minSizeToCompress, bufSize);
        else
        {
            errorStream() << "In BinaryArchiveBase constructor: "
                          << "invalid archive opening mode \"" << mode << '"';
            const std::string& errInfo = err.str();
            if (!errInfo.empty())
                errorStream() << ": " << errInfo;
        }
    }


    void BinaryArchiveBase::releaseClassIds()
    {
        delete storedEntryId_; storedEntryId_ = 0;
        delete storedLocationId_; storedLocationId_ = 0;
    }


    BinaryArchiveBase::~BinaryArchiveBase()
    {
        releaseClassIds();
        delete errorStream_;
        delete catalog_;
        delete cStream_;
    }


    void BinaryArchiveBase::writeHeader(std::ostream& os)
    {
        const unsigned format = GENERS_BINARY_ARCHIVE_FORMAT_ID;
        write_pod(os, format);

        // Write some other info
        const unsigned multiplex = addCatalogToData_ ? 1 : 0;
        const unsigned sizeoflong = sizeof(long);
        const unsigned infoword = (sizeoflong << 1) | multiplex;
        write_pod(os, infoword);

        if (multiplex)
        {
            // Write class ids for CatalogEntry and ItemLocation
            releaseClassIds();
            storedEntryId_ = new ClassId(ClassId::makeId<CatalogEntry>());
            storedEntryId_->write(os);
            storedLocationId_ = new ClassId(ClassId::makeId<ItemLocation>());
            storedLocationId_->write(os);
        }
    }


    bool BinaryArchiveBase::readHeader(std::istream& is)
    {
        const unsigned expectedFormat = GENERS_BINARY_ARCHIVE_FORMAT_ID;
        is.seekg(0, std::ios_base::beg);
        unsigned format = 0;
        read_pod(is, &format);
        if (format != expectedFormat)
            return false;

        unsigned infoword = 0xffffffff;
        read_pod(is, &infoword);
        const unsigned multiplex = infoword & 0x1U;
        const unsigned sizeoflong = infoword >> 1;

        // The following check will make sure that we are not reading
        // an archive created on a 32-bit machine with a 64-bit system
        // (and otherwise)
        if (sizeoflong != sizeof(long))
            return false;

        addCatalogToData_ = multiplex;
        if (addCatalogToData_)
        {
            releaseClassIds();
            storedEntryId_ = new ClassId(is, 1);
            storedLocationId_ = new ClassId(is, 1);

            // Can't open this archive for update if the above class ids
            // are obsolete -- otherwise we will loose the capability to
            // restore the catalog
            if (mode_ & std::ios_base::out)
            {
                const ClassId& entryId = ClassId::makeId<CatalogEntry>();
                const ClassId& locId = ClassId::makeId<ItemLocation>();
                if (entryId != *storedEntryId_ || locId != *storedLocationId_)
                    throw IOInvalidData(
                        "In gs::BinaryArchiveBase::readHeader: this "
                        "archive can no longer be open for update as it was "
                        "created using an older version of I/O software");
            }
        }
        return !is.fail();
    }


    void BinaryArchiveBase::openDataFile(std::fstream& stream,
                                         const char* filename)
    {
        assert(filename);
        if (stream.is_open())
            stream.close();
        stream.clear();
        stream.open(filename, mode_);
        if (!stream.is_open())
            throw IOOpeningFailure("gs::BinaryArchiveBase::openDataFile",
                                   filename);

        // Do we need to write the header out or to read it in?
        bool writeHead = false;
        if (mode_ & std::ios_base::out)
        {
            if (mode_ & std::ios_base::trunc)
                writeHead = true;
            else if (isEmptyFile(stream))
                writeHead = true;
        }

        if (writeHead)
        {
            writeHeader(stream);
            if (stream.fail())
            {
                stream.close();
                std::string e = "In gs::BinaryArchiveBase::openDataFile: "
                    "failed to write archive header to file \"";
                e += filename;
                e += "\"";
                throw IOWriteFailure(e);
            }
        }
        else
        {
            if (!readHeader(stream))
            {
                const bool failed = stream.fail();
                stream.close();                
                std::string e = "In gs::BinaryArchiveBase::openDataFile: ";
                if (failed)
                {
                    e += "could not read archive header from file \"";
                    e += filename;
                    e += "\"";
                    throw IOReadFailure(e);
                }
                else
                {
                    e += "no valid archive header in file \"";
                    e += filename;
                    e += "\"";
                    throw IOInvalidData(e);
                }
            }
        }
    }


    void BinaryArchiveBase::setCatalog(AbsCatalog* c)
    {
        if (c)
        {
            assert(!catalogIsSet_);
            catalogIsSet_ = true;
        }
        delete catalog_;
        catalog_ = c;
    }


    void BinaryArchiveBase::itemSearch(
        const SearchSpecifier& namePattern,
        const SearchSpecifier& categoryPattern,
        std::vector<unsigned long long>* idsFound) const
    {
        if (catalog_)
            catalog_->search(namePattern, categoryPattern, idsFound);
        else
        {
            assert(idsFound);
            idsFound->clear();
        }
    }


    bool BinaryArchiveBase::parseArchiveOptions(
        std::ostringstream& err,
        const char* modeIn, CStringStream::CompressionMode* m,
        int* compressionLevel, unsigned* minSizeToCompress,
        unsigned* bufSize, bool* multiplexCatalog)
    {
        if (!modeIn)
            return true;
        std::string cmode(modeIn ? modeIn : "");
        if (cmode.empty())
            return true;
        char* mode = const_cast<char*>(cmode.c_str());

        unsigned cnt = 0;
        for (char* opt = strtok(mode, ":"); opt; opt = strtok(0, ":"), ++cnt)
        {
            // Skip the first word -- this is the file opening mode
            if (!cnt)
                continue;
            char* eq = strchr(opt, '=');
            if (eq)
            {
                // Get rid of spaces around option name
                char* optname = opt;
                while (isspace(*optname) && optname < eq)
                    ++optname;
                if (optname == eq)
                {
                    err << "invalid binary archive option \"\"";
                    return false;
                }
                char* optend = eq - 1;
                while (isspace(*optend))
                    --optend;
                ++optend;
                *optend = '\0';

                // Get rid of spaces around option value
                char* optval = eq + 1;
                while (*optval && isspace(*optval))
                    ++optval;
                if (!*optval)
                {
                    err << "invalid binary archive option value \"\"";
                    return false;
                }
                char* valend = opt + strlen(opt) - 1;
                while (isspace(*valend))
                    --valend;
                ++valend;
                *valend = '\0';
                if (strlen(optval) == 0)
                {
                    err << "invalid binary archive option value \"\"";
                    return false;
                }

                // Go over possible options
                if (!strcasecmp(optname, "z"))
                {
                    // Compression type
                    if (!CStringStream::getCompressionModeByName(optval, m))
                    {
                        err << "invalid compression type \"" << optval << '"';
                        return false;
                    }
                }
                else if (!strcasecmp(optname, "cl"))
                {
                    // Compression level
                    if (!parse_int(err, optval, compressionLevel))
                        return false;
                    if (*compressionLevel < -1 || *compressionLevel > 9)
                    {
                        err << "compression level is out of range";
                        return false;
                    }
                }
                else if (!strcasecmp(optname, "cb"))
                {
                    // Compression buffer size
                    if (!parse_unsigned(err, optval, bufSize))
                        return false;
                }
                else if (!strcasecmp(optname, "cm"))
                {
                    // Compression minimum size
                    if (!parse_unsigned(err, optval, minSizeToCompress))
                        return false;
                }
                else if (!strcasecmp(optname, "cat"))
                {
                    // Internal or external catalog
                    if (optval[0] == 'i' || optval[0] == 'I')
                        *multiplexCatalog = true;
                    else if (optval[0] == 's' || optval[0] == 'S')
                        *multiplexCatalog = false;
                    else
                    {
                        err << "invalid catalog mode \"" << optval << '"';
                        return false;
                    }
                }
                else
                {
                    // Unknown option
                    err << "unrecognized binary archive option \""
                        << optname << '"';
                    return false;
                }
            }
            else
            {
                err << "invalid binary archive option \"" << opt << '"';
                return false;
            }
        }
        return true;
    }


    std::ios_base::openmode BinaryArchiveBase::parseMode(const char* mode)
    {
        std::ios_base::openmode m = std::ios_base::binary;
        if (mode)
        {
            const unsigned len = strlen(mode);
            for (unsigned i=0; i<len; ++i)
            {
                // Note that all characters other than 'r', 'w',
                // 'a', and '+' are basically ignored inside this cycle
                if (mode[i] == 'r')
                    m |= std::ios_base::in;
                else if (mode[i] == 'w')
                    m |= (std::ios_base::out | std::ios_base::trunc);
                else if (mode[i] == 'a')
                    m |= (std::ios_base::out | std::ios_base::app);
                else if (mode[i] == '+')
                    m |= (std::ios_base::in | std::ios_base::out);
                else if (mode[i] == ':')
                    break;
            }
        }

        // Make sure that we are at least reading
        if (!(m & (std::ios_base::in | std::ios_base::out)))
            m |= std::ios_base::in;
        return m;
    }


    void BinaryArchiveBase::search(AbsReference& reference)
    {
        if (catalog_)
        {
            std::vector<unsigned long long> idlist;
            catalog_->search(reference.namePattern(),
                             reference.categoryPattern(),
                             &idlist);
            const unsigned long nfound = idlist.size();
            for (unsigned long i=0; i<nfound; ++i)
            {
                CPP11_shared_ptr<const CatalogEntry> pentry = 
                    catalog_->retrieveEntry(idlist[i]);
                if (reference.isIOCompatible(*pentry))
                    addItemToReference(reference, idlist[i]);
            }
        }
    }


    bool BinaryArchiveBase::isEmptyFile(std::fstream& s)
    {
        s.seekg(0, std::ios_base::end);
        return s.tellg() == std::streampos(0);
    }


    std::istream& BinaryArchiveBase::inputStream(const unsigned long long id,
                                                 long long *sz)
    {
        unsigned long long length = 0;
        unsigned compressionCode = 0;
        std::istream& is = plainInputStream(id, &compressionCode, &length);
        if (cStream_->compressionMode() == CStringStream::NOT_COMPRESSED)
        {
            if (sz)
                *sz = -1LL;
            return is;
        }
        else
        {
            cStream_->readCompressed(is, compressionCode, length);
            if (sz)
            {
                std::streamoff off = cStream_->tellp();
                *sz = off;
            }
            return *cStream_;
        }
    }


    std::ostream& BinaryArchiveBase::outputStream()
    {
        return plainOutputStream();
    }


    std::ostream& BinaryArchiveBase::compressedStream(std::ostream& os)
    {
        if (cStream_->compressionMode() == CStringStream::NOT_COMPRESSED)
            return os;
        else
        {
            cStream_->reset();
            cStream_->setSink(os);
            return *cStream_;
        }
    }


    unsigned BinaryArchiveBase::flushCompressedRecord(std::ostream&)
    {
        CStringStream::CompressionMode m = cStream_->compressionMode();
        if (m != CStringStream::NOT_COMPRESSED)
        {
            cStream_->flush();
            m = cStream_->writeCompressed();
        }
        return static_cast<unsigned>(m);
    }
}
