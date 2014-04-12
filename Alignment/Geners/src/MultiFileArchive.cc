#include <sstream>
#include <cstdio>

#include "Alignment/Geners/interface/MultiFileArchive.hh"

#include "Alignment/Geners/interface/uriUtils.hh"
#include "Alignment/Geners/interface/ContiguousCatalog.hh"
#include "Alignment/Geners/interface/WriteOnlyCatalog.hh"
#include "Alignment/Geners/interface/IOException.hh"
#include "Alignment/Geners/interface/streamposIO.hh"

namespace gs {
    MultiFileArchive::MultiFileArchive(const char* basename, const char* mode,
                                       const char* ann,
                                       const unsigned typicalFileSizeInMB,
                                       const unsigned dataFileBufferSize,
                                       const unsigned catalogFileBufferSize)
        : BinaryArchiveBase(basename, mode),
          filebuf_(0),
          readbuf_(0),
          catabuf_(0),
          annotation_(ann ? std::string(ann) : std::string("")),
          catalogFileName_(AbsArchive::name() + ".gsbmf"), // binary metafile
          writeFileURI_("/ / / / / / /\\ \\ \\ \\"),
          readFileURI_(writeFileURI_),
          lastpos_(0),
          jumppos_(0),
          maxpos_(std::streamoff(1048576LL*typicalFileSizeInMB)),
          writeFileNumber_(0),
          catalogMergeLevel_(1),
          annotationsMerged_(false),
          streamFlushed_(true)
    {
        if (!modeValid()) return;

        try
        {
            // Get a new buffer for the output stream
            if (dataFileBufferSize)
                filebuf_ = new char[dataFileBufferSize];
            writeStream_.rdbuf()->pubsetbuf(filebuf_, dataFileBufferSize);

            // Get a new buffer for the input stream
            if (dataFileBufferSize)
                readbuf_ = new char[dataFileBufferSize];
            separateReadStream_.rdbuf()->pubsetbuf(readbuf_,dataFileBufferSize);

            // Get a new buffer for the catalog and open the catalog stream.
            // We may have to rewrite the complete catalog, so remove the flag
            // std::ios_base::app from the opening mode.
            if (catalogFileBufferSize)
                catabuf_ = new char[catalogFileBufferSize];
            catStream_.rdbuf()->pubsetbuf(catabuf_, catalogFileBufferSize);
            catStream_.open(catalogFileName_.c_str(),
                            openmode() & ~std::ios_base::app);
            if (!catStream_.is_open())
                throw IOOpeningFailure("gs::MultiFileArchive constructor",
                                       catalogFileName_);

            // Can we use a write-only catalog?
            if (openmode() & std::ios_base::in)
            {
                // Reading is allowed. Have to use in-memory catalog.
                // If the file data already exists, get the catalog in.
                if (isEmptyFile(catStream_))
                    setCatalog(new ContiguousCatalog());
                else
                    readCatalog<ContiguousCatalog>();
            }
            else
            {
                // Yes, we can use a write-only catalog.
                // Is the catalog file empty? If so, write out
                // the stuff needed at the beginning of the file.
                // If not, assume that the necessary stuff is
                // already there. Note that in this case we will
                // not be able to add the annotation.
                if (isEmptyFile(catStream_))
                {
                    setCatalog(new WriteOnlyCatalog(catStream_));
                    writeCatalog();
                }
                else
                {
                    catStream_.close();
                    catStream_.clear();
                    catStream_.open(catalogFileName_.c_str(),
                                    openmode() | std::ios_base::in);
                    if (!catStream_.is_open()) throw IOOpeningFailure(
                        "gs::MultiFileArchive constructor", catalogFileName_);
                    readCatalog<WriteOnlyCatalog>();
                    catStream_.seekp(0, std::ios_base::end);
                }
            }

            // Open the write stream
            if (openmode() & std::ios_base::out)
            {
                setupWriteStream();
                const std::streampos pos1 = writeStream_.tellp();
                if (maxpos_ < pos1)
                    maxpos_ = pos1;
            }
        }
        catch (std::exception& e)
        {
            setCatalog(0);
            releaseBuffers();
            errorStream() << e.what();
        }
    }

    void MultiFileArchive::releaseBuffers()
    {
        if (writeStream_.is_open()) writeStream_.close();
        if (separateReadStream_.is_open()) separateReadStream_.close();
        if (catStream_.is_open()) catStream_.close();
        catStream_.rdbuf()->pubsetbuf(0, 0);
        writeStream_.rdbuf()->pubsetbuf(0, 0);
        separateReadStream_.rdbuf()->pubsetbuf(0, 0);
        delete [] catabuf_; catabuf_ = 0;
        delete [] readbuf_; readbuf_ = 0;
        delete [] filebuf_; filebuf_ = 0;
    }

    MultiFileArchive::~MultiFileArchive()
    {
        flush();
        releaseBuffers();
    }

    void MultiFileArchive::writeCatalog()
    {
        if (isOpen())
        {
            if (!annotationsMerged_)
            {
                if (annotation_.size())
                    catalogAnnotations_.push_back(annotation_);
                annotationsMerged_ = true;
            }
            const unsigned compress = static_cast<unsigned>(compressionMode());
            if (!writeBinaryCatalog(catStream_, compress, catalogMergeLevel_,
                                    catalogAnnotations_, *catalog()))
            {
                std::ostringstream os;
                os << "In MultiFileArchive::writeCatalog: "
                   << "failed to write catalog data to file "
                   << catalogFileName_;
                throw IOWriteFailure(os.str());
            }
        }
    }

    void MultiFileArchive::openWriteStream()
    {
        assert(openmode() & std::ios_base::out);
        assert(!writeStream_.is_open());
        {
            std::ostringstream os;
            os << AbsArchive::name() << '.' << writeFileNumber_ << ".gsbd";
            writeFileName_  = os.str();
        }
        writeFileURI_ = localFileURI(writeFileName_.c_str());
        openDataFile(writeStream_, writeFileName_.c_str());
    }

    std::ostream& MultiFileArchive::plainOutputStream()
    {
        if (isOpen())
        {
            assert(openmode() & std::ios_base::out);
            if (writeStream_.is_open())
            {
                writeStream_.seekp(0, std::ios_base::end);
                lastpos_ = writeStream_.tellp();
                if (lastpos_ > maxpos_)
                {
                    writeStream_.close();
                    // Don't have to clear. "openDataFile" will do it.
                    // writeStream_.clear();
                    ++writeFileNumber_;
                }
                else if (injectMetadata())
                {
                    jumppos_ = lastpos_;
                    std::streampos catpos(0);
                    write_pod(writeStream_, catpos);
                    lastpos_ = writeStream_.tellp();
                }
            }
            if (!writeStream_.is_open())
            {
                openWriteStream();
                writeStream_.seekp(0, std::ios_base::end);
                if (injectMetadata())
                {
                    jumppos_ = writeStream_.tellp();
                    std::streampos catpos(0);
                    write_pod(writeStream_, catpos);
                }
                lastpos_ = writeStream_.tellp();
            }
            streamFlushed_ = false;
        }
        return writeStream_;
    }

    void MultiFileArchive::flush()
    {
        if (isOpen())
        {
            if (!streamFlushed_)
            {
                writeStream_.flush();
                streamFlushed_ = true;
            }

            if (openmode() & std::ios_base::out)
            {
                if (dynamic_cast<WriteOnlyCatalog*>(catalog()) == 0)
                    writeCatalog();
                catStream_.flush();
            }
        }
    }

    void MultiFileArchive::setupWriteStream()
    {
        if (openmode() & std::ios_base::trunc)
        {
            bool removed = true;
            for (unsigned i=0; removed; ++i)
            {
                std::ostringstream os;
                os << AbsArchive::name() << '.' << i << ".gsbd";
                std::string fname = os.str();
                removed = std::remove(fname.c_str()) == 0;
            }
            writeFileNumber_ = 0;
        }
        else
        {
            unsigned long firstNonExistent = 0;
            for (; ; ++firstNonExistent)
            {
                std::ostringstream os;
                os << AbsArchive::name() << '.' << firstNonExistent << ".gsbd";
                std::string fname = os.str();
                std::ifstream f(fname.c_str());
                if (!f)
                    break;
            }
            writeFileNumber_ = firstNonExistent ? firstNonExistent - 1UL : 0UL;
        }
        openWriteStream();
    }

    std::istream& MultiFileArchive::plainInputStream(
        const unsigned long long id,
        unsigned* compressionCode,
        unsigned long long* length)
    {
        std::fstream* readStream = &writeStream_;
        if (isOpen())
        {
            assert(openmode() & std::ios_base::in);
            if (!id) throw gs::IOInvalidArgument(
                "In gs::MultiFileArchive::plainInputStream: invalid item id");

            // If we have a write stream, and if the archive
            // has one file only, we should be able to retrieve
            // stream position quickly
            std::streampos pos(0);
            if ((openmode() & std::ios_base::out) && writeFileNumber_ == 0UL)
            {
                if (!catalog()->retrieveStreampos(
                        id, compressionCode, length, &pos))
                {
                    std::ostringstream os;
                    os << "In gs::MultiFileArchive::plainInputStream: "
                       << "failed to locate item with id " << id
                       << "in the catalog stored in file " << catalogFileName_;
                    throw gs::IOInvalidArgument(os.str());
                }
            }
            else
            {
                // Here, we have to do a full catalog search
                CPP11_shared_ptr<const CatalogEntry> sptr = 
                    catalog()->retrieveEntry(id);
                const CatalogEntry* pe = sptr.get();
                if (!pe)
                {
                    std::ostringstream os;
                    os << "In gs::MultiFileArchive::plainInputStream: "
                       << "failed to locate item with id " << id
                       << "in the catalog stored in file " << catalogFileName_;
                    throw gs::IOInvalidArgument(os.str());
                }
                pos = pe->location().streamPosition();
                if (pe->location().URI() != writeFileURI_)
                {
                    updateReadStream(pe->location().URI());
                    readStream = &separateReadStream_;
                }
                *compressionCode = pe->compressionCode();
                *length = pe->itemLength();
            }

            // Flush the write stream if it will be used for reading
            if (readStream == &writeStream_)
            {
                assert(writeStream_.is_open());
                if (!streamFlushed_)
                {
                    writeStream_.flush();
                    streamFlushed_ = true;
                }
            }

            readStream->seekg(pos);
        }
        return *readStream;
    }

    void MultiFileArchive::updateReadStream(const std::string& uri)
    {
        if (uri == readFileURI_)
            return;

        assert(openmode() & std::ios_base::in);
        if (separateReadStream_.is_open())
        {
            separateReadStream_.close();
            separateReadStream_.clear();
        }

        // We need to get the name of the local file from the URI.
        // We will assume that it belongs to the archive we are
        // working with right now.
        readFileName_ = joinDir1WithName2(AbsArchive::name().c_str(),
                                          uri.c_str());
        separateReadStream_.open(readFileName_.c_str(), std::ios_base::binary |
                                                        std::ios_base::in);
        if (!separateReadStream_.is_open())
            throw IOOpeningFailure("gs::MultiFileArchive::updateReadStream",
                                   readFileName_);
        readFileURI_ = uri;
    }

    unsigned long long MultiFileArchive::addToCatalog(
        const AbsRecord& record, const unsigned compressionCode,
        const unsigned long long itemLength)
    {
        unsigned long long id = 0;
        if (isOpen())
        {
            id = catalog()->makeEntry(
                record, compressionCode, itemLength,
                ItemLocation(lastpos_, writeFileURI_.c_str()));
            if (id && injectMetadata())
            {
                const CatalogEntry* entry = catalog()->lastEntryMade();
                assert(entry);
                writeStream_.seekp(0, std::ios_base::end);
                std::streampos now = writeStream_.tellp();
                if (entry->write(writeStream_))
                {
                    writeStream_.seekp(jumppos_);
                    write_pod(writeStream_, now);
                    writeStream_.seekp(0, std::ios_base::end);
                }
                else
                    id = 0;
            }
        }
        return id;
    }
}
