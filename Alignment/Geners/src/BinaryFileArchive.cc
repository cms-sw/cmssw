#include <sstream>

#include "Alignment/Geners/interface/BinaryFileArchive.hh"

#include "Alignment/Geners/interface/uriUtils.hh"
#include "Alignment/Geners/interface/ContiguousCatalog.hh"
#include "Alignment/Geners/interface/WriteOnlyCatalog.hh"
#include "Alignment/Geners/interface/IOException.hh"
#include "Alignment/Geners/interface/streamposIO.hh"

namespace gs {
    BinaryFileArchive::BinaryFileArchive(const char* basename,
                                         const char* mode, const char* ann,
                                         const unsigned dataFileBufferSize,
                                         const unsigned catalogFileBufferSize)
        : BinaryArchiveBase(basename, mode),
          filebuf_(0),
          catabuf_(0),
          annotation_(ann ? std::string(ann) : std::string("")),
          dataFileName_(AbsArchive::name() + ".0.gsbd"),   // binary data
          catalogFileName_(AbsArchive::name() + ".gsbmf"), // binary metafile
          dataFileURI_(localFileURI(dataFileName_.c_str())),
          lastpos_(0),
          jumppos_(0),
          catalogMergeLevel_(1),
          annotationsMerged_(false),
          streamFlushed_(true)
    {
        if (!modeValid()) return;

        try
        {
            // Get a new buffer for the data and open the data stream
            if (dataFileBufferSize)
                filebuf_ = new char[dataFileBufferSize];
            dataStream_.rdbuf()->pubsetbuf(filebuf_, dataFileBufferSize);
            openDataFile(dataStream_, dataFileName_.c_str());
            dataStream_.seekp(0, std::ios_base::end);

            // Get a new buffer for the catalog and open the catalog stream.
            // We may have to rewrite the complete catalog, so remove the flag
            // std::ios_base::app from the opening mode.
            if (catalogFileBufferSize)
                catabuf_ = new char[catalogFileBufferSize];
            catStream_.rdbuf()->pubsetbuf(catabuf_, catalogFileBufferSize);
            catStream_.open(catalogFileName_.c_str(),
                            openmode() & ~std::ios_base::app);
            if (!catStream_.is_open())
                throw IOOpeningFailure("gs::BinaryFileArchive constructor",
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
                    // Have to read in the catalog contents
                    catStream_.close();
                    catStream_.clear();
                    catStream_.open(catalogFileName_.c_str(),
                                    openmode() | std::ios_base::in);
                    if (!catStream_.is_open()) throw IOOpeningFailure(
                        "gs::BinaryFileArchive constructor", catalogFileName_);
                    readCatalog<WriteOnlyCatalog>();
                    catStream_.seekp(0, std::ios_base::end);
                }
            }
        }
        catch (std::exception& e)
        {
            setCatalog(0);
            releaseBuffers();
            errorStream() << e.what();
        }
    }

    void BinaryFileArchive::releaseBuffers()
    {
        if (dataStream_.is_open()) dataStream_.close();
        if (catStream_.is_open()) catStream_.close();
        catStream_.rdbuf()->pubsetbuf(0, 0);
        dataStream_.rdbuf()->pubsetbuf(0, 0);
        delete [] catabuf_; catabuf_ = 0;
        delete [] filebuf_; filebuf_ = 0;
    }

    BinaryFileArchive::~BinaryFileArchive()
    {
        flush();
        releaseBuffers();
    }

    void BinaryFileArchive::writeCatalog()
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
                os << "In BinaryFileArchive::writeCatalog: "
                   << "failed to write catalog data to file "
                   << catalogFileName_;
                throw IOWriteFailure(os.str());
            }
        }
    }

    std::istream& BinaryFileArchive::plainInputStream(
        const unsigned long long id,
        unsigned* compressionCode,
        unsigned long long* length)
    {
        if (isOpen())
        {
            assert(openmode() & std::ios_base::in);
            if (!id) throw gs::IOInvalidArgument(
                "In gs::BinaryFileArchive::plainInputStream: invalid item id");
            std::streampos pos;
            if (!catalog()->retrieveStreampos(
                    id, compressionCode, length, &pos))
            {
                std::ostringstream os;
                os << "In gs::BinaryFileArchive::plainInputStream: "
                   << "failed to locate item with id " << id
                   << "in the catalog stored in file " << catalogFileName_;
                throw gs::IOInvalidArgument(os.str());
            }
            if (!streamFlushed_)
            {
                dataStream_.flush();
                streamFlushed_ = true;
            }
            dataStream_.seekg(pos);
        }
        return dataStream_;
    }

    unsigned long long BinaryFileArchive::addToCatalog(
        const AbsRecord& record, const unsigned compressionCode,
        const unsigned long long itemLength)
    {
        unsigned long long id = 0;
        if (isOpen())
        {
            id = catalog()->makeEntry(
                record, compressionCode, itemLength,
                ItemLocation(lastpos_, dataFileURI_.c_str()));
            if (id && injectMetadata())
            {
                const CatalogEntry* entry = catalog()->lastEntryMade();
                assert(entry);
                dataStream_.seekp(0, std::ios_base::end);
                std::streampos now = dataStream_.tellp();
                if (entry->write(dataStream_))
                {
                    dataStream_.seekp(jumppos_);
                    write_pod(dataStream_, now);
                    dataStream_.seekp(0, std::ios_base::end);
                }
                else
                    id = 0;
            }
        }
        return id;
    }

    std::ostream& BinaryFileArchive::plainOutputStream()
    {
        if (isOpen())
        {
            assert(openmode() & std::ios_base::out);
            dataStream_.seekp(0, std::ios_base::end);
            if (injectMetadata())
            {
                jumppos_ = dataStream_.tellp();
                std::streampos catpos(0);
                write_pod(dataStream_, catpos);
            }
            lastpos_ = dataStream_.tellp();
            streamFlushed_ = false;
        }
        return dataStream_;
    }

    void BinaryFileArchive::flush()
    {
        if (isOpen())
        {
            if (!streamFlushed_)
            {
                dataStream_.flush();
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
}
