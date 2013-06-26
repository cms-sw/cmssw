#ifndef GENERS_MULTIFILEARCHIVE_HH_
#define GENERS_MULTIFILEARCHIVE_HH_

#include <cassert>

#include "Alignment/Geners/interface/BinaryArchiveBase.hh"
#include "Alignment/Geners/interface/CatalogIO.hh"

namespace gs {
    class MultiFileArchive : public BinaryArchiveBase
    {
    public:
        // See the note inside the "BinaryArchiveBase.hh" header
        // for the meaning of the "mode" argument
        MultiFileArchive(const char* basename, const char* mode,
                         const char* annotation = 0,
                         unsigned typicalFileSizeInMB = 1000U,
                         unsigned dataFileBufferSize = 1048576U,
                         unsigned catalogFileBufferSize = 131072U);
        virtual ~MultiFileArchive();

        void flush();

    private:
        void writeCatalog();
        void openWriteStream();
        void setupWriteStream();
        void updateReadStream(const std::string& uri);
        void releaseBuffers();

        template<class Catalog>
        void readCatalog()
        {
            assert(!catalog());
            unsigned compressionMode;
            setCatalog(readBinaryCatalog<Catalog>(
                           catStream_, &compressionMode, &catalogMergeLevel_,
                           &catalogAnnotations_, true));
            assert(catalog());
            setCompressionMode(compressionMode);
        }

        // The following methods have to be overriden from the base
        std::ostream& plainOutputStream();
        std::istream& plainInputStream(unsigned long long id,
                                       unsigned* compressionCode,
                                       unsigned long long* length);

        unsigned long long addToCatalog(
            const AbsRecord& record, unsigned compressionCode,
            unsigned long long itemLength);

        char* filebuf_;
        char* readbuf_;
        char* catabuf_;
        std::string annotation_;
        std::string catalogFileName_;
        std::string writeFileName_;
        std::string writeFileURI_;
        std::string readFileName_;
        std::string readFileURI_;
        std::fstream writeStream_;
        std::fstream catStream_;
        std::fstream separateReadStream_;
        std::streampos lastpos_;
        std::streampos jumppos_;
        std::streampos maxpos_;
        std::vector<std::string> catalogAnnotations_;
        unsigned long writeFileNumber_;
        unsigned catalogMergeLevel_;
        bool annotationsMerged_;
        bool streamFlushed_;
    };
}

#endif // GENERS_MULTIFILEARCHIVE_HH_

