#ifndef GENERS_BINARYFILEARCHIVE_HH_
#define GENERS_BINARYFILEARCHIVE_HH_

#include <cassert>

#include "Alignment/Geners/interface/BinaryArchiveBase.hh"
#include "Alignment/Geners/interface/CatalogIO.hh"

namespace gs {
    class BinaryFileArchive : public BinaryArchiveBase
    {
    public:
        // See the note inside the "BinaryArchiveBase.hh" header
        // for the meaning of the "mode" argument
        BinaryFileArchive(const char* basename, const char* mode,
                          const char* annotation = 0,
                          unsigned dataFileBufferSize = 1048576U,
                          unsigned catalogFileBufferSize = 131072U);
        virtual ~BinaryFileArchive();

        void flush();

    private:
        void writeCatalog();
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
        char* catabuf_;
        std::string annotation_;
        std::string dataFileName_;
        std::string catalogFileName_;
        std::string dataFileURI_;
        std::fstream dataStream_;
        std::fstream catStream_;
        std::streampos lastpos_;
        std::streampos jumppos_;
        std::vector<std::string> catalogAnnotations_;
        unsigned catalogMergeLevel_;
        bool annotationsMerged_;
        bool streamFlushed_;
    };
}

#endif // GENERS_BINARYFILEARCHIVE_HH_

