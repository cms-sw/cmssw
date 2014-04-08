#ifndef GENERS_BINARYARCHIVEBASE_HH_
#define GENERS_BINARYARCHIVEBASE_HH_

#include <fstream>
#include <sstream>

#include "Alignment/Geners/interface/AbsCatalog.hh"
#include "Alignment/Geners/interface/AbsArchive.hh"
#include "Alignment/Geners/interface/CStringStream.hh"

namespace gs {
    class BinaryArchiveBase : public AbsArchive
    {
    public:
        // The "mode" argument is a string which can have one or more
        // sections, separated by ":". The first section has the same
        // meaning as in the "fopen" call (see "man 3 fopen"). Additional
        // sections can specify other aspects of the archive behavior
        // using the format "option=value" if the default settings are
        // not suitable. The available options are:
        //
        // "z"   -- compression type. Possible option values are
        //           "n" -- no compression (default)
        //           "z" -- compress with zlib
        //           "b" -- compress with bzlib2
        //
        // "cl"  -- compression level (an integer between -1 and 9,
        //           -1 is default. Meanigful for zlib compression
        //           only. See zlib documentation for details.
        //
        // "cb"  -- compression buffer size in bytes (unsigned integer).
        //           Default is 1 MB.
        //
        // "cm"  -- minimum object size in bytes to compress (unsigned
        //           integer). Objects whose serialized size is below
        //           this limit are not compressed. Default is 1 KB.
        //
        // "cat" -- if the value is set to "i" (which means "internal"
        //           or "injected"), the catalog data will be injected
        //           into the data stream in addition to having it in
        //           a separate catalog file. This allows for catalog
        //           recovery from the data stream in cases of program
        //           failure but also increases the data file size.
        //           The default value of this option is "s" which means
        //           that the catalog data will be stored separately.
        //           This option is meaningful for new archives only,
        //           for existing archives the value of this option is
        //           taken from the archive header record.
        //
        // Example: "w+:z=z:cl=9:cm=2048:cat=s". This will compress
        // objects with 2 KB or larger size using level 9 zlib compression
        // (the best compression ratio possible in zlib which is also
        // the slowest). The archive will be open for reading and writing.
        // If an archive with the same name already exists, it will be
        // overwritten. The catalog will be stored in a separate file
        // created when the archive is closed, catalog recovery from the
        // data file(s) in case of a catastrophic program failure will not
        // be possible.
        //
        BinaryArchiveBase(const char* name, const char* mode);

        virtual ~BinaryArchiveBase();

        inline bool isOpen() const {return modeIsValid_ && catalog_;}

        inline bool isReadable() const
            {return modeIsValid_ && catalog_ && (mode_ & std::ios_base::in);}

        inline bool isWritable() const
            {return modeIsValid_ && catalog_ && (mode_ & std::ios_base::out);}

        // Error message produced in case the archive could not be opened
        inline std::string error() const
            {return errorStream_ ? errorStream_->str() : std::string("");}

        // Check whether the constructor "mode" argument was valid.
        // If it was not, derived classes should behave as if the
        // archive could not be opened.
        inline bool modeValid() const {return modeIsValid_;}

        inline unsigned long long size() const
            {return catalog_ ? catalog_->size() : 0ULL;}

        inline unsigned long long smallestId() const
            {return catalog_ ? catalog_->smallestId() : 0ULL;}

        inline unsigned long long largestId() const
            {return catalog_ ? catalog_->largestId() : 0ULL;}

        inline bool idsAreContiguous() const
            {return catalog_ ? catalog_->isContiguous() : false;}

        inline bool itemExists(const unsigned long long id) const
            {return catalog_ ? catalog_->itemExists(id) : false;}

        void itemSearch(const SearchSpecifier& namePattern,
                        const SearchSpecifier& categoryPattern,
                        std::vector<unsigned long long>* idsFound) const;

        inline CPP11_shared_ptr<const CatalogEntry>
        catalogEntry(const unsigned long long id)
            {return catalog_ ? catalog_->retrieveEntry(id) :
             CPP11_shared_ptr<const CatalogEntry>((const CatalogEntry*)0);}

        // Inspection methods for compression options
        inline CStringStream::CompressionMode compressionMode() const
            {return cStream_->compressionMode();}

        inline std::size_t compressionBufferSize() const
            {return cStream_->bufferSize();}

        inline int compressionLevel() const
            {return cStream_->compressionLevel();}

        inline unsigned minSizeToCompress() const
            {return cStream_->minSizeToCompress();}

        // Inject metadata into the data stream when writing? If this
        // method returns "true", we either had "cat=i" in the opening
        // mode for a new archive or a corresponding flag was set in
        // the header of an existing archive data file.
        inline bool injectMetadata() const {return addCatalogToData_;}

        // The following method moves the "get pointer" of the stream
        // to the end of file as a side effect
        static bool isEmptyFile(std::fstream& s);

        // The following method converts the first section of the "mode"
        // argument into std::ios_base::openmode
        static std::ios_base::openmode parseMode(const char* mode);

    protected:
        inline AbsCatalog* catalog() const {return catalog_;}

        // Non-null catalog must be set exactly once. This object will
        // assume the catalog ownership. Null catalog can be set after
        // non-null in case some essential operation on the catalog has
        // failed (such as writing it to file) in order to indicate
        // failure to open the archive.
        void setCatalog(AbsCatalog* c);

        // Set compression mode (can be used when the catalog is read).
        // The argment must be consistent with one of the modes defined
        // in the CStringStream.hh header.
        inline void setCompressionMode(const unsigned cMode)
            {cStream_->setCompressionMode(
                 static_cast<CStringStream::CompressionMode>(cMode));}

        // Stream for error messages. To be used from constructors
        // of derived classes in case of problems, to indicate the
        // reason why the archive could not be opened.
        inline std::ostringstream& errorStream()
        {
            if (!errorStream_) errorStream_ = new std::ostringstream();
            return *errorStream_;
        }

        // The following method opens a binary archive. It makes sure
        // that a proper header is written out in case an empty file
        // is open or in case the file is truncated, and that the header
        // is there when a non-empty file is open without truncation.
        // If the argument fstream is open when this method is invoked,
        // it is closed first. After invocation of this method, the
        // "injectMetadata()" flag will be properly set up for the
        // data file open last. If the method is not successful, it
        // closes the stream and throws an exception inherited from
        // "IOException".
        //
        void openDataFile(std::fstream& stream, const char* filename);

        // Stream mode used to open the archive data file(s)
        inline std::ios_base::openmode openmode() const {return mode_;}

        // Info needed for catalog recovery. These methods will return
        // null pointers if item metadata is not in the data stream.
        const ClassId* catalogEntryClassId() const {return storedEntryId_;}
        const ClassId* itemLocationClassId() const {return storedLocationId_;}

    private:
        BinaryArchiveBase();
        BinaryArchiveBase(const BinaryArchiveBase&);
        BinaryArchiveBase& operator=(const BinaryArchiveBase&);

        static bool parseArchiveOptions(
            std::ostringstream& errmes,
            const char* mode, CStringStream::CompressionMode* m,
            int* compressionLevel, unsigned* minSizeToCompress,
            unsigned* bufSize, bool* multiplexCatalog);

        void writeHeader(std::ostream& os);

        // The following method returns "true" if a correctly
        // formatted header was found
        bool readHeader(std::istream& is);

        virtual void search(AbsReference& reference);

        // The derived classes must override the following two methods
        virtual std::ostream& plainOutputStream() = 0;
        virtual std::istream& plainInputStream(unsigned long long id,
                                               unsigned* compressionCode,
                                               unsigned long long* length) = 0;

        std::istream& inputStream(unsigned long long id, long long* sz);
        std::ostream& outputStream();
        std::ostream& compressedStream(std::ostream& uncompressed);
        unsigned flushCompressedRecord(std::ostream& compressed);
        void releaseClassIds();

        const std::ios_base::openmode mode_;
        std::ostringstream* errorStream_;
        CStringStream* cStream_;
        AbsCatalog* catalog_;
        ClassId* storedEntryId_;
        ClassId* storedLocationId_;
        bool catalogIsSet_;
        bool modeIsValid_;
        bool addCatalogToData_;
    };
}

#endif // GENERS_BINARYARCHIVEBASE_HH_

