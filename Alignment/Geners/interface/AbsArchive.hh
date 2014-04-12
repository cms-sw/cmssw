#ifndef GENERS_ABSARCHIVE_HH_
#define GENERS_ABSARCHIVE_HH_

#include <vector>

#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"
#include "Alignment/Geners/interface/AbsRecord.hh"
#include "Alignment/Geners/interface/AbsReference.hh"
#include "Alignment/Geners/interface/CatalogEntry.hh"
#include "Alignment/Geners/interface/SearchSpecifier.hh"

namespace gs {
    class AbsArchive;
}

gs::AbsArchive& operator<<(gs::AbsArchive& ar, const gs::AbsRecord& record);

namespace gs {
    //
    // If you need to retrieve the items, use the interface provided
    // by the "Reference" class. Public interface of this class only
    // allows to examine the item metadata and to copy items as BLOBs.
    // 
    class AbsArchive
    {
    public:
        AbsArchive(const char* name);
        inline virtual ~AbsArchive() {}

        // Archive name
        const std::string& name() const {return name_;}

        // Is it correctly open?
        virtual bool isOpen() const = 0;

        // If an attempt to open the archive failed, call the following
        // method to find out why
        virtual std::string error() const = 0;

        // Is the archive readable?
        virtual bool isReadable() const = 0;

        // Is the archive writable?
        virtual bool isWritable() const = 0;

        // Number of items in the archive. Note that id value
        // of 0 refers to an invalid item.
        virtual unsigned long long size() const = 0;

        // Smallest and largest ids of any item in the archive
        virtual unsigned long long smallestId() const = 0;
        virtual unsigned long long largestId() const = 0;

        // Are the item ids contiguous between the smallest and
        // the largest?
        virtual bool idsAreContiguous() const = 0;

        // Check if the item with given id is actually present
        // in the archive
        virtual bool itemExists(unsigned long long id) const = 0;

        // Search for matching items based on item name and
        // category (no type match required)
        virtual void itemSearch(const SearchSpecifier& namePattern,
                                const SearchSpecifier& categoryPattern,
                                std::vector<unsigned long long>* found) const=0;

        // Fetch metadata for the item with given id. NULL pointer is
        // returned if there is no item in the archive with the given id
        // (and there is an automatic cast from CPP11_shared_ptr to bool).
        virtual CPP11_shared_ptr<const CatalogEntry> 
        catalogEntry(unsigned long long id) = 0;

        // Dump everything to storage (if the archive is open for writing
        // and if this makes sense for the archive)
        virtual void flush() = 0;

        // Copy an item from this archive to a destination archive.
        // This archive and destination archive must be distinct.
        // Note that, while it is always possible to copy an item
        // with correct id, a standalone operation like this is
        // not necessarily going to make sense because the item
        // could be just a part of some distributed object.
        //
        // If "newName" and/or "newCategory" arguments are left at their
        // default value of 0, the existing name and/or category are used.
        //
        // This method returns the id of the copy in the destination
        // archive.
        unsigned long long copyItem(unsigned long long id,
                                    AbsArchive* destination,
                                    const char* newName = 0,
                                    const char* newCategory = 0);

        // The id and the length of the last item written
        // (the results make sense only for the archives
        //  that have been open for writing)
        inline unsigned long long lastItemId() const
            {return lastItemId_;}
        inline unsigned long long lastItemLength() const
            {return lastItemLength_;}

        inline bool operator==(const AbsArchive& r) const
            {return (typeid(*this) == typeid(r)) && this->isEqual(r);}
        inline bool operator!=(const AbsArchive& r) const
            {return !(*this == r);}

    protected:
        void addItemToReference(AbsReference& r, unsigned long long id) const;

        // Archives which want to implement reasonable comparisons
        // should override the function below
        virtual bool isEqual(const AbsArchive&) const {return false;}

    private:
        friend class AbsReference;
        friend gs::AbsArchive& ::operator<<(gs::AbsArchive& ar,
                                            const gs::AbsRecord& record);

        // Search for items which correspond to the given reference.
        // The "reference" works both as the input and as the output in
        // this query. The concrete implementations should utilize the
        // "addItemToReference" method in order to produce results.
        virtual void search(AbsReference& reference) = 0;

        // Position the input stream for reading the item with given id.
        // The reading must follow immediately, any other operaction on
        // the archive can invalidate the result.
        //
        // The "sz" argument could be NULL in which case it should be
        // ignored. If it is not NULL, the *sz value on completion
        // should be set as follows:
        //
        // *sz = -1  -- if the uncompressed object size in the stream
        //              is the same as in the catalog
        // *sz >= 0  -- *sz is the object size after decompression
        //              (usually not the same as in the catalog)
        //
        virtual std::istream& inputStream(unsigned long long id,
                                          long long* sz) = 0;

        // Get the stream for writing the next object
        virtual std::ostream& outputStream() = 0;

        // The following function should return 0 on failure and
        // the item id in the archive on success. The id must be positive.
        virtual unsigned long long addToCatalog(
            const AbsRecord& record, unsigned compressCode,
            unsigned long long itemLength) = 0;

        // The archives which want to support compression should
        // override the following two methods
        virtual std::ostream& compressedStream(std::ostream& uncompressed)
            {return uncompressed;}

        // The following function returns some kind of a code which
        // tells us how the item was compressed. Default value of 0
        // means no compression.
        virtual unsigned flushCompressedRecord(std::ostream& /* compressed */)
            {return 0U;}

        std::string name_;
        unsigned long long lastItemId_;
        unsigned long long lastItemLength_;
    };
}

#endif // GENERS_ABSARCHIVE_HH_

