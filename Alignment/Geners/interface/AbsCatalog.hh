#ifndef GENERS_ABSCATALOG_HH_
#define GENERS_ABSCATALOG_HH_

#include <vector>
#include <cassert>

#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"
#include "Alignment/Geners/interface/ItemDescriptor.hh"
#include "Alignment/Geners/interface/CatalogEntry.hh"
#include "Alignment/Geners/interface/SearchSpecifier.hh"

namespace gs {
    //
    // This abstract class defines interfaces for adding entries to
    // a catalog but not for removing them. Of course, derived classes
    // can also implement catalog entry removal if necessary.
    //
    struct AbsCatalog
    {
        virtual ~AbsCatalog() {}

        // The number of entries in the catalog
        virtual unsigned long long size() const = 0;

        // Smallest and largest ids of any item in the catalog
        virtual unsigned long long smallestId() const = 0;
        virtual unsigned long long largestId() const = 0;
        virtual bool isContiguous() const = 0;

        // Check if an item with the given id is actually present in
        // the catalog. Catalogs which support non-contiguous item ids
        // (for example, if items can be removed) MUST override this.
        virtual bool itemExists(const unsigned long long id) const
        {
            if (id == 0ULL) return false;
            assert(isContiguous());
            return id >= smallestId() && id <= largestId();
        }

        // The following function should return the id of the new entry
        virtual unsigned long long makeEntry(const ItemDescriptor& descriptor,
                                             unsigned compressionCode,
                                             unsigned long long itemLength,
                                             const ItemLocation& loc,
                                             unsigned long long offset=0ULL)=0;

        // It must be possible to retrieve the entry made by the last
        // "makeEntry" call. If no "makeEntry" calls were made in this
        // program (e.g., the catalog was only read and not written),
        // null pointer should be returned. The entry should be owned
        // by the catalog itself.
        virtual const CatalogEntry* lastEntryMade() const = 0;

        // The following function returns a shared pointer to the entry.
        // The pointer will contain NULL in case the item is not found.
        virtual CPP11_shared_ptr<const CatalogEntry> retrieveEntry(
            unsigned long long id) const = 0;

        // The following function fetches just the stream position
        // associated with the entry. "true" is returned on success.
        // Useful for catalogs which serve a single stream.
        virtual bool retrieveStreampos(
            unsigned long long id, unsigned* compressionCode,
            unsigned long long* length, std::streampos* pos) const = 0;

        // Search for matching entries based on item name and category
        virtual void search(const SearchSpecifier& namePattern,
                            const SearchSpecifier& categoryPattern,
                            std::vector<unsigned long long>* idsFound) const=0;

        inline bool operator==(const AbsCatalog& r) const
            {return (typeid(*this) == typeid(r)) && this->isEqual(r);}
        inline bool operator!=(const AbsCatalog& r) const
            {return !(*this == r);}

        // Prototypes needed for I/O
        virtual ClassId classId() const = 0;
        virtual bool write(std::ostream&) const = 0;

    protected:
        virtual bool isEqual(const AbsCatalog&) const = 0;
    };
}

#endif // GENERS_ABSCATALOG_HH_

