#ifndef GENERS_ABSREFERENCE_HH_
#define GENERS_ABSREFERENCE_HH_

#include <vector>
#include <iostream>

#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"
#include "Alignment/Geners/interface/ClassId.hh"
#include "Alignment/Geners/interface/SearchSpecifier.hh"

namespace gs {
    class AbsArchive;
    class CatalogEntry;

    class AbsReference
    {
    public:
        inline virtual ~AbsReference() {}

        inline AbsArchive& archive() const {return archive_;}
        inline const ClassId& type() const {return classId_;}
        inline const std::string& ioPrototype() const {return ioProto_;}
        inline const SearchSpecifier& namePattern() const
            {return namePattern_;}
        inline const SearchSpecifier& categoryPattern() const
            {return categoryPattern_;}

        // Determine if the item in the catalog is compatible for I/O 
        // purposes with the one referenced here 
        virtual bool isIOCompatible(const CatalogEntry& r) const;

        // Check I/O prototype only allowing for class id mismatch
        bool isSameIOPrototype(const CatalogEntry& r) const;

        // Are there any items referenced?
        bool empty() const;

        // Exactly one item referenced?
        bool unique() const;

        // How many items are referenced?
        unsigned long size() const;

        // The following function throws gs::IOOutOfRange exception
        // if the index is out of range
        unsigned long long id(unsigned long index) const;

        // Catalog entry retrieval by index in the list of referenced items.
        // Throws gs::IOOutOfRange exception if the index is out of range.
        CPP11_shared_ptr<const CatalogEntry> 
        indexedCatalogEntry(unsigned long index) const;

    protected:
        // Use the following constructor to retrieve an item with
        // a known id
        AbsReference(AbsArchive& ar, const ClassId& classId,
                     const char* ioProto,
                     unsigned long long itemId);

        // Use the following constructor to search for items which
        // match name and category patterns
        AbsReference(AbsArchive& ar, const ClassId& classId,
                     const char* ioProto,
                     const SearchSpecifier& namePattern,
                     const SearchSpecifier& categoryPattern);

        std::istream& positionInputStream(unsigned long long id) const;

    private:
        friend class AbsArchive;

        AbsReference();

        void initialize() const;
        void addItemId(unsigned long long id);

        AbsArchive& archive_;
        ClassId classId_;
        std::string ioProto_;

        // The following items will be filled or not,
        // depending on which constructor was called
        unsigned long long searchId_;
        SearchSpecifier namePattern_;
        SearchSpecifier categoryPattern_;

        // Id for a unique verified item
        unsigned long long itemId_;

        // The item list in case ids are not unique
        std::vector<unsigned long long> idList_;

        // We can't talk to the archive from the constructor
        // because we need correct "isIOCompatible" which
        // can be overriden by derived classes. Therefore,
        // we will delay archive searching until the first
        // function call.
        bool initialized_;
    };
}

#endif // GENERS_ABSREFERENCE_HH_

