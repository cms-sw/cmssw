#include <algorithm>

#include "Alignment/Geners/interface/AbsReference.hh"
#include "Alignment/Geners/interface/CatalogEntry.hh"
#include "Alignment/Geners/interface/AbsArchive.hh"
#include "Alignment/Geners/interface/IOException.hh"

// Notes on implementation
//
// The following invariants should be preserved:
//
// 0 items:    must have itemId_ == 0 && idList_.empty()
// 1 item:     must have itemId_ != 0 && idList_.empty()
// more items: must have itemId_ == 0 && !idList_.empty()
//
namespace gs {
    std::istream& AbsReference::positionInputStream(
        const unsigned long long id) const
    {
        return archive_.inputStream(id, 0);
    }

    void AbsReference::initialize() const
    {
        AbsReference* modifiable = const_cast<AbsReference*>(this);
        if (searchId_)
        {
            // Make sure that the searchId_ item id is going to produce
            // a valid item
            CPP11_shared_ptr<const CatalogEntry> record = 
                archive_.catalogEntry(searchId_);
            const unsigned long long idFound = record->id();

            // Check for valid id in the archive
            if (idFound && this->isIOCompatible(*record))
            {
                if (idFound != searchId_) throw IOInvalidData(
                    "In AbsReference::initialize: catalog is corrupted");
                modifiable->itemId_ = searchId_;
            }
        }
        else
        {
            // Search for matching items in the archive
            archive_.search(*modifiable);
            if (!idList_.empty())
            {
                // Check if the list is sorted.
                // Sort if it is not.
                const unsigned long nFound = idList_.size();
                const unsigned long long* ids = &idList_[0];
                bool sorted = true;
                for (unsigned long i=1; i<nFound; ++i)
                    if (ids[i-1] >= ids[i])
                    {
                        sorted = false;
                        break;
                    }
                if (!sorted)
                    std::sort(modifiable->idList_.begin(),
                              modifiable->idList_.end());
            }            
        }
        modifiable->initialized_ = true;
    }

    bool AbsReference::isIOCompatible(const CatalogEntry& r) const
    {
        return !classId_.name().empty() && 
                classId_.name() == r.type().name() &&
                ioProto_ == r.ioPrototype();
    }

    bool AbsReference::empty() const
    {
        if (!initialized_)
            initialize();
        return itemId_ == 0 && idList_.empty();
    }

    bool AbsReference::unique() const
    {
        if (!initialized_)
            initialize();
        return itemId_ && idList_.empty();
    }

    unsigned long AbsReference::size() const
    {
        if (!initialized_)
            initialize();
        return itemId_ ? 1UL : idList_.size();
    }

    unsigned long long AbsReference::id(const unsigned long index) const
    {
        if (!initialized_)
            initialize();
        if (itemId_ && index == 0UL)
            return itemId_;
        else if (index < idList_.size())
            return idList_[index];
        else
            throw gs::IOOutOfRange("In gs::AbsReference::id: "
                                    "index out of range");
    }

    CPP11_shared_ptr<const CatalogEntry> 
    AbsReference::indexedCatalogEntry(const unsigned long index) const
    {
        return archive_.catalogEntry(id(index));
    }

    bool AbsReference::isSameIOPrototype(const CatalogEntry& r) const
    {
        return ioProto_ == r.ioPrototype();
    }

    void AbsReference::addItemId(const unsigned long long idIn)
    {
        if (!idIn) throw gs::IOInvalidArgument(
            "In AbsReference::addItemId: invalid item id");
        const unsigned long mySize = itemId_ ? 1UL : idList_.size();
        switch (mySize)
        {
        case 0UL:
            itemId_ = idIn;
            break;

        case 1UL:
            idList_.reserve(2);
            idList_.push_back(itemId_);
            idList_.push_back(idIn);
            itemId_ = 0ULL;
            break;

        default:
            idList_.push_back(idIn);
            break;
        }
    }

    AbsReference::AbsReference(AbsArchive& ar, const ClassId& classId,
                               const char* ioPrototype,
                               const unsigned long long itemId)
        : archive_(ar),
          classId_(classId),
          ioProto_(ioPrototype ? ioPrototype : ""),
          searchId_(itemId),
          namePattern_(0),
          categoryPattern_(0),
          itemId_(0),
          initialized_(false)
    {
        if (!itemId) throw gs::IOInvalidArgument(
            "In AbsReference constructor: invalid item id");
    }

    AbsReference::AbsReference(AbsArchive& ar, const ClassId& classId,
                               const char* ioPrototype,
                               const SearchSpecifier& namePattern,
                               const SearchSpecifier& categoryPattern)
        : archive_(ar),
          classId_(classId),
          ioProto_(ioPrototype ? ioPrototype : ""),
          searchId_(0),
          namePattern_(namePattern),
          categoryPattern_(categoryPattern),
          itemId_(0),
          initialized_(false)
    {
    }
}
