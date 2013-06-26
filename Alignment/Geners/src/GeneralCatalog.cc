#include <cassert>
#include <utility>
#include <algorithm>

#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"
#include "Alignment/Geners/interface/GeneralCatalog.hh"
#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

namespace gs {
    GeneralCatalog::GeneralCatalog()
        : smallestId_(1ULL),
          largestId_(0)
    {
    }

    void GeneralCatalog::findByName(
        const NameMap& m,
        const SearchSpecifier& namePattern,
        std::vector<unsigned long long>* found) const
    {
        typedef NameMap::const_iterator Nameiter;

        if (namePattern.useRegex())
        {
            const Nameiter itend = m.end();
            for (Nameiter it = m.begin(); it != itend; ++it)
                if (namePattern.matches(it->first))
                    found->push_back(it->second->id());
        }
        else
        {
            const std::pair<Nameiter, Nameiter> limits =
                m.equal_range(namePattern.pattern());
            for (Nameiter it = limits.first; it != limits.second; ++it)
                found->push_back(it->second->id());
        }
    }

    bool GeneralCatalog::addEntry(const SPtr inptr)
    {
        assert(inptr.get());

        const bool first = records_.empty();
        const unsigned long long id = inptr->id();
        if (id && records_.insert(std::make_pair(id, inptr)).second)
        {
            recordMap_[inptr->category()].insert(
                std::make_pair(inptr->name(), inptr));
            if (first)
            {
                smallestId_ = id;
                largestId_ = id;
            }
            else
            {
                if (id < smallestId_)
                    smallestId_ = id;
                if (id > largestId_)
                    largestId_ = id;
            }
            return true;
        }
        else
            return false;
    }

    bool GeneralCatalog::removeEntry(const unsigned long long id)
    {
        typedef RecordMap::iterator Mapiter;
        typedef NameMap::iterator Nameiter;

        IdMap::iterator rit = records_.find(id);
        if (rit == records_.end())
            return false;

        const SPtr item = rit->second;
        records_.erase(rit);

        bool found = false;
        const Mapiter mit = recordMap_.find(item->category());
        assert(mit != recordMap_.end());
        const std::pair<Nameiter, Nameiter> limits =
            mit->second.equal_range(item->name());
        for (Nameiter nit = limits.first; nit != limits.second; ++nit)
            if (nit->second->id() == id)
            {
                mit->second.erase(nit);
                found = true;
                break;
            }
        assert(found);
        if (mit->second.empty())
            recordMap_.erase(mit);

        if (records_.empty())
        {
            recordMap_.clear();
            smallestId_ = 0;
            largestId_ = 0;
        }
        else if (id == smallestId_ || id == largestId_)
        {
            IdMap::const_iterator it = records_.begin();
            smallestId_ = it->first;
            largestId_ = it->first;
            const IdMap::const_iterator itend = records_.end();
            for (++it; it != itend; ++it)
                if (it->first < smallestId_)
                    smallestId_ = it->first;
                else if (it->first > largestId_)
                    largestId_ = it->first;
        }
        return true;
    }

    unsigned long long GeneralCatalog::makeEntry(
        const ItemDescriptor& descriptor,
        const unsigned compressionCode,
        const unsigned long long itemLength,
        const ItemLocation& loc,
        const unsigned long long offset)
    {
        const unsigned long long nextId = records_.empty() ? 1ULL :
                                          largestId_ + 1;
        lastEntry_ = SPtr(new CatalogEntry(
            descriptor, nextId, compressionCode, itemLength, loc, offset));
        assert(addEntry(lastEntry_));
        return nextId;
    }

    void GeneralCatalog::search(const SearchSpecifier& namePattern,
                                const SearchSpecifier& categoryPattern,
                                std::vector<unsigned long long>* found) const
    {
        typedef RecordMap::const_iterator Mapiter;

        assert(found);
        found->clear();

        const Mapiter endMap = recordMap_.end();
        if (categoryPattern.useRegex())
        {
            for (Mapiter it = recordMap_.begin(); it != endMap; ++it)
                if (categoryPattern.matches(it->first))
                    findByName(it->second, namePattern, found);
        }
        else
        {
            Mapiter it = recordMap_.find(categoryPattern.pattern());
            if (it != endMap)
                findByName(it->second, namePattern, found);
        }
        std::sort(found->begin(), found->end());
    }

    bool GeneralCatalog::isEqual(const AbsCatalog& other) const
    {
        if ((void*)this == (void*)(&other))
            return true;
        const GeneralCatalog& r = static_cast<const GeneralCatalog&>(other);
        if (smallestId_ != r.smallestId_)
            return false;
        if (largestId_ != r.largestId_)
            return false;
        if (records_.size() != r.records_.size())
            return false;
        IdMap::const_iterator itend = records_.end();
        IdMap::const_iterator itend2 = r.records_.end();
        for (IdMap::const_iterator it = records_.begin();
             it != itend; ++it)
        {
            IdMap::const_iterator it2 = r.records_.find(it->first);
            if (it2 == itend2)
                return false;
            if (!(*it->second == *it2->second))
                return false;
        }
        return true;
    }

    // Version 1 write function
    // bool GeneralCatalog::write(std::ostream& os) const
    // {
    //     if (!ClassId::makeId<CatalogEntry>().write(os))
    //         return false;
    //     if (!ClassId::makeId<ItemLocation>().write(os))
    //         return false;

    //     // Sort item ids in the increasing order first
    //     std::vector<unsigned long long> idlist;
    //     const unsigned long sz = records_.size();
    //     idlist.reserve(sz);
    //     const IdMap::const_iterator itend = records_.end();
    //     for (IdMap::const_iterator it = records_.begin(); it != itend; ++it)
    //         idlist.push_back(it->first);
    //     std::sort(idlist.begin(), idlist.end());

    //     // Now, write the catalog records in the order of increasing ids
    //     for (unsigned long i=0; i<sz; ++i)
    //     {
    //         IdMap::const_iterator it = records_.find(idlist[i]);
    //         if (!it->second->write(os))
    //             return false;
    //     }

    //     return true;
    // }

    bool GeneralCatalog::write(std::ostream& os) const
    {
        const unsigned long sz = records_.size();
        long long ltmp = sz;
        write_pod(os, ltmp);
        if (os.fail())
            return false;
        if (!ClassId::makeId<CatalogEntry>().write(os))
            return false;
        if (!ClassId::makeId<ItemLocation>().write(os))
            return false;

        // Sort item ids in the increasing order first
        std::vector<unsigned long long> idlist;
        idlist.reserve(sz);
        const IdMap::const_iterator itend = records_.end();
        for (IdMap::const_iterator it = records_.begin(); it != itend; ++it)
            idlist.push_back(it->first);
        std::sort(idlist.begin(), idlist.end());

        // Now, write the catalog records in the order of increasing ids
        for (unsigned long i=0; i<sz; ++i)
        {
            IdMap::const_iterator it = records_.find(idlist[i]);
            if (!it->second->write(os))
                return false;
        }

        return true;
    }

    GeneralCatalog* GeneralCatalog::read(const ClassId& id, std::istream& in)
    {
        static const ClassId current(ClassId::makeId<GeneralCatalog>());
        id.ensureSameName(current);
        id.ensureVersionInRange(1, version());

        if (id.version() == 1)
            return read_v1(in);

        long long nRecords;
        read_pod(in, &nRecords);
        if (nRecords < 0)
            return read_v1(in);

        ClassId rId(in, 1);
        ClassId locId(in, 1);

        GeneralCatalog* catalog = new GeneralCatalog();
        bool ok = true;
        for (long long recnum=0; ok && recnum<nRecords; ++recnum)
        {
            CatalogEntry* rec = CatalogEntry::read(rId, locId, in);
            if (rec)
            {
                if (!catalog->addEntry(
                        CPP11_shared_ptr<const CatalogEntry>(rec)))
                    ok = false;
            }
            else
                ok = false;
        }

        if (!ok)
        {
            delete catalog;
            throw IOInvalidData("In gs::GeneralCatalog::read: "
                                "duplicate item id. "
                                "Catalog is corrupted.");
        }
        return catalog;
    }

    GeneralCatalog* GeneralCatalog::read_v1(std::istream& in)
    {
        ClassId rId(in, 1);
        ClassId locId(in, 1);

        GeneralCatalog* catalog = new GeneralCatalog();
        bool ok = true;
        for (in.peek(); ok && !in.eof(); in.peek())
        {
            CatalogEntry* rec = CatalogEntry::read(rId, locId, in);
            if (rec)
            {
                if (!catalog->addEntry(
                        CPP11_shared_ptr<const CatalogEntry>(rec)))
                    ok = false;
            }
            else
                ok = false;
        }

        if (!ok)
        {
            delete catalog;
            throw IOInvalidData("In gs::GeneralCatalog::read_v1: "
                                "duplicate item id. "
                                "Catalog is corrupted.");
        }
        return catalog;
    }

    CPP11_shared_ptr<const CatalogEntry> GeneralCatalog::retrieveEntry(
        const unsigned long long id) const
    {
        IdMap::const_iterator it = records_.find(id);
        if (it == records_.end())
        {
            CatalogEntry* ptr = 0;
            return CPP11_shared_ptr<const CatalogEntry>(ptr);
        }
        else
            return it->second;
    }

    bool GeneralCatalog::retrieveStreampos(
        unsigned long long id, unsigned* compressionCode,
        unsigned long long* length, std::streampos* pos) const
    {
        IdMap::const_iterator it = records_.find(id);
        if (it == records_.end())
            return false;

        assert(compressionCode);
        assert(length);
        assert(pos);

        *compressionCode = it->second->compressionCode();
        *length = it->second->itemLength();
        *pos = it->second->location().streamPosition();

        return true;
    }
}
