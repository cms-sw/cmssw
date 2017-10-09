#include <cassert>
#include <utility>
#include <algorithm>

#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/ContiguousCatalog.hh"
#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

namespace gs {
    void ContiguousCatalog::findByName(
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
                    found->push_back(it->second);
        }
        else
        {
            const std::pair<Nameiter, Nameiter> limits =
                m.equal_range(namePattern.pattern());
            for (Nameiter it = limits.first; it != limits.second; ++it)
                found->push_back(it->second);
        }
    }

    unsigned long long ContiguousCatalog::makeEntry(
        const ItemDescriptor& descriptor,
        const unsigned compressionCode,
        const unsigned long long itemLength,
        const ItemLocation& loc,
        const unsigned long long offset)
    {
        const unsigned long long nextId = records_.size() + firstId_;
        lastEntry_ = SPtr(new CatalogEntry(
            descriptor, nextId, compressionCode, itemLength, loc, offset));
        records_.push_back(lastEntry_);
        recordMap_[descriptor.category()].insert(
            std::make_pair(descriptor.name(), nextId));

        return nextId;
    }

    void ContiguousCatalog::search(const SearchSpecifier& namePattern,
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

    bool ContiguousCatalog::isEqual(const AbsCatalog& other) const
    {
        if ((void*)this == (void*)(&other))
            return true;
        const ContiguousCatalog& r = static_cast<const ContiguousCatalog&>(other);
        if (firstId_ != r.firstId_)
            return false;
        if (recordMap_ != r.recordMap_)
            return false;
        const unsigned long nRecords = records_.size();
        if (nRecords != r.records_.size())
            return false;
        for (unsigned long i=0; i<nRecords; ++i)
            if (*records_[i] != *r.records_[i])
                return false;
        return true;
    }

    // Version 1 write function
    // bool ContiguousCatalog::write(std::ostream& os) const
    // {
    //     bool status = ClassId::makeId<CatalogEntry>().write(os) &&
    //                   ClassId::makeId<ItemLocation>().write(os);
    //     const unsigned long long sz = records_.size();
    //     for (unsigned long long i=0; i<sz && status; ++i)
    //         status = records_[i]->write(os);
    //     return status;
    // }

    bool ContiguousCatalog::write(std::ostream& os) const
    {
        const unsigned long long sz = records_.size();
        long long nRecords = sz;
        write_pod(os, nRecords);
        bool status = !os.fail() && ClassId::makeId<CatalogEntry>().write(os) &&
                                    ClassId::makeId<ItemLocation>().write(os);
        for (unsigned long long i=0; i<sz && status; ++i)
            status = records_[i]->write(os);
        return status;
    }

    ContiguousCatalog* ContiguousCatalog::read(const ClassId& cid,
                                               std::istream& in)
    {
        static const ClassId current(ClassId::makeId<ContiguousCatalog>());
        cid.ensureSameName(current);
        cid.ensureVersionInRange(1, version());

        if (cid.version() == 1)
            return read_v1(in);

        long long nRecords;
        read_pod(in, &nRecords);
        if (nRecords < 0)
            return read_v1(in);

        ClassId rId(in, 1);
        ClassId locId(in, 1);

        CPP11_auto_ptr<ContiguousCatalog> catalog(new ContiguousCatalog());
        bool firstEntry = true;
        for (long long recnum=0; recnum<nRecords; ++recnum)
        {
            CatalogEntry* rec = CatalogEntry::read(rId, locId, in);
            if (rec)
            {
                const unsigned long long id = rec->id();
                if (firstEntry)
                {
                    catalog->firstId_ = id;
                    firstEntry = false;
                }
                else
                {
                    const unsigned long long nextId = 
                        catalog->records_.size() + catalog->firstId_;
                    if (id != nextId)
                    {
                        delete rec;
                        throw IOInvalidData("In gs::ContiguousCatalog::read:"
                                            " unexpected item id. "
                                            "Catalog is corrupted.");
                    }
                }
                catalog->records_.push_back(SPtr(rec));
                catalog->recordMap_[rec->category()].insert(
                    std::make_pair(rec->name(), id));
            }
            else
                throw IOInvalidData("In gs::ContiguousCatalog::read:"
                                    " failed to read catalog entry");
        }
        return catalog.release();
    }

    ContiguousCatalog* ContiguousCatalog::read_v1(std::istream& in)
    {
        ClassId rId(in, 1);
        ClassId locId(in, 1);

        CPP11_auto_ptr<ContiguousCatalog> catalog(new ContiguousCatalog());
        bool firstEntry = true;
        for (in.peek(); !in.eof(); in.peek())
        {
            CatalogEntry* rec = CatalogEntry::read(rId, locId, in);
            if (rec)
            {
                const unsigned long long id = rec->id();
                if (firstEntry)
                {
                    catalog->firstId_ = id;
                    firstEntry = false;
                }
                else
                {
                    const unsigned long long nextId = 
                        catalog->records_.size() + catalog->firstId_;
                    if (id != nextId)
                    {
                        delete rec;
                        throw IOInvalidData("In gs::ContiguousCatalog::read_v1:"
                                            " unexpected item id. "
                                            "Catalog is corrupted.");
                    }
                }
                catalog->records_.push_back(SPtr(rec));
                catalog->recordMap_[rec->category()].insert(
                    std::make_pair(rec->name(), id));
            }
            else
                throw IOInvalidData("In gs::ContiguousCatalog::read_v1:"
                                    " failed to read catalog entry");
        }
        return catalog.release();
    }

    CPP11_shared_ptr<const CatalogEntry> ContiguousCatalog::retrieveEntry(
        const unsigned long long id) const
    {
        if (id >= firstId_ && id < records_.size() + firstId_)
            return records_[id - firstId_];
        else
        {
            CatalogEntry* ptr = 0;
            return CPP11_shared_ptr<const CatalogEntry>(ptr);
        }
    }

    bool ContiguousCatalog::retrieveStreampos(
        unsigned long long id, unsigned* compressionCode,
        unsigned long long* length, std::streampos* pos) const
    {
        assert(compressionCode);
        assert(length);
        assert(pos);

        if (id >= firstId_ && id < records_.size() + firstId_)
        {
            const CPP11_shared_ptr<const CatalogEntry>&
                rec(records_[id-firstId_]);
            *compressionCode = rec->compressionCode();
            *length = rec->itemLength();
            *pos = rec->location().streamPosition();
            return true;
        }
        else
            return false;
    }
}
