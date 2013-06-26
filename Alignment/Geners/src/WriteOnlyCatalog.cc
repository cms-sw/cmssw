#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"
#include "Alignment/Geners/interface/WriteOnlyCatalog.hh"
#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

namespace gs {
    WriteOnlyCatalog::WriteOnlyCatalog(std::ostream& os,
                                       const unsigned long long firstId)
        : AbsCatalog(),
          os_(os),
          count_(0),
          smallestId_(firstId ? firstId : 1ULL),
          largestId_(0)
    {
    }

    unsigned long long WriteOnlyCatalog::makeEntry(
        const ItemDescriptor& descriptor,
        const unsigned compressionCode,
        const unsigned long long itemLen,
        const ItemLocation& loc,
        const unsigned long long off)
    {
        const unsigned long long id = count_ ? largestId_ + 1 : smallestId_;
        lastEntry_ = CPP11_auto_ptr<const CatalogEntry>(new CatalogEntry(
            descriptor, id, compressionCode, itemLen, loc, off));
        if (lastEntry_->write(os_))
        {
            ++count_;
            largestId_ = id;
            return id;
        }
        else
        {
            delete lastEntry_.release();
            return 0ULL;
        }
    }

    // Version 1 write function
    // bool WriteOnlyCatalog::write(std::ostream& os) const
    // {
    //     return ClassId::makeId<CatalogEntry>().write(os) &&
    //            ClassId::makeId<ItemLocation>().write(os);
    // }

    bool WriteOnlyCatalog::write(std::ostream& os) const
    {
        long long dummy = -1;
        write_pod(os, dummy);
        return !os.fail() && ClassId::makeId<CatalogEntry>().write(os) &&
                             ClassId::makeId<ItemLocation>().write(os);
    }

    WriteOnlyCatalog* WriteOnlyCatalog::read(const ClassId& id,
                                             std::istream& in)
    {
        static const ClassId current(ClassId::makeId<WriteOnlyCatalog>());
        id.ensureSameName(current);
        id.ensureVersionInRange(1, version());

        if (id.version() > 1)
        {
            long long dummy;
            read_pod(in, &dummy);
        }

        ClassId rId(in, 1);
        ClassId locId(in, 1);

        CPP11_auto_ptr<WriteOnlyCatalog> cat(new WriteOnlyCatalog(
                dynamic_cast<std::ostream&>(in)));
        bool firstEntry = true;
        for (in.peek(); !in.eof(); in.peek())
        {
            CatalogEntry* rec = CatalogEntry::read(rId, locId, in);
            if (rec)
            {
                bool ordered = true;
                const unsigned long long id = rec->id();
                if (firstEntry)
                {
                    cat->smallestId_ = id;
                    cat->count_ = 1;
                    cat->largestId_ = id;
                    firstEntry = false;
                }
                else
                {
                    if (id < cat->smallestId_)
                    {
                        cat->smallestId_ = id;
                        ++cat->count_;
                    }
                    else if (id > cat->largestId_)
                    {
                        cat->largestId_ = id;
                        ++cat->count_;
                    }
                    else
                        ordered = false;
                }
                delete rec;
                if (!ordered)
                    throw IOInvalidData("In gs::WriteOnlyCatalog::read: "
                                        "entry out of order. Catalog is "
                                        "likely to be corrupted.");
            }
            else
                throw IOInvalidData("In gs::WriteOnlyCatalog::read: "
                                    "failed to read catalog entry");
        }
        return cat.release();
    }

    CPP11_shared_ptr<const CatalogEntry> WriteOnlyCatalog::retrieveEntry(
        unsigned long long) const
    {
        throw IOReadFailure("In gs::WriteOnlyCatalog::retrieveEntry: "
                            "entries can not be retrieved "
                            "from a write-only catalog");
        return CPP11_shared_ptr<CatalogEntry>(
            reinterpret_cast<CatalogEntry*>(0));
    }

    bool WriteOnlyCatalog::retrieveStreampos(
        unsigned long long /* id */, unsigned* /* compressionCode */,
        unsigned long long* /* length */, std::streampos* /* pos */) const
    {
        throw IOReadFailure("In gs::WriteOnlyCatalog::retrieveStreampos: "
                            "stream positions can not be retrieved "
                            "from a write-only catalog");
        return false;
    }

    void WriteOnlyCatalog::search(const SearchSpecifier&,
                                  const SearchSpecifier&,
                                  std::vector<unsigned long long>*) const
    {
        throw IOReadFailure("In gs::WriteOnlyCatalog::search: "
                            "entries can not be searched "
                            "in a write-only catalog");
    }
}
