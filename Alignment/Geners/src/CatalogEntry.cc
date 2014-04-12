#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/CatalogEntry.hh"
#include "Alignment/Geners/interface/binaryIO.hh"

namespace gs {
    CatalogEntry::CatalogEntry()
        : ItemDescriptor(),
          id_(0),
          len_(0),
          location_(ItemLocation(std::streampos(0), 0))
    {
    }

    CatalogEntry::CatalogEntry(const ItemDescriptor& r,
                               const unsigned long long id,
                               const unsigned compressionCod,
                               const unsigned long long itemLength,
                               const ItemLocation& location,
                               const unsigned long long offset)
        : ItemDescriptor(r),
          id_(id),
          len_(itemLength),
          offset_(offset),
          compressionCode_(compressionCod),
          location_(location)
    {
        if (!id) throw gs::IOInvalidArgument(
            "In CatalogEntry constructor: invalid item id");
    }

    bool CatalogEntry::isEqual(const ItemDescriptor& other) const
    {
        if ((void*)this == (void*)(&other))
            return true;
        if (!ItemDescriptor::isEqual(other))
            return false;
        const CatalogEntry& r = static_cast<const CatalogEntry&>(other);
        return id_ == r.id_ && len_ == r.len_ &&
               offset_ == r.offset_ &&
               compressionCode_ == r.compressionCode_ &&
               location_ == r.location_;
    }

    bool CatalogEntry::write(std::ostream& of) const
    {
        type().write(of);
        write_pod(of, ioPrototype());
        write_pod(of, name());
        write_pod(of, category());
        write_pod(of, id_);
        write_pod(of, len_);
        write_pod(of, compressionCode_);

        // Most items will not have offsets
        unsigned char hasOffset = offset_ > 0ULL;
        write_pod(of, hasOffset);
        if (hasOffset)
            write_pod(of, offset_);

        location_.write(of);

        return !of.fail();
    }

    CatalogEntry* CatalogEntry::read(const ClassId& id,
                                     const ClassId& locId,
                                     std::istream& in)
    {
        static const ClassId current(ClassId::makeId<CatalogEntry>());
        current.ensureSameId(id);

        ClassId itemClass(in, 1);

        std::string ioPrototype, name, category;
        read_pod(in, &ioPrototype);
        read_pod(in, &name);
        read_pod(in, &category);

        unsigned long long itemId = 0, itemLen = 0;
        read_pod(in, &itemId);
        read_pod(in, &itemLen);

        unsigned coCode;
        read_pod(in, &coCode);

        unsigned long long offset = 0;
        unsigned char hasOffset = 0;
        read_pod(in, &hasOffset);
        if (hasOffset)
            read_pod(in, &offset);

        CatalogEntry* rec = 0;
        if (!in.fail())
        {
            CPP11_auto_ptr<ItemLocation> loc(ItemLocation::read(locId, in));
            if (loc.get())
                rec = new CatalogEntry(
                    ItemDescriptor(itemClass, ioPrototype.c_str(),
                                   name.c_str(), category.c_str()),
                    itemId, coCode, itemLen, *loc, offset);
        }
        return rec;
    }

    bool CatalogEntry::humanReadable(std::ostream& os) const
    {
        os << "Id: " << id_ << '\n'
           << "Class: " << type().id() << '\n'
           << "Name: " << name() << '\n'
           << "Category: " << category() << '\n'
           << "I/O prototype: " << ioPrototype() << '\n'
           << "URI: " << location().URI() << '\n'
           << "Cached: " << location().cachedItemURI() << '\n'
           << "Compression: " << compressionCode_ << '\n'
           << "Length: " << len_ << '\n'
           << "Streampos: " << location().streamPosition() << '\n'
           << "Offset: " << offset_
           << std::endl;
        return !os.fail();
    }
}
