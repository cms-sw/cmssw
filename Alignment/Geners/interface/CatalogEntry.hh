#ifndef GENERS_CATALOGENTRY_HH_
#define GENERS_CATALOGENTRY_HH_

#include "Alignment/Geners/interface/ItemDescriptor.hh"
#include "Alignment/Geners/interface/ItemLocation.hh"

namespace gs {
    class CatalogEntry : public ItemDescriptor
    {
    public:
        // Default constructor returns an invalid entry
        CatalogEntry();

        // Use this constructor to build valid entries
        CatalogEntry(const ItemDescriptor& descr,
                     unsigned long long id,
                     unsigned compressionCode,
                     unsigned long long itemLength,
                     const ItemLocation& location,
                     unsigned long long offset=0ULL);

        inline virtual ~CatalogEntry() {}

        inline unsigned long long id() const {return id_;}
        inline unsigned long long offset() const {return offset_;}
        inline const ItemLocation& location() const {return location_;}
        inline unsigned long long itemLength() const {return len_;}
        inline unsigned compressionCode() const {return compressionCode_;}

        inline CatalogEntry& setStreamPosition(std::streampos pos)
            {location_.setStreamPosition(pos); return *this;}
        inline CatalogEntry& setURI(const char* newURI)
            {location_.setURI(newURI); return *this;}
        inline CatalogEntry& setCachedItemURI(const char* newURI)
            {location_.setCachedItemURI(newURI); return *this;}
        inline CatalogEntry& setOffset(const unsigned long long off)
            {offset_ = off; return *this;}

        // Dump a simple human-readable representation
        bool humanReadable(std::ostream& os) const;

        // Methods related to I/O for this record itself
        inline virtual ClassId classId() const {return ClassId(*this);}
        virtual bool write(std::ostream& of) const;

        static inline const char* classname() {return "gs::CatalogEntry";}
        static inline unsigned version() {return 1;}

        // "locId" is the class id for ItemLocation. Should be written
        // out by the catalog together with the CatalogEntry class id.
        static CatalogEntry* read(const ClassId& id, const ClassId& locId,
                                  std::istream& in);

    protected:
        virtual bool isEqual(const ItemDescriptor&) const;

    private:
        unsigned long long id_;
        unsigned long long len_;
        unsigned long long offset_;
        unsigned compressionCode_;
        ItemLocation location_;
    };
}

#endif // GENERS_CATALOGENTRY_HH_

