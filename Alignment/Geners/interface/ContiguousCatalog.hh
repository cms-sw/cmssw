#ifndef GENERS_CONTIGUOUSCATALOG_HH_
#define GENERS_CONTIGUOUSCATALOG_HH_

#include <map>
#include <vector>

#include "Alignment/Geners/interface/AbsCatalog.hh"
#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"

namespace gs {
    class ContiguousCatalog : public AbsCatalog
    {
    public:
        // Default constructor creates an empty catalog
        inline ContiguousCatalog(const unsigned long long firstId=1)
            : firstId_(firstId ? firstId : 1ULL) {}
        inline virtual ~ContiguousCatalog() {}

        inline unsigned long long size() const {return records_.size();}
        inline unsigned long long smallestId() const {return firstId_;}
        inline unsigned long long largestId() const
            {return firstId_ + records_.size() - 1;}
        inline bool isContiguous() const {return true;}

        CPP11_shared_ptr<const CatalogEntry> retrieveEntry(
            unsigned long long id) const;

        bool retrieveStreampos(
            unsigned long long id, unsigned* compressionCode,
            unsigned long long* length, std::streampos* pos) const;

        // Add a new entry without an id (id will be generated internally
        // and returned)
        unsigned long long makeEntry(const ItemDescriptor& descriptor,
                                     unsigned compressionCode,
                                     unsigned long long itemLength,
                                     const ItemLocation& loc,
                                     unsigned long long offset=0ULL);

        inline const CatalogEntry* lastEntryMade() const
            {return lastEntry_.get();}

        // Search for matching entries based on item name and category
        void search(const SearchSpecifier& namePattern,
                    const SearchSpecifier& categoryPattern,
                    std::vector<unsigned long long>* idsFound) const;

        // Methods needed for I/O
        virtual ClassId classId() const {return ClassId(*this);}
        virtual bool write(std::ostream& os) const;

        static inline const char* classname() {return "gs::ContiguousCatalog";}
        static inline unsigned version() {return 2;}
        static ContiguousCatalog* read(const ClassId& id, std::istream& in);

    protected:
        virtual bool isEqual(const AbsCatalog&) const;

    private:
        typedef CPP11_shared_ptr<const CatalogEntry> SPtr;

        // In the following multimap, item name is the key and
        // item id is the value
        typedef std::multimap<std::string,unsigned long long> NameMap;

        // In the following map, item category is the key
        typedef std::map<std::string,NameMap> RecordMap;

        void findByName(const NameMap& nmap,
                        const SearchSpecifier& namePattern,
                        std::vector<unsigned long long>* found) const;

        std::vector<SPtr> records_;
        unsigned long long firstId_;
        RecordMap recordMap_;
        SPtr lastEntry_;

        static ContiguousCatalog* read_v1(std::istream& in);
    };
}

#endif // GENERS_CONTIGUOUSCATALOG_HH_

