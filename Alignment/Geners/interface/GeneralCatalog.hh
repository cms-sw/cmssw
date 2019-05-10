#ifndef GENERS_GENERALCATALOG_HH_
#define GENERS_GENERALCATALOG_HH_

#include <map>

#include "Alignment/Geners/interface/AbsCatalog.hh"

namespace gs {
  class GeneralCatalog : public AbsCatalog {
  public:
    // Default constructor creates an empty catalog
    GeneralCatalog();
    inline ~GeneralCatalog() override {}

    inline unsigned long long size() const override { return records_.size(); }
    inline unsigned long long smallestId() const override { return smallestId_; }
    inline unsigned long long largestId() const override { return largestId_; }
    inline bool isContiguous() const override { return false; }
    inline bool itemExists(const unsigned long long id) const override { return records_.find(id) != records_.end(); }

    CPP11_shared_ptr<const CatalogEntry> retrieveEntry(const unsigned long long id) const override;

    bool retrieveStreampos(unsigned long long id,
                           unsigned *compressionCode,
                           unsigned long long *length,
                           std::streampos *pos) const override;

    // Add a new entry without an id (id will be generated internally
    // and returned)
    unsigned long long makeEntry(const ItemDescriptor &descriptor,
                                 unsigned compressionCode,
                                 unsigned long long itemLength,
                                 const ItemLocation &loc,
                                 unsigned long long offset = 0ULL) override;

    inline const CatalogEntry *lastEntryMade() const override { return lastEntry_.get(); }

    // Add a new entry with id (presumably, from another catalog).
    // Returns "true" on success. The entry is not included (and "false"
    // is returned) in case the entry with the given id already exists.
    bool addEntry(CPP11_shared_ptr<const CatalogEntry> ptr);

    // Remove an entry with the given id. "false" is returned in case
    // an entry with the specified id does not exist.
    bool removeEntry(unsigned long long id);

    // Search for matching entries based on item name and category
    void search(const SearchSpecifier &namePattern,
                const SearchSpecifier &categoryPattern,
                std::vector<unsigned long long> *idsFound) const override;

    // Methods needed for I/O
    ClassId classId() const override { return ClassId(*this); }
    bool write(std::ostream &os) const override;

    static inline const char *classname() { return "gs::GeneralCatalog"; }
    static inline unsigned version() { return 2; }
    static GeneralCatalog *read(const ClassId &id, std::istream &in);

  protected:
    bool isEqual(const AbsCatalog &) const override;

  private:
    typedef CPP11_shared_ptr<const CatalogEntry> SPtr;

    // In the following multimap, item name is the key and
    // catalog entry pointer is the value
    typedef std::multimap<std::string, SPtr> NameMap;

    // In the following map, item category is the key
    typedef std::map<std::string, NameMap> RecordMap;

    // In the following map, item id is the key
    typedef std::map<unsigned long long, SPtr> IdMap;

    void findByName(const NameMap &nmap,
                    const SearchSpecifier &namePattern,
                    std::vector<unsigned long long> *found) const;

    IdMap records_;
    RecordMap recordMap_;
    unsigned long long smallestId_;
    unsigned long long largestId_;
    SPtr lastEntry_;

    static GeneralCatalog *read_v1(std::istream &in);
  };
}  // namespace gs

#endif  // GENERS_GENERALCATALOG_HH_
