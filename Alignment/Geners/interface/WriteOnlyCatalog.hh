#ifndef GENERS_WRITEONLYCATALOG_HH_
#define GENERS_WRITEONLYCATALOG_HH_

#include <iostream>

#include "Alignment/Geners/interface/AbsCatalog.hh"
#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"

namespace gs {
  class WriteOnlyCatalog : public AbsCatalog {
  public:
    // The output stream should be dedicated exclusively to this catalog
    WriteOnlyCatalog(std::ostream &os, unsigned long long firstId = 1);
    inline ~WriteOnlyCatalog() override {}

    inline unsigned long long size() const override { return count_; }
    inline unsigned long long smallestId() const override { return smallestId_; }
    inline unsigned long long largestId() const override { return largestId_; }
    inline bool isContiguous() const override { return true; }

    // The following methods will cause a run-time error: there is
    // no way to read a write-only catalog or to search it
    CPP11_shared_ptr<const CatalogEntry> retrieveEntry(unsigned long long) const override;

    bool retrieveStreampos(unsigned long long id,
                           unsigned *compressionCode,
                           unsigned long long *length,
                           std::streampos *pos) const override;

    void search(const SearchSpecifier &namePattern,
                const SearchSpecifier &categoryPattern,
                std::vector<unsigned long long> *idsFound) const override;

    // Added entries will be immediately written out
    unsigned long long makeEntry(const ItemDescriptor &descriptor,
                                 unsigned compressionCode,
                                 unsigned long long itemLength,
                                 const ItemLocation &loc,
                                 unsigned long long offset = 0ULL) override;

    inline const CatalogEntry *lastEntryMade() const override { return lastEntry_.get(); }

    // Methods needed for I/O (not really useful,
    //  but must be overriden anyway)
    ClassId classId() const override { return ClassId(*this); }
    bool write(std::ostream &) const override;

    static inline const char *classname() { return "gs::WriteOnlyCatalog"; }
    static inline unsigned version() { return 2; }

    // The following function works only if there is a dynamic cast
    // which can convert "in" into std::ostream.
    static WriteOnlyCatalog *read(const ClassId &id, std::istream &in);

  protected:
    inline bool isEqual(const AbsCatalog &) const override { return false; }

  private:
    WriteOnlyCatalog(const WriteOnlyCatalog &) = delete;
    WriteOnlyCatalog &operator=(const WriteOnlyCatalog &) = delete;

    std::ostream &os_;
    unsigned long long count_;
    unsigned long long smallestId_;
    unsigned long long largestId_;
    CPP11_auto_ptr<const CatalogEntry> lastEntry_;
  };
}  // namespace gs

#endif  // GENERS_WRITEONLYCATALOG_HH_
