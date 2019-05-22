#ifndef GENERS_STRINGARCHIVE_HH_
#define GENERS_STRINGARCHIVE_HH_

#include "Alignment/Geners/interface/AbsArchive.hh"
#include "Alignment/Geners/interface/CharBuffer.hh"
#include "Alignment/Geners/interface/ContiguousCatalog.hh"

namespace gs {
  class StringArchive : public AbsArchive {
  public:
    inline StringArchive(const char *name = nullptr) : AbsArchive(name) {}
    ~StringArchive() override {}

    inline bool isOpen() const override { return true; }
    inline std::string error() const override { return std::string(""); }
    inline bool isReadable() const override { return true; }
    inline bool isWritable() const override { return true; }
    inline unsigned long long size() const override { return catalog_.size(); }
    inline unsigned long long smallestId() const override { return catalog_.smallestId(); }
    inline unsigned long long largestId() const override { return catalog_.largestId(); }
    inline bool idsAreContiguous() const override { return catalog_.isContiguous(); }
    inline bool itemExists(const unsigned long long id) const override { return catalog_.itemExists(id); }
    inline void itemSearch(const SearchSpecifier &namePattern,
                           const SearchSpecifier &categoryPattern,
                           std::vector<unsigned long long> *idsFound) const override {
      catalog_.search(namePattern, categoryPattern, idsFound);
    }

    inline CPP11_shared_ptr<const CatalogEntry> catalogEntry(const unsigned long long id) override {
      return catalog_.retrieveEntry(id);
    }

    inline void flush() override { stream_.flush(); }

    inline std::string str() const { return static_cast<const std::stringbuf *>(stream_.rdbuf())->str(); }

    // The following operation is equivalent to but much more efficient
    // than calling str() and finding out the size of the obtained string
    inline unsigned long dataSize() const { return stream_.size(); }

    // Methods related to I/O
    inline ClassId classId() const { return ClassId(*this); }
    bool write(std::ostream &of) const;

    static inline const char *classname() { return "gs::StringArchive"; }
    static inline unsigned version() { return 1; }

    // Can't have "restore" function: no default constructor.
    // Note that lastItemId() and lastItemLength() methods
    // of the parent class will return 0 when the archive is
    // read back from a stream and nothing has been inserted
    // into it yet.
    static StringArchive *read(const ClassId &id, std::istream &in);

  protected:
    bool isEqual(const AbsArchive &) const override;

  private:
    void search(AbsReference &reference) override;
    std::istream &inputStream(unsigned long long id, long long *sz) override;

    inline std::ostream &outputStream() override {
      lastpos_ = stream_.tellp();
      return stream_;
    }

    inline unsigned long long addToCatalog(const AbsRecord &record,
                                           const unsigned compressCode,
                                           const unsigned long long itemLength) override {
      return catalog_.makeEntry(record, compressCode, itemLength, ItemLocation(lastpos_, nullptr));
    }

    CharBuffer stream_;
    std::streampos lastpos_;
    ContiguousCatalog catalog_;
  };
}  // namespace gs

#endif  // GENERS_STRINGARCHIVE_HH_
