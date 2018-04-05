#ifndef L1TMuonEndCap_PtLUTWriter_h
#define L1TMuonEndCap_PtLUTWriter_h

#include <cstdint>
#include <string>
#include <vector>


class PtLUTWriter {
public:
  explicit PtLUTWriter();
  ~PtLUTWriter();

  typedef uint16_t               content_t;
  typedef uint64_t               address_t;
  typedef std::vector<content_t> table_t;

  void write(const std::string& lut_full_path, const uint16_t num_, const uint16_t denom_) const;

  void push_back(const content_t& pt);

  void set_version(content_t ver) { version_ = ver; }

  content_t get_version() const { return version_; }

private:
  mutable table_t ptlut_;
  content_t version_;
  bool ok_;
};

#endif
