#ifndef FWCore_Utilities_Digest_h
#define FWCore_Utilities_Digest_h

#include "md5.h"

#include <iosfwd>
#include <string>
#include <array>

namespace cms {

  struct MD5Result {
    // The default-constructed MD5Result is invalid; all others are
    // valid. The MD5 digest of the empty string is the value of the
    // default-constructed MD5Result.
    MD5Result();

    // This is the MD5 digest.
    std::array<unsigned char, 16> bytes;

    // Convert the digest to a printable string (the 'hexdigest')
    std::string toString() const;

    // The MD5 digest (not hexdigest) in string form
    // 'std::basic_string<char>', rather than
    // 'unsigned char [16]'
    std::string compactForm() const;

    // Set our data from the given hexdigest string.
    void fromHexifiedString(std::string const& s);

    bool isValid() const;
  };

  bool operator==(MD5Result const& a, MD5Result const& b);
  bool operator<(MD5Result const& a, MD5Result const& b);

  inline bool operator!=(MD5Result const& a, MD5Result const& b) { return !(a == b); }

  inline std::ostream& operator<<(std::ostream& os, MD5Result const& r) {
    os << r.toString();
    return os;
  }

  // Digest creates an MD5 digest of the given string. The digest can
  // be updated by using 'append'.
  class Digest {
  public:
    Digest();
    explicit Digest(std::string const& s);

    void append(std::string const& s);
    void append(const char* data, size_t size);

    MD5Result digest();

  private:
    md5_state_t state_;
  };
}  // namespace cms

#endif
