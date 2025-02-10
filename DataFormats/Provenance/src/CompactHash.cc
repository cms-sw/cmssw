#include "DataFormats/Provenance/interface/CompactHash.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <functional>
#include <cassert>

namespace {
  std::array<unsigned char, 16> convert(std::string const& v) {
    assert(v.size() == 16);
    std::array<unsigned char, 16> retValue;
    std::copy(v.begin(), v.end(), retValue.begin());
    return retValue;
  }
}  // namespace
namespace edm {
  namespace detail {
    // This string is the 16-byte, non-printable version.
    std::array<unsigned char, 16> const& InvalidCompactHash() {
      static std::array<unsigned char, 16> const invalid = convert(cms::MD5Result().compactForm());
      return invalid;
    }
  }  // namespace detail

  namespace compact_hash_detail {
    size_t smallHash_(value_type const& hash) {
      //NOTE: In future we could try to xor the first 8bytes into the second 8bytes of the string to make the hash
      std::hash<std::string_view> h;
      return h(std::string_view(reinterpret_cast<const char*>(hash.data()), hash.size()));
    }

    std::array<unsigned char, 16> fromHex_(std::string_view v) {
      cms::MD5Result temp;
      temp.fromHexifiedString(v);
      auto hash = temp.compactForm();
      std::array<unsigned char, 16> ret;
      std::copy(hash.begin(), hash.end(), ret.begin());
      return ret;
    }

    bool isValid_(value_type const& hash) { return hash != detail::InvalidCompactHash(); }

    void throwIfIllFormed(std::string_view v) {
      // Fixup not needed here.
      if (v.size() != 16) {
        throw Exception(errors::LogicError) << "Ill-formed CompactHash instance. "
                                            << "A string_view of size " << v.size() << " passed to constructor.";
      }
    }

    void toString_(std::string& result, value_type const& hash) {
      cms::MD5Result temp;
      copy_all(hash, temp.bytes.begin());
      result += temp.toString();
    }

    void toDigest_(cms::Digest& digest, value_type const& hash) {
      cms::MD5Result temp;
      copy_all(hash, temp.bytes.begin());
      digest.append(temp.toString());
    }

    std::ostream& print_(std::ostream& os, value_type const& hash) {
      cms::MD5Result temp;
      copy_all(hash, temp.bytes.begin());
      os << temp.toString();
      return os;
    }
  }  // namespace compact_hash_detail
}  // namespace edm
