#ifndef DataFormats_Provenance_CompactHash_h
#define DataFormats_Provenance_CompactHash_h

#include <string_view>
#include <array>
#include <functional>

namespace cms {
  class Digest;
}

namespace edm {

  namespace detail {
    // This string is the 16-byte, non-printable version.
    std::array<unsigned char, 16> const& InvalidCompactHash();
  }  // namespace detail

  namespace compact_hash_detail {
    using value_type = std::array<unsigned char, 16>;
    void toString_(std::string& result, value_type const& hash);
    void toDigest_(cms::Digest& digest, value_type const& hash);
    std::ostream& print_(std::ostream& os, value_type const& hash);
    bool isValid_(value_type const& hash);
    size_t smallHash_(value_type const& hash);
    value_type fromHex_(std::string_view);
    void throwIfIllFormed(std::string_view v);
  }  // namespace compact_hash_detail

  template <int I>
  class CompactHash {
  public:
    typedef compact_hash_detail::value_type value_type;

    CompactHash();
    explicit CompactHash(value_type const& v);
    explicit CompactHash(std::string_view v);

    CompactHash(CompactHash<I> const&) = default;
    CompactHash<I>& operator=(CompactHash<I> const& iRHS) = default;

    CompactHash(CompactHash<I>&&) = default;
    CompactHash<I>& operator=(CompactHash<I>&&) = default;

    void reset();

    // For now, just check the most basic: a default constructed
    // ParameterSetID is not valid. This is very crude: we are
    // assuming that nobody created a ParameterSetID from an empty
    // string, nor from any string that is not a valid string
    // representation of an MD5 checksum.
    bool isValid() const;

    bool operator<(CompactHash<I> const& other) const;
    bool operator>(CompactHash<I> const& other) const;
    bool operator==(CompactHash<I> const& other) const;
    bool operator!=(CompactHash<I> const& other) const;
    std::ostream& print(std::ostream& os) const;
    void toString(std::string& result) const;
    void toDigest(cms::Digest& digest) const;

    // Return the 16-byte (non-printable) string form.
    value_type const& compactForm() const;

    ///returns a short hash which can be used with hashing containers
    size_t smallHash() const;

    //Used by ROOT storage
    // CMS_CLASS_VERSION(3) // This macro is not defined here, so expand it.
    static short Class_Version() { return 3; }

  private:
    template <typename Op>
    bool compareUsing(CompactHash<I> const& iOther, Op op) const {
      return op(this->hash_, iOther.hash_);
    }

    value_type hash_;
  };

  //--------------------------------------------------------------------
  //
  // Implementation details follow...
  //--------------------------------------------------------------------

  template <int I>
  inline CompactHash<I>::CompactHash() : hash_(detail::InvalidCompactHash()) {}

  template <int I>
  inline CompactHash<I>::CompactHash(value_type const& v) : hash_(v) {}

  template <int I>
  inline CompactHash<I>::CompactHash(std::string_view v) {
    if (v.size() == 32) {
      hash_ = compact_hash_detail::fromHex_(v);
    } else {
      compact_hash_detail::throwIfIllFormed(v);
      std::copy(v.begin(), v.end(), hash_.begin());
    }
  }

  template <int I>
  inline void CompactHash<I>::reset() {
    hash_ = detail::InvalidCompactHash();
  }

  template <int I>
  inline bool CompactHash<I>::isValid() const {
    return compact_hash_detail::isValid_(hash_);
  }

  template <int I>
  inline bool CompactHash<I>::operator<(CompactHash<I> const& other) const {
    return this->compareUsing(other, std::less<value_type>());
  }

  template <int I>
  inline bool CompactHash<I>::operator>(CompactHash<I> const& other) const {
    return this->compareUsing(other, std::greater<value_type>());
  }

  template <int I>
  inline bool CompactHash<I>::operator==(CompactHash<I> const& other) const {
    return this->compareUsing(other, std::equal_to<value_type>());
  }

  template <int I>
  inline bool CompactHash<I>::operator!=(CompactHash<I> const& other) const {
    return this->compareUsing(other, std::not_equal_to<value_type>());
  }

  template <int I>
  inline std::ostream& CompactHash<I>::print(std::ostream& os) const {
    return compact_hash_detail::print_(os, hash_);
  }

  template <int I>
  inline void CompactHash<I>::toString(std::string& result) const {
    compact_hash_detail::toString_(result, hash_);
  }

  template <int I>
  inline void CompactHash<I>::toDigest(cms::Digest& digest) const {
    compact_hash_detail::toDigest_(digest, hash_);
  }

  template <int I>
  inline typename CompactHash<I>::value_type const& CompactHash<I>::compactForm() const {
    return hash_;
  }

  template <int I>
  inline size_t CompactHash<I>::smallHash() const {
    return compact_hash_detail::smallHash_(hash_);
  }

  template <int I>
  inline std::ostream& operator<<(std::ostream& os, CompactHash<I> const& h) {
    return h.print(os);
  }

}  // namespace edm
#endif
