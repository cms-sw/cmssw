#ifndef DataFormats_Provenance_Hash_h
#define DataFormats_Provenance_Hash_h

#include <string>

/*----------------------------------------------------------------------
  
Hash:

  Note: The call to 'fixup' in every member function is a temporary
  measure for backwards compatibility. It is necessary in every function
  because Root creates instances of the class *without* using the
  interface of the class, thus making it insufficient to assure that
  all constructors make corrected instances.

----------------------------------------------------------------------*/
namespace cms {
  class Digest;
}

namespace edm {

  namespace detail {
    // This string is the 16-byte, non-printable version.
    std::string const& InvalidHash();
  }

  namespace hash_detail {
    typedef std::string value_type;
    value_type compactForm_(value_type const& hash);
    void fixup_(value_type& hash);
    bool isCompactForm_(value_type const& hash);
    bool isValid_(value_type const& hash);
    void throwIfIllFormed(value_type const& hash);
    void toString_(std::string& result, value_type const& hash);
    void toDigest_(cms::Digest& digest, value_type const& hash);
    std::ostream& print_(std::ostream& os, value_type const& hash);
  }

  template <int I>
  class Hash {
  public:
    typedef hash_detail::value_type value_type;

    Hash();
    explicit Hash(value_type const& v);

    Hash(Hash<I> const&);
    Hash<I>& operator=(Hash<I> const& iRHS);

    void reset();

    // For now, just check the most basic: a default constructed
    // ParameterSetID is not valid. This is very crude: we are
    // assuming that nobody created a ParameterSetID from an empty
    // string, nor from any string that is not a valid string
    // representation of an MD5 checksum.
    bool isValid() const;

    bool operator<(Hash<I> const& other) const;
    bool operator>(Hash<I> const& other) const;
    bool operator==(Hash<I> const& other) const;
    bool operator!=(Hash<I> const& other) const;
    std::ostream& print(std::ostream& os) const;
    void toString(std::string& result) const;
    void toDigest(cms::Digest& digest) const;
    void swap(Hash<I>& other);

    // Return the 16-byte (non-printable) string form.
    value_type compactForm() const;
    
    bool isCompactForm() const;
    
    //Used by ROOT storage
    // CMS_CLASS_VERSION(10) // This macro is not defined here, so expand it.
    static short Class_Version() {return 10;}

  private:

    /// Hexified version of data *must* contain a multiple of 2
    /// bytes. If it does not, throw an exception.
    void throwIfIllFormed() const;

    template<typename Op>
      bool
      compareUsing(Hash<I> const& iOther, Op op) const {
        bool meCF = hash_detail::isCompactForm_(hash_);
        bool otherCF = hash_detail::isCompactForm_(iOther.hash_);
        if(meCF == otherCF) {
          return op(this->hash_,iOther.hash_);
        }
        //copy constructor will do compact form conversion
        if(meCF) {
           Hash<I> temp(iOther);
           return op(this->hash_,temp.hash_);
        } 
        Hash<I> temp(*this);
        return op(temp.hash_,iOther.hash_);
      }

    value_type hash_;
  };


  //--------------------------------------------------------------------
  //
  // Implementation details follow...
  //--------------------------------------------------------------------


  template <int I>
  inline
  Hash<I>::Hash() : hash_(detail::InvalidHash()) {}

  template <int I>
  inline
  Hash<I>::Hash(typename Hash<I>::value_type const& v) : hash_(v) {
    hash_detail::fixup_(hash_);
  }

  template <int I>
  inline
  Hash<I>::Hash(Hash<I> const& iOther) : hash_(iOther.hash_) {
     hash_detail::fixup_(hash_);
  }

  template <int I>
  inline
  Hash<I>& 
  Hash<I>::operator=(Hash<I> const& iRHS) {
    hash_ = iRHS.hash_;
    hash_detail::fixup_(hash_);
    return *this;
  }
  
  template <int I>
  inline
  void 
  Hash<I>::reset() {
    hash_ = detail::InvalidHash();
  }
  
  template <int I>
  inline
  bool 
  Hash<I>::isValid() const {
    return hash_detail::isValid_(hash_);
  }
  
  template <int I>
  inline
  bool
  Hash<I>::operator<(Hash<I> const& other) const {
    return this->compareUsing(other, std::less<std::string>());
  }

  template <int I>
  inline
  bool 
  Hash<I>::operator>(Hash<I> const& other) const {
    return this->compareUsing(other, std::greater<std::string>());
  }

  template <int I>
  inline
  bool 
  Hash<I>::operator==(Hash<I> const& other) const {
    return this->compareUsing(other, std::equal_to<std::string>());
  }

  template <int I>
  inline
  bool 
  Hash<I>::operator!=(Hash<I> const& other) const {
    return this->compareUsing(other, std::not_equal_to<std::string>());
  }

  template <int I>
  inline
  std::ostream& 
  Hash<I>::print(std::ostream& os) const {
    return hash_detail::print_(os, hash_);
  }

  template <int I>
  inline
  void
  Hash<I>::toString(std::string& result) const {
    hash_detail::toString_(result, hash_);
  }

  template <int I>
  inline
  void
  Hash<I>::toDigest(cms::Digest& digest) const {
    hash_detail::toDigest_(digest, hash_);
  }

  template <int I>
  inline
  void 
  Hash<I>::swap(Hash<I>& other) {
    hash_.swap(other.hash_);
  }

  template <int I>
  inline
  typename Hash<I>::value_type
  Hash<I>::compactForm() const {
    return hash_detail::compactForm_(hash_);
  }

  // Note: this template is not declared 'inline' because of the
  // switch statement.

  template <int I>
  inline
  bool Hash<I>::isCompactForm() const {
    return hash_detail::isCompactForm_(hash_);
  }
  

  // Free swap function
  template <int I>
  inline
  void
  swap(Hash<I>& a, Hash<I>& b) {
    a.swap(b);
  }

  template <int I>
  inline
  std::ostream&
  operator<<(std::ostream& os, Hash<I> const& h) {
    return h.print(os);
  }

}
#endif
