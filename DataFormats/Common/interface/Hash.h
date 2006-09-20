#ifndef Common_Hash_h
#define Common_Hash_h

#include <algorithm>
#include <string>

#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/EDMException.h"

/*----------------------------------------------------------------------
  
Hash:

$Id: Hash.h,v 1.6 2006/09/20 16:07:08 paterno Exp $
----------------------------------------------------------------------*/
namespace edm {

  namespace detail
  {
    // This string is the 16-byte, non-printable version.
    std::string const& InvalidHash();
  }

  template <int I>
  class Hash {
  public:
    typedef std::string value_type;

    Hash();
    explicit Hash(value_type const& v);

    // compiler-generator copy c'tor, copy assignment, d'tor all OK

    // For now, just check the most basic: a default constructed
    // ParameterSetID is not valid. This is very crude: we are
    // assuming that nobody created a ParameterSetID from an empty
    // string, nor from any string that is not a valid string
    // representation of an MD5 checksum.
    bool isValid() const;

    bool operator< (Hash<I> const& other) const;
    bool operator> (Hash<I> const& other) const;
    bool operator== (Hash<I> const& other) const;
    bool operator!= (Hash<I> const& other) const;
    std::ostream& print(std::ostream& os) const;
    void swap(Hash<I>& other);

    // Return the 16-byte (non-printable) string form.
    value_type compactForm() const;
    
  private:

    /// Hexified version of data *must* contain a multiple of 2
    /// bytes. If it does not, throw an exception.
    void throwIfIllFormed() const;

    // 'Fix' the string data member of this Hash, i.e., if it is in
    // the hexified (32 byte) representation, make it be in the
    // 16-byte (unhexified) representation.
    void fixup() const;

    mutable value_type hash_;
  };


  //--------------------------------------------------------------------
  //
  // Implementation details follow...
  //--------------------------------------------------------------------


  template <int I>
  inline
  Hash<I>::Hash() : 
    hash_() 
  {
    fixup();
  }

  template <int I>
  inline
  Hash<I>::Hash(typename Hash<I>::value_type const& v) :
    hash_(v)
  {
    fixup();
  }

  template <int I>
  inline
  bool 
  Hash<I>::isValid() const
  {
    return hash_ != edm::detail::InvalidHash();
  }

  template <int I>
  inline
  bool
  Hash<I>::operator< (Hash<I> const& other) const
  {
    return hash_ < other.hash_; 
  }

  template <int I>
  inline
  bool 
  Hash<I>::operator> (Hash<I> const& other) const 
  {
    return other.hash_ < hash_;
  }

  template <int I>
  inline
  bool 
  Hash<I>::operator== (Hash<I> const& other) const 
  {
    return hash_ == other.hash_;
  }

  template <int I>
  inline
  bool 
  Hash<I>::operator!= (Hash<I> const& other) const 
  {
    return !(hash_ == other.hash_);
  }

  template <int I>
  inline
  std::ostream& 
  Hash<I>::print(std::ostream& os) const
  {
    cms::MD5Result temp;
    std::copy(hash_.begin(), hash_.end(), temp.bytes);
    os << temp.toString();
    return os;
  }

  template <int I>
  inline
  void 
  Hash<I>::swap(Hash<I>& other) 
  {
    hash_.swap(other.hash_);
  }

  template <int I>
  inline
  typename Hash<I>::value_type
  Hash<I>::compactForm() const
  {
    return hash_;
  }

  template <int I>
  inline
  void 
  Hash<I>::throwIfIllFormed() const 
  {
    if ( hash_.size() % 2 == 1 )
      {
	throw edm::Exception(edm::errors::LogicError)
	  << "Ill-formed Hash instance. "
	  << "Please report this to the core framework developers";
      }
  }

  // Note: this template is not declared 'inline' because of the
  // switch statement.

  template <int I>
  void 
  Hash<I>::fixup() const {
    switch (hash_.size() ) {
    case 0:
      {
	hash_ = edm::detail::InvalidHash();
	break;
      }	
    case 16: 
      {
	break;
      }
	
    case 32:
      {
	cms::MD5Result temp;
	temp.fromHexifiedString(hash_);
	hash_ = temp.compactForm();
	break;
      }
	  
    default:
      {
	throw edm::Exception(edm::errors::LogicError)
	  << "edm::Hash<> instance with data in illegal state:\n"
	  << hash_
	  << "\nPlease report this to the core framework developers";
      }
    }
  }


  // Free swap function
  template <int I>
  inline
  void
  swap(Hash<I>& a, Hash<I>& b) 
  {
    a.swap(b);
  }

  template <int I>
  inline
  std::ostream&
  operator<< (std::ostream& os, Hash<I> const& h)
  {
    return h.print(os);
  }

}
#endif
