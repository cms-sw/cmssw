#ifndef Common_Hash_h
#define Common_Hash_h

#include <string>

// We do not yet use the MD5Hash class
//
//#include "DataFormats/Common/interface/MD5Hash.h"
/*----------------------------------------------------------------------
  
Hash:

$Id: Hash.h,v 1.1.2.3 2006/06/27 21:05:17 paterno Exp $
----------------------------------------------------------------------*/
namespace edm {
  template <int I>
  class Hash {
  public:
    typedef std::string value_type;

    Hash() : hash_() {}
    explicit Hash(value_type const& v) : hash_(v) { }

    // compiler-generator copy c'tor, copy assignment, d'tor all OK


    // For now, just check the most basic: a default constructed
    // ParameterSetID is not valid. This is very crude: we are
    // assuming that nobody created a ParameterSetID from an empty
    // string, nor from any string that is not a valid string
    // representation of an MD5 checksum.
    bool isValid() const { return ! hash_.empty(); }

    bool operator< (Hash<I> const& other) const 
    { return hash_ < other.hash_; }


    bool operator> (Hash<I> const& other) const 
    { return other.hash_ < hash_; }

    bool operator== (Hash<I> const& other) const 
    { return hash_ == other.hash_; }

    bool operator!= (Hash<I> const& other) const 
    { return !(hash_ == other.hash_); }

    std::ostream& print(std::ostream& os) const
    { return os << hash_; }
    
  private:
    value_type hash_;
  };

  template <int I>
  inline
  std::ostream&
  operator<< (std::ostream& os, Hash<I> const& h)
  {
    return h.print(os);
  }

}
#endif
