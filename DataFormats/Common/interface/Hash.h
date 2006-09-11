#ifndef Common_Hash_h
#define Common_Hash_h

#include <string>
#include "FWCore/Utilities/interface/EDMException.h"

// We do not yet use the MD5Hash class
//
//#include "DataFormats/Common/interface/MD5Hash.h"
/*----------------------------------------------------------------------
  
Hash:

$Id: Hash.h,v 1.4 2006/08/10 23:34:53 wmtan Exp $
----------------------------------------------------------------------*/
namespace edm {


  namespace detail {
    inline
    std::string::value_type
    unhexify(std::string::value_type hexed)
    {
      switch (hexed) {
      case '0': case '1': case '2': case '3': case '4':
      case '5': case '6': case '7': case '8': case '9':
	return hexed - '0';
      case 'a': case 'b': case 'c': case 'd': case 'e': case 'f':
	return hexed - 'a' + 10;
      case 'A': case 'B': case 'C': case 'D': case 'E': case 'F':
	return hexed - 'A' + 10;
      default:
	throw edm::Exception(edm::errors::LogicError)
	  << "Non-hex character in Hash "
	  << "Please report this to the core framework developers";
      }
      // We never get here.
      return '\0';
    }
  } // namespace detail

  template <int I>
  class Hash {
  public:
    typedef std::string value_type;

    Hash() : hash_() {}
    explicit Hash(value_type const& v) : hash_(v) { throwIfIllFormed(); }

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
    
    void swap(Hash<I>& other) {
      std::swap(hash_, other.hash_);
    }

    /// Hexified version of data *must* contain a multiple of 2
    /// bytes. If it does not, throw an exception.
    void throwIfIllFormed() const {
      if ( hash_.size() % 2 == 1 )
	{
	  throw edm::Exception(edm::errors::LogicError)
	    << "Ill-formed Hash instance. "
	    << "Please report this to the core framework developers";
	}
    }

    value_type compactForm() const {
      value_type v;
      if (! hash_.empty() )
	{
	  value_type::size_type nloop = hash_.size()/2;
	  v.resize(nloop);
	  value_type::const_iterator it = hash_.begin();
	  for (value_type::size_type i = 0; i != nloop; ++i )
	    {
	      // first nybble
	      v[i] = ( detail::unhexify(*it++) << 4 );
	      // second nybble
	      v[i] += ( detail::unhexify(*it++) );
	    }
	}
      return v;
    }
    
  private:
    value_type hash_;
  };

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
