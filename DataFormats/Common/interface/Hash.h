#ifndef Common_Hash_h
#define Common_Hash_h

#include <string>
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/EDMException.h"

/*----------------------------------------------------------------------
  
Hash:

$Id: Hash.h,v 1.5 2006/09/11 21:28:34 paterno Exp $
----------------------------------------------------------------------*/
namespace edm {


  //   namespace detail {
  //     inline
  //     std::string::value_type
  //     unhexify(std::string::value_type hexed)
  //     {
  //       switch (hexed) {
  //       case '0': case '1': case '2': case '3': case '4':
  //       case '5': case '6': case '7': case '8': case '9':
  // 	return hexed - '0';
  //       case 'a': case 'b': case 'c': case 'd': case 'e': case 'f':
  // 	return hexed - 'a' + 10;
  //       case 'A': case 'B': case 'C': case 'D': case 'E': case 'F':
  // 	return hexed - 'A' + 10;
  //       default:
  // 	throw edm::Exception(edm::errors::LogicError)
  // 	  << "Non-hex character in Hash "
  // 	  << "Please report this to the core framework developers";
  //       }
  //       // We never get here.
  //       return '\0';
  //     }
  //   } // namespace detail

  namespace
  {
    // This string is the 16-byte, non-printable version.
    std::string const& InvalidHash()
    {
      static const std::string invalid = cms::MD5Result().compactForm();
      return invalid;
    }
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

    // Return the 16-byte (non-printable) string form
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
    return hash_ != InvalidHash();
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
    return os << hash_;
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
    //     value_type v;
    //     if (! hash_.empty() )
    //       {
    // 	value_type::size_type nloop = hash_.size()/2;
    // 	v.resize(nloop);
    // 	value_type::const_iterator it = hash_.begin();
    // 	for (value_type::size_type i = 0; i != nloop; ++i )
    // 	  {
    // 	    // first nybble
    // 	    v[i] = ( detail::unhexify(*it++) << 4 );
    // 	    // second nybble
    // 	    v[i] += ( detail::unhexify(*it++) );
    // 	  }
    //       }
    //     return v;
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
	hash_ = InvalidHash();
	// 	  hash_.resize(16);
	// 	  hash_[0] = 0xd4;
	// 	  hash_[1] = 0x1d;
	// 	  hash_[2] = 0x8c;
	// 	  hash_[3] = 0xd9;
	// 	  hash_[4] = 0x8f;
	// 	  hash_[5] = 0x00;
	// 	  hash_[6] = 0xb2;
	// 	  hash_[7] = 0x04;
	// 	  hash_[8] = 0xe9;
	// 	  hash_[9] = 0x80;
	// 	  hash_[10] = 0x09;
	// 	  hash_[11] = 0x98;
	// 	  hash_[12] = 0xec;
	// 	  hash_[13] = 0xf8;
	// 	  hash_[14] = 0x42;
	// 	  hash_[15] = 0x7e;
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
	// 	  std::string unhexified(16, ' ');
	// 	  std::string::const_iterator it = hash_.begin();
	// 	  for (std::string::size_type i = 0; i != hash_.size(); ++i)
	// 	    {
	// 	      unhexified[i]  = (detail::unhexify(*it++) << 4);
	// 	      unhexified[i] +=  detail::unhexify(*it++);
	// 	    }
	// 	  hash_.swap(unhexified);
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
