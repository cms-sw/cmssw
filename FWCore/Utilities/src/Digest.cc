#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Digest.h"

using std::string;

namespace cms
{
  namespace
  {
    MD5Result invalidResult = Digest().digest();


    
    char unhexify(char hexed)
    {
      switch (hexed) 
	{
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
      // We never get here; return put in place to calm the compiler's
      // anxieties.
      return '\0';
    }
  }

  //--------------------------------------------------------------------
  //
  // MD5Result and associated free functions
  //

  MD5Result::MD5Result() 
  {
    std::copy(invalidResult.bytes,
	      invalidResult.bytes + sizeof(invalidResult.bytes),
	      bytes);
  }


  string MD5Result::toString() const
  {
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (size_t i = 0 ; i < sizeof(bytes) ; ++i) 
      os << std::setw(2) << static_cast<int>(bytes[i]);
    return os.str();
  }

  string MD5Result::compactForm() const
  {
    // This is somewhat dangerous, because the conversion of 'unsigned
    // char' to 'char' may be undefined if 'char' is a signed type
    // (4.7p3 in the Standard).
    const char* p = reinterpret_cast<const char*>(&bytes[0]);
    return string(p, p+sizeof(bytes));
  }

  void MD5Result::fromHexifiedString(string const& hexy)
  {
    switch (hexy.size())
      {
      case 0:
	{
	  *this = invalidResult;
	}
	break;
      case 32:
	{
	  string::const_iterator it = hexy.begin();
	  for (size_t i = 0; i != 16; ++i)
	    {
	      // first nybble
	      bytes[i] = ( unhexify(*it++) << 4 );
	      // second nybble
	      bytes[i] += ( unhexify(*it++) );
	    }
	}
	break;
      default:
	{
	  // Not really sure of what sort of exception to throw...
	  throw edm::Exception(edm::errors::LogicError)
	    << "String of illegal length: "
	    << hexy.size()
	    << " given to MD5Result::fromHexifiedString";	  
	}
      }
  }

  bool MD5Result::isValid() const
  {
    return (*this != invalidResult);
  }

  bool operator==(MD5Result const& a, MD5Result const& b)
  {
    return std::equal(a.bytes, a.bytes+sizeof(a.bytes), b.bytes);
  }


  bool operator< (MD5Result const& a, MD5Result const& b)
  {
    return std::lexicographical_compare(a.bytes, 
					a.bytes+sizeof(a.bytes), 
					b.bytes,
					b.bytes+sizeof(b.bytes));
  }



  //--------------------------------------------------------------------
  //
  // Digest
  //

  Digest::Digest() :
    state_()
  {
    md5_init(&state_);
  }

  Digest::Digest(string const& s) :
    state_()
  {
    md5_init(&state_);
    this->append(s);
  }

  void Digest::append(string const& s) 
  {
    const md5_byte_t* data = reinterpret_cast<const md5_byte_t*>(s.data());
    md5_append(&state_, const_cast<md5_byte_t*>(data), s.size());
  }

  MD5Result Digest::digest() const
  {
    MD5Result aDigest;
    md5_finish(&state_, aDigest.bytes);
    return aDigest;
  }
}
