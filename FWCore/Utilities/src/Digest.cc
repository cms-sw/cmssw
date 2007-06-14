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
    MD5Result const& invalidResult()
    {
      static const MD5Result val;      
      return val;
    }

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

  void set_to_default(MD5Result& val)
  {
    val.bytes[0] = 0xd4;
    val.bytes[1] = 0x1d;
    val.bytes[2] = 0x8c;
    val.bytes[3] = 0xd9;
    val.bytes[4] = 0x8f;
    val.bytes[5] = 0x00;
    val.bytes[6] = 0xb2;
    val.bytes[7] = 0x04;
    val.bytes[8] = 0xe9;
    val.bytes[9] = 0x80;
    val.bytes[10] = 0x09;
    val.bytes[11] = 0x98;
    val.bytes[12] = 0xec;
    val.bytes[13] = 0xf8;
    val.bytes[14] = 0x42;
    val.bytes[15] = 0x7e;
  }

  MD5Result::MD5Result() 
  {
    set_to_default(*this);
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
	  set_to_default(*this);
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
    return (*this != invalidResult());
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
