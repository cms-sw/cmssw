#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>

#include "FWCore/Utilities/interface/Digest.h"

namespace cms
{
  namespace
  {
    MD5Result invalidResult = Digest().digest();
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


  std::string MD5Result::toString() const
  {
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (size_t i = 0 ; i < sizeof(bytes) ; ++i) 
      os << std::setw(2) << static_cast<int>(bytes[i]);
    return os.str();
  }

  std::string MD5Result::compactForm() const
  {
    // This is somewhat dangerous, because the conversion of 'unsigned
    // char' to 'char' may be undefined if 'char' is a signed type
    // (4.7p3 in the Standard).
    const char* p = reinterpret_cast<const char*>(&bytes[0]);
    return std::string(p, p+sizeof(bytes));
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

  Digest::Digest(std::string const& s) :
    state_()
  {
    md5_init(&state_);
    this->append(s);
  }

  void Digest::append(std::string const& s) 
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
