#include "FWCore/Utilities/interface/Digest.h"
#include <iomanip>
#include <sstream>
#include <string>

namespace cms
{

  std::string MD5Result::toString() const
  {
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (int i = 0 ; i < 16 ; ++i) os << std::setw(2) << static_cast<int>(bytes[i]);
    return os.str();
  }

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
