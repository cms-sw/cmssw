#include "DataFormats/Common/interface/MD5Hash.h"

/*----------------------------------------------------------------------

$Id: MD5Hash.cc,v 1.2 2006/07/06 18:34:06 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {

  MD5Hash::MD5Hash() 
  {
    data[0] = 0xd4;
    data[1] = 0x1d;
    data[2] = 0x8c;
    data[3] = 0xd9;
    data[4] = 0x8f;
    data[5] = 0x00;
    data[6] = 0xb2;
    data[7] = 0x04;
    data[8] = 0xe9;
    data[9] = 0x80;
    data[10] = 0x09;
    data[11] = 0x98;
    data[12] = 0xec;
    data[13] = 0xf8;
    data[14] = 0x42;
    data[15] = 0x7e;
  }

  MD5Hash::MD5Hash(std::string const& s)
  {
    const int hash_size = 16; // in bytes
    if (s.size() != hash_size*2)
      throw edm::Exception(edm::errors::LogicError)
	<< "Illegal conversion: MD5 hexdigest must be 32 bytes long";

    std::string::const_iterator it = s.begin();
    for (int i = 0; i != hash_size; ++i)
      {
	data[i]  = (detail::unhexify(*it++) << 4);
	data[i] +=  detail::unhexify(*it++);
      }
  }

  bool operator<(MD5Hash const& a, MD5Hash const& b) 
  {
    for (int i = 0; i < MD5Hash::size; ++i) 
      {
      if (a.data[i] < b.data[i]) return true;
      if (b.data[i] < a.data[i]) return false;
      }
    return false;
  }
  bool operator==(MD5Hash const& a, MD5Hash const& b) 
  {
    for (int i = 0; i < MD5Hash::size; ++i) 
      {
	if (a.data[i] != b.data[i]) return false;
      }
    return true;
  }
}

