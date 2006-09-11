#ifndef Common_MD5Hash_h
#define Common_MD5Hash_h

#include "DataFormats/Common/interface/Hash.h"
#include "FWCore/Utilities/interface/EDMException.h"
/*----------------------------------------------------------------------
  
MD5Hash:

$Id: MD5Hash.h,v 1.2 2006/07/06 18:34:05 wmtan Exp $
----------------------------------------------------------------------*/
namespace edm {
  struct MD5Hash {
    static int const size = 16;

    // Default c'tor creates a hash that is the same as that of an empty
    // string.
    MD5Hash();

    // The argument is expected to be the 32-byte 'hexified' version
    // of the 16-byte hash data.
    explicit MD5Hash(std::string const& s);

    unsigned char data[size];
  };

  bool operator<(MD5Hash const& a, MD5Hash const& b);

  bool operator==(MD5Hash const& a, MD5Hash const& b);

  inline
  bool operator!=(MD5Hash const& a, MD5Hash const& b) 
  {
    return !(a == b);
  }
}
#endif
