#include "DataFormats/Common/interface/MD5Hash.h"

/*----------------------------------------------------------------------

$Id: MD5Hash.cc,v 1.1.2.2 2006/06/27 02:17:50 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  bool operator<(MD5Hash const& a, MD5Hash const& b) {
    for (int i = 0; i < MD5Hash::size; ++i) {
      if (a.value_[i] < b.value_[i]) return true;
      if (b.value_[i] < a.value_[i]) return false;
    }
    return false;
  }
  bool operator==(MD5Hash const& a, MD5Hash const& b) {
    for (int i = 0; i < MD5Hash::size; ++i) {
      if (a.value_[i] != b.value_[i]) return false;
    }
    return true;
  }
}

