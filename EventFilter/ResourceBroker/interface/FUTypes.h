//
// FU type definitions
//

#include "toolbox/include/toolbox/mem/Reference.h"

#include <vector>
#include <queue>
#include <deque>

namespace evf {
  
  typedef toolbox::mem::Reference MemRef_t;
  
  typedef unsigned int             UInt_t;
  typedef unsigned short           UShort_t;
  typedef unsigned char            UChar_t;
  typedef std::vector<UInt_t>      UIntVec_t;
  typedef std::queue<UInt_t>       UIntQueue_t;
  typedef std::deque<UInt_t>       UIntDeque_t;
  
} // namespace evf
