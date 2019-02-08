#include "FWCore/Utilities/interface/RunIndex.h"

#include <limits>
int function_with_warning(){
  int i; int j;
  return i + j;
}

namespace edm {

  const unsigned int RunIndex::invalidValue_ = std::numeric_limits<unsigned int>::max();    

  RunIndex RunIndex::invalidRunIndex() {
    return RunIndex(invalidValue_);
  }
}
