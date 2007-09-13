#ifndef DataFormats_Provenance_Selections_h
#define DataFormats_Provenance_Selections_h

#include "boost/array.hpp"
#include <vector>

#include "DataFormats/Provenance/interface/BranchType.h"

namespace edm {
  class BranchDescription;
  typedef std::vector<BranchDescription const *> Selections;
  typedef boost::array<Selections, NumBranchTypes> SelectionsArray;
}

#endif
