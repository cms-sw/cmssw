/*----------------------------------------------------------------------
$Id: EmptySource.cc,v 1.1 2005/09/07 19:09:26 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/NoDelayedReader.h"

namespace edm {
  class BranchKey;
  NoDelayedReader::~NoDelayedReader() {}

  std::auto_ptr<EDProduct>
  NoDelayedReader::get(BranchKey const& k) const {
    throw cms::Exception("LogicError","NoDelayedReader")
      << "get() called for branchkey: " << k << "\n";
  }
}
