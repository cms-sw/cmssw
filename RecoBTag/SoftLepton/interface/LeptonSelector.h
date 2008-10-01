#ifndef RecoBTag_SoftLepton_LeptonSelector_h
#define RecoBTag_SoftLepton_LeptonSelector_h

#include <string>

#include "FWCore/Utilities/interface/EDMException.h"

namespace btag {

namespace LeptonSelector {

/// optionally select leptons based on their impact parameter sign
enum sign {
  negative = -1,
  any      =  0,
  positive =  1
};

inline sign option(const std::string & selection) {
  if (selection == "any")
    return any;
  else if (selection == "negative")
    return negative;
  else if (selection == "positive")
    return positive;
  else 
    throw edm::Exception( edm::errors::Configuration ) << "invalid parameter specified for soft lepton selection";
}

} // namespace LeptonSelector

} // namespace btag

#endif // RecoBTag_SoftLepton_LeptonSelector_h
