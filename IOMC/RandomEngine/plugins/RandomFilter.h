// -*- C++ -*-
//
// Package:    IOMC/RandomEngine
// Class:      RandomFilter
//
/**\class edm::RandomFilter

 Description: The output of this module is used for test purposes.
It is a filter module that makes a filter decision based on a
randomly generated number.  The fraction of events that pass the
filter (in the limit of infinite statistics) is a parameter
that must be set in the configuration file.  The parameter
type and name is "untracked double acceptRate".

*/
//
// Original Author:  W. David Dagenhart
//         Created:  26 March 2007
//

#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

namespace edm {

  class RandomFilter : public edm::global::EDFilter<> {
  public:
    explicit RandomFilter(edm::ParameterSet const&);
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  private:
    // value between 0 and 1
    double acceptRate_;
  };
}  // namespace edm
