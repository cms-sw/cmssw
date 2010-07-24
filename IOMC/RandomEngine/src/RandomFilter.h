// -*- C++ -*-
//
// Package:    RandomEngine
// Class:      RandomFilter
// 
/**\class RandomFilter RandomFilter.h IOMC/RandomEngine/src/RandomFilter.h

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

#include "FWCore/Framework/interface/EDFilter.h"

namespace CLHEP {
  class RandFlat;
}

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;

  class RandomFilter : public edm::EDFilter {
  public:
    explicit RandomFilter(edm::ParameterSet const& ps);
    virtual ~RandomFilter();

    virtual bool filter(edm::Event& e, edm::EventSetup const& c);

  private:

    // value between 0 and 1
    double acceptRate_;

    boost::shared_ptr<CLHEP::RandFlat> flatDistribution_;
  };
}
