/*----------------------------------------------------------------------
$Id: EDInputSource.cc,v 1.6 2006/04/04 22:15:22 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  //used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;
  
  EDInputSource::EDInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    catalog_(pset),
    remainingEvents_(maxEvents())
  { }

  EDInputSource::~EDInputSource() {
  }

  std::auto_ptr<EventPrincipal>
  EDInputSource::read() {
    std::auto_ptr<EventPrincipal> result(0);
    
    if (remainingEvents_ != 0) {
      result = readOneEvent();
      if (result.get() != 0) {
        --remainingEvents_;
      }
    }
    return result;
  }

}
