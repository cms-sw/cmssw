/*----------------------------------------------------------------------
$Id: GenericInputSource.cc,v 1.1 2006/01/18 00:38:44 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/GenericInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  //used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;
  
  GenericInputSource::GenericInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    fileNames_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames")),
    remainingEvents_(maxEvents())
  { }

  GenericInputSource::~GenericInputSource() {
  }

  std::auto_ptr<EventPrincipal>
  GenericInputSource::read() {
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
