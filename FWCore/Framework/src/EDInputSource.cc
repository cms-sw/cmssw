/*----------------------------------------------------------------------
$Id: EDInputSource.cc,v 1.1 2006/04/06 23:26:29 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  
  EDInputSource::EDInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    catalog_(pset)
  { }

  EDInputSource::~EDInputSource() {
  }

  void
  EDInputSource::setRun(RunNumber_t) {
      throw cms::Exception("LogicError","EDInputSource::setRun()")
        << "Run number cannot be modified for an EDInputSource\n"
        << "Contact a Framework Developer\n";
  }
}
