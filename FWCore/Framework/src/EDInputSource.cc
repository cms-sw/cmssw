/*----------------------------------------------------------------------
$Id: EDInputSource.cc,v 1.3 2006/04/24 22:35:03 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
      LogWarning("IllegalCall")
        << "EDInputSource::setRun()\n"
        << "Run number cannot be modified for an EDInputSource\n";
  }
}
