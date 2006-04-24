/*----------------------------------------------------------------------
$Id: EDInputSource.cc,v 1.2 2006/04/13 22:24:08 wmtan Exp $
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
      throw edm::Exception(edm::errors::LogicError)
        << "EDInputSource::setRun()\n"
        << "Run number cannot be modified for an EDInputSource\n"
        << "Contact a Framework Developer\n";
  }
}
