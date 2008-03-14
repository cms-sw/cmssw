/*----------------------------------------------------------------------
$Id: EDInputSource.cc,v 1.3 2008/02/22 19:09:37 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/EDInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {
  
  EDInputSource::EDInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
      InputSource(pset, desc),
      poolCatalog_(),
      catalog_(pset, poolCatalog_),
      secondaryCatalog_(pset, poolCatalog_, std::string("secondaryFileNames"), true) {}

  EDInputSource::~EDInputSource() {
  }

  void
  EDInputSource::setRun(RunNumber_t) {
      LogWarning("IllegalCall")
        << "EDInputSource::setRun()\n"
        << "Run number cannot be modified for an EDInputSource\n";
  }

  void
  EDInputSource::setLumi(LuminosityBlockNumber_t) {
      LogWarning("IllegalCall")
        << "EDInputSource::setLumi()\n"
        << "Luminosity Block ID cannot be modified for an EDInputSource\n";
  }
}
