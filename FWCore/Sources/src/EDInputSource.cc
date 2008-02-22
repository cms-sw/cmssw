/*----------------------------------------------------------------------
$Id: EDInputSource.cc,v 1.2 2007/06/29 16:32:58 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/EDInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {
  
  EDInputSource::EDInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
      InputSource(pset, desc),
      catalogs_() {
    catalogs_.reserve(2);
    catalogs_.push_back(InputFileCatalog(pset));
    catalogs_.push_back(InputFileCatalog(pset, std::string("secondaryFileNames"), true));
  }

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
