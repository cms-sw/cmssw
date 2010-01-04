/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/EDInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  
  EDInputSource::EDInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
      InputSource(pset, desc),
      poolCatalog_(),
      catalog_(pset, poolCatalog_),
      secondaryCatalog_(pset, poolCatalog_,
        std::string("secondaryFileNames"),
        // The default value provided as the second argument to the getUntrackedParameter function call
        // is not used when the ParameterSet has been validated and the parameters are not optional
        // in the description.  This is currently true when PoolSource is the primary input source.
        // The modules that use PoolSource as a SecSource have not defined their fillDescriptions function
        // yet, so the ParameterSet does not get validated yet.  As soon as all the modules with a SecSource
        // have defined descriptions, the defaults in the getUntrackedParameterSet function calls can
        // and should be deleted from the code.
        !pset.getUntrackedParameter<bool>("needSecondaryFileNames", false)) {}

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

  void
  EDInputSource::fillDescription(ParameterSetDescription & desc) {
    std::vector<std::string> defaultStrings;
    desc.addUntracked<std::vector<std::string> >("fileNames", defaultStrings);
    desc.addUntracked<std::vector<std::string> >("secondaryFileNames", defaultStrings);
    desc.addUntracked<bool>("needSecondaryFileNames", false);
    InputFileCatalog::fillDescription(desc);
    InputSource::fillDescription(desc);
  }
}
