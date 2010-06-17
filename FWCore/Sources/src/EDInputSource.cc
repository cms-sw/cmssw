/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/EDInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  
  // The default value provided as the second argument to the getUntrackedParameter function call
  // is not used when the ParameterSet has been validated and the parameters are not optional
  // in the description.  As soon as all primary input sources and all modules with a secondary
  // input sources have defined descriptions, the defaults in the getUntrackedParameterSet function
  // calls can and should be deleted from the code.
  EDInputSource::EDInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
      InputSource(pset, desc),
      catalog_(pset, pset.getUntrackedParameter<std::vector<std::string> >("fileNames")) ,
      secondaryCatalog_(pset, pset.getUntrackedParameter<bool>("needSecondaryFileNames", false) ?
	pset.getUntrackedParameter<std::vector<std::string> >("secondaryFileNames") : 
	pset.getUntrackedParameter<std::vector<std::string> >("secondaryFileNames", std::vector<std::string>()))
  {}

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
    InputSource::fillDescription(desc);
  }
}
