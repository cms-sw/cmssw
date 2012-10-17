/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/EDInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  
  // The default value provided as the second argument to the getUntrackedParameter function call
  // is not used when the ParameterSet has been validated and the parameters are not optional
  // in the description.  As soon as all primary input sources and all modules with a secondary
  // input sources have defined descriptions, the defaults in the getUntrackedParameterSet function
  // calls can and should be deleted from the code.
  EDInputSource::EDInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
      InputSource(pset, desc),
      catalog_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"),
        pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())),
      secondaryCatalog_(pset.getUntrackedParameter<std::vector<std::string> >("secondaryFileNames", std::vector<std::string>()),
        pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())) {

     if (secondaryCatalog_.empty() && pset.getUntrackedParameter<bool>("needSecondaryFileNames", false)) {
	throw Exception(errors::Configuration, "EDInputSource") << "'secondaryFileNames' must be specified\n";
     }
  }

  EDInputSource::~EDInputSource() {
  }

  void
  EDInputSource::fillDescription(ParameterSetDescription & desc) {
    std::vector<std::string> defaultStrings;
    desc.addUntracked<std::vector<std::string> >("fileNames")
        ->setComment("Names of files to be processed.");
    desc.addUntracked<std::vector<std::string> >("secondaryFileNames", defaultStrings)
        ->setComment("Names of secondary files to be processed.");
    desc.addUntracked<bool>("needSecondaryFileNames", false)
        ->setComment("If True, 'secondaryFileNames' must be specified and be non-empty.");
    desc.addUntracked<std::string>("overrideCatalog", std::string());
    InputSource::fillDescription(desc);
  }
}
