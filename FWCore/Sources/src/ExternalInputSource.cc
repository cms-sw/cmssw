#include "FWCore/Sources/interface/ExternalInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  ExternalInputSource::ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc, bool realData) :
    ConfigurableInputSource(pset, desc, realData),
    catalog_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"),
             pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())) {
  }

  ExternalInputSource::~ExternalInputSource() {}

  void
   ExternalInputSource::fillDescription(ParameterSetDescription & desc) {
    std::vector<std::string> defaultStrings;
    desc.addUntracked<std::vector<std::string> >("fileNames", defaultStrings)
        ->setComment("Names of files to be processed.");
    desc.addUntracked<std::string>("overrideCatalog", std::string());
    ConfigurableInputSource::fillDescription(desc);
  }
}


