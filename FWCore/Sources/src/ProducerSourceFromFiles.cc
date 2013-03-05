#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  ProducerSourceFromFiles::ProducerSourceFromFiles(ParameterSet const& pset, InputSourceDescription const& desc, bool realData) :
    ProducerSourceBase(pset, desc, realData),
    catalog_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"),
             pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())) {
  }

  ProducerSourceFromFiles::~ProducerSourceFromFiles() {}

  void
  ProducerSourceFromFiles::fillDescription(ParameterSetDescription & desc) {
    std::vector<std::string> defaultStrings;
    desc.addUntracked<std::vector<std::string> >("fileNames", defaultStrings)
        ->setComment("Names of files to be processed.");
    desc.addUntracked<std::string>("overrideCatalog", std::string());
    ProducerSourceBase::fillDescription(desc);
  }

  bool
  ProducerSourceFromFiles::noFiles() const {
    return catalog_.fileCatalogItems().empty();
  }
}


