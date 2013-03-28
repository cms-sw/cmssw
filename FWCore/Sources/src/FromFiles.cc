#include "FWCore/Sources/interface/FromFiles.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  FromFiles::FromFiles(ParameterSet const& pset) :
    catalog_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"),
             pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())),
    fileIndex_(0) {
  }

  FromFiles::~FromFiles() {}

  void
  FromFiles::fillDescription(ParameterSetDescription & desc) {
    std::vector<std::string> defaultStrings;
    desc.addUntracked<std::vector<std::string> >("fileNames", defaultStrings)
        ->setComment("Names of files to be processed.");
    desc.addUntracked<std::string>("overrideCatalog", std::string());
  }

  size_t
  FromFiles::fileIndex() const {
    return fileIndex_;
  }
}


