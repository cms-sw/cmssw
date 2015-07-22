/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "EmbeddedRootSource.h"
#include "InputFile.h"
#include "RootEmbeddedFileSequence.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceDescription.h"

namespace edm {

  class EventID;
  class EventPrincipal;

  EmbeddedRootSource::EmbeddedRootSource(ParameterSet const& pset, VectorInputSourceDescription const& desc) :
    VectorInputSource(pset, desc),
    rootServiceChecker_(),
    catalog_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"),
      pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())),
    fileSequence_(new RootEmbeddedFileSequence(pset, *this, catalog_, desc.allocations_->numberOfStreams())) {
  }

  EmbeddedRootSource::~EmbeddedRootSource() {}

  void
  EmbeddedRootSource::beginJob() {
  }

  void
  EmbeddedRootSource::endJob() {
    fileSequence_->endJob();
    InputFile::reportReadBranches();
  }

  void EmbeddedRootSource::closeFile_() {
    fileSequence_->closeFile_();
  }

  bool
  EmbeddedRootSource::readOneEvent(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine* engine, EventID const* id) {
    return fileSequence_->readOneEvent(cache, fileNameHash, engine, id);
  }

  void
  EmbeddedRootSource::readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, SecondaryEventIDAndFileInfo const& id) {
    fileSequence_->readOneSpecified(cache, fileNameHash, id);
  }

  void
  EmbeddedRootSource::dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) {
    fileSequence_->dropUnwantedBranches_(wantedBranches);
  }

  void
  EmbeddedRootSource::fillDescriptions(ConfigurationDescriptions& descriptions) {

    ParameterSetDescription desc;

    std::vector<std::string> defaultStrings;
    desc.setComment("Reads EDM/Root files for mixing.");
    desc.addUntracked<std::vector<std::string> >("fileNames")
        ->setComment("Names of files to be processed.");
    desc.addUntracked<std::string>("overrideCatalog", std::string());
    RootEmbeddedFileSequence::fillDescription(desc);

    descriptions.add("source", desc);
  }
}
