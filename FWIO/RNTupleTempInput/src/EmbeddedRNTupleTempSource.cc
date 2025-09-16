/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "EmbeddedRNTupleTempSource.h"
#include "InputFile.h"
#include "RootEmbeddedFileSequence.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceDescription.h"
#include "FWCore/Sources/interface/InputSourceRunHelper.h"

namespace edm {

  class EventID;
  class EventPrincipal;
}  // namespace edm
namespace edm::rntuple_temp {

  EmbeddedRNTupleTempSource::EmbeddedRNTupleTempSource(ParameterSet const& pset,
                                                       VectorInputSourceDescription const& desc)
      : VectorInputSource(pset, desc),
        rootServiceChecker_(),
        nStreams_(desc.allocations_->numberOfStreams()),
        // The default value provided as the second argument to the getUntrackedParameter function call
        // is not used when the ParameterSet has been validated and the parameters are not optional
        // in the description.  This is currently true when RNTupleTempSource is the primary input source.
        // The modules that use RNTupleTempSource as a SecSource have not defined their fillDescriptions function
        // yet, so the ParameterSet does not get validated yet.  As soon as all the modules with a SecSource
        // have defined descriptions, the defaults in the getUntrackedParameterSet function calls can
        // and should be deleted from the code.
        //
        skipBadFiles_(pset.getUntrackedParameter<bool>("skipBadFiles", false)),
        bypassVersionCheck_(pset.getUntrackedParameter<bool>("bypassVersionCheck", false)),
        treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize", -1)),
        productSelectorRules_(pset, "inputCommands", "InputSource"),
        runHelper_(new DefaultInputSourceRunHelper()),
        catalog_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames"),
                 pset.getUntrackedParameter<std::string>("overrideCatalog", std::string())),
        // Note: fileSequence_ needs to be initialized last, because it uses data members
        // initialized previously in its own initialization.
        fileSequence_(new RootEmbeddedFileSequence(pset, *this, catalog_)) {}

  EmbeddedRNTupleTempSource::~EmbeddedRNTupleTempSource() {}

  void EmbeddedRNTupleTempSource::beginJob() {}

  void EmbeddedRNTupleTempSource::endJob() {
    fileSequence_->endJob();
    InputFile::reportReadBranches();
  }

  void EmbeddedRNTupleTempSource::closeFile_() { fileSequence_->closeFile(); }

  bool EmbeddedRNTupleTempSource::readOneEvent(EventPrincipal& cache,
                                               size_t& fileNameHash,
                                               CLHEP::HepRandomEngine* engine,
                                               EventID const* id,
                                               bool recycleFiles) {
    return fileSequence_->readOneEvent(cache, fileNameHash, engine, id, recycleFiles);
  }

  void EmbeddedRNTupleTempSource::readOneSpecified(EventPrincipal& cache,
                                                   size_t& fileNameHash,
                                                   SecondaryEventIDAndFileInfo const& id) {
    fileSequence_->readOneSpecified(cache, fileNameHash, id);
  }

  void EmbeddedRNTupleTempSource::dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) {
    std::vector<std::string> rules;
    rules.reserve(wantedBranches.size() + 1);
    rules.emplace_back("drop *");
    for (std::string const& branch : wantedBranches) {
      rules.push_back("keep " + branch + "_*");
    }
    ParameterSet pset;
    pset.addUntrackedParameter("inputCommands", rules);
    productSelectorRules_ = ProductSelectorRules(pset, "inputCommands", "InputSource");
  }

  void EmbeddedRNTupleTempSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;

    std::vector<std::string> defaultStrings;
    desc.setComment("Reads EDM/Root files for mixing.");
    desc.addUntracked<std::vector<std::string> >("fileNames")->setComment("Names of files to be processed.");
    desc.addUntracked<std::string>("overrideCatalog", std::string());
    desc.addUntracked<bool>("skipBadFiles", false)
        ->setComment(
            "True:  Ignore any missing or unopenable input file.\n"
            "False: Throw exception if missing or unopenable input file.");
    desc.addUntracked<bool>("bypassVersionCheck", false)
        ->setComment(
            "True:  Bypass release version check.\n"
            "False: Throw exception if reading file in a release prior to the release in which the file was written.");
    desc.addUntracked<int>("treeMaxVirtualSize", -1)
        ->setComment("Size of ROOT TTree TBasket cache.  Affects performance.");

    ProductSelectorRules::fillDescription(desc, "inputCommands");
    RootEmbeddedFileSequence::fillDescription(desc);

    descriptions.add("source", desc);
  }
}  // namespace edm::rntuple_temp
