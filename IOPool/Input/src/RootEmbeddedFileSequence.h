#ifndef IOPool_Input_RootEmbeddedFileSequence_h
#define IOPool_Input_RootEmbeddedFileSequence_h

/*----------------------------------------------------------------------

RootEmbeddedFileSequence: This is an InputSource

----------------------------------------------------------------------*/

#include "RootInputFileSequence.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include <memory>
#include <string>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  class BranchID;
  class FileCatalogItem;
  class InputFileCatalog;
  class ParameterSetDescription;
  class EmbeddedRootSource;
  class RootFile;

  class RootEmbeddedFileSequence : public RootInputFileSequence {
  public:
    explicit RootEmbeddedFileSequence(ParameterSet const& pset,
                                   EmbeddedRootSource& input,
                                   InputFileCatalog const& catalog);
    virtual ~RootEmbeddedFileSequence();

    RootEmbeddedFileSequence(RootEmbeddedFileSequence const&) = delete; // Disallow copying and moving
    RootEmbeddedFileSequence& operator=(RootEmbeddedFileSequence const&) = delete; // Disallow copying and moving

    virtual void closeFile_() override;
    void endJob();
    void skipEntries(unsigned int offset);
    bool readOneEvent(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const* id);
    bool readOneRandom(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const*);
    bool readOneRandomWithID(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const* id);
    bool readOneSequential(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const*);
    bool readOneSequentialWithID(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const* id);
    void readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, SecondaryEventIDAndFileInfo const& id);

    static void fillDescription(ParameterSetDescription & desc);
  private:
    virtual void initFile_(bool skipBadFiles) override;
    virtual RootFileSharedPtr makeRootFile(std::shared_ptr<InputFile> filePtr) override; 

    EmbeddedRootSource& input_;

    std::vector<ProcessHistoryID> orderedProcessHistoryIDs_;

    bool sequential_;
    bool sameLumiBlock_;
    bool (RootEmbeddedFileSequence::* fptr_)(EventPrincipal&, size_t&, CLHEP::HepRandomEngine*, EventID const*);
    int eventsRemainingInFile_;
    int initialNumberOfEventsToSkip_;
    unsigned int treeCacheSize_;
    bool enablePrefetching_;
  }; // class RootEmbeddedFileSequence
}
#endif
