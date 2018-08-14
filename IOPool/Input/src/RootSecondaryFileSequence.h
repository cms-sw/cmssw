#ifndef IOPool_Input_RootSecondaryFileSequence_h
#define IOPool_Input_RootSecondaryFileSequence_h

/*----------------------------------------------------------------------

RootSecondaryFileSequence: This is an InputSource

----------------------------------------------------------------------*/

#include "RootInputFileSequence.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace edm {

  class BranchID;
  class FileCatalogItem;
  class InputFileCatalog;
  class PoolSource;
  class RootFile;

  class RootSecondaryFileSequence : public RootInputFileSequence {
  public:
    explicit RootSecondaryFileSequence(ParameterSet const& pset,
                                   PoolSource& input,
                                   InputFileCatalog const& catalog);
    ~RootSecondaryFileSequence() override;

    RootSecondaryFileSequence(RootSecondaryFileSequence const&) = delete; // Disallow copying and moving
    RootSecondaryFileSequence& operator=(RootSecondaryFileSequence const&) = delete; // Disallow copying and moving

    void closeFile_() override;
    void endJob();
    void initAssociationsFromSecondary(std::set<BranchID> const&);
  private:
    void initFile_(bool skipBadFiles) override;
    RootFileSharedPtr makeRootFile(std::shared_ptr<InputFile> filePtr) override; 

    PoolSource& input_;
    std::vector<BranchID> associationsFromSecondary_;
    std::vector<ProcessHistoryID> orderedProcessHistoryIDs_;
    bool enablePrefetching_;
  }; // class RootSecondaryFileSequence
}
#endif
