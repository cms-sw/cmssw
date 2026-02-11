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
// line chnaged
namespace edm {

  class BranchID;
  class FileCatalogItem;
  class InputFileCatalog;
  class RNTupleTempSource;
  class RootFile;
}  // namespace edm
namespace edm::rntuple_temp {

  class RootSecondaryFileSequence : public RootInputFileSequence {
  public:
    explicit RootSecondaryFileSequence(ParameterSet const& pset,
                                       RNTupleTempSource& input,
                                       InputFileCatalog const& catalog);
    ~RootSecondaryFileSequence() override;

    RootSecondaryFileSequence(RootSecondaryFileSequence const&) = delete;             // Disallow copying and moving
    RootSecondaryFileSequence& operator=(RootSecondaryFileSequence const&) = delete;  // Disallow copying and moving

    void endJob();
    void initAssociationsFromSecondary(std::set<BranchID> const&);

  private:
    void closeFile_() override;
    void initFile_(bool skipBadFiles) override;
    RootFileSharedPtr makeRootFile(std::shared_ptr<InputFile> filePtr) override;

    RNTupleTempSource& input_;
    std::vector<BranchID> associationsFromSecondary_;
    std::vector<ProcessHistoryID> orderedProcessHistoryIDs_;
    bool enablePrefetching_;
    bool enforceGUIDInFileName_;
  };  // class RootSecondaryFileSequence
}  // namespace edm::rntuple_temp
#endif
