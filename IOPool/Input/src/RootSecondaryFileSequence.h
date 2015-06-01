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
  class ParameterSetDescription;
  class PoolSource;
  class RootFile;

  class RootSecondaryFileSequence : public RootInputFileSequence {
  public:
    explicit RootSecondaryFileSequence(ParameterSet const& pset,
                                   PoolSource& input,
                                   InputFileCatalog const& catalog,
                                   unsigned int nStreams);
    virtual ~RootSecondaryFileSequence();

    RootSecondaryFileSequence(RootSecondaryFileSequence const&) = delete; // Disallow copying and moving
    RootSecondaryFileSequence& operator=(RootSecondaryFileSequence const&) = delete; // Disallow copying and moving

    typedef std::shared_ptr<RootFile> RootFileSharedPtr;
    virtual void closeFile_() override;
    void endJob();
    static void fillDescription(ParameterSetDescription & desc);
    void initAssociationsFromSecondary(std::set<BranchID> const&);
  private:
    virtual void initFile_(bool skipBadFiles) override;
    virtual RootFileSharedPtr makeRootFile(std::shared_ptr<InputFile> filePtr) override; 

    PoolSource& input_;
    bool firstFile_;
    std::vector<BranchID> associationsFromSecondary_;
    std::vector<ProcessHistoryID> orderedProcessHistoryIDs_;

    unsigned int nStreams_; 
    bool skipBadFiles_;
    bool bypassVersionCheck_;
    int const treeMaxVirtualSize_;
    RunNumber_t setRun_;
    ProductSelectorRules productSelectorRules_;
    bool dropDescendants_;
    bool labelRawDataLikeMC_;
    bool enablePrefetching_;
  }; // class RootSecondaryFileSequence
}
#endif
