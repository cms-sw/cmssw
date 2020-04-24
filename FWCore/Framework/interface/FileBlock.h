#ifndef FWCore_Framework_FileBlock_h
#define FWCore_Framework_FileBlock_h

/*----------------------------------------------------------------------

FileBlock: Properties of an input file.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "FWCore/Utilities/interface/BranchType.h"
class TTree;
#include <memory>
#include <array>
#include <string>

namespace edm {
  class BranchDescription;
  class FileBlock {
  public:
    // bit mask for reasons fast cloning can be disabled or not applicable
    enum
    WhyNotFastClonable {
      CanFastClone = 0x0,

      // For entire job
      NoRootInputSource = 0x1,
      ParallelProcesses = (NoRootInputSource << 1),
      NotProcessingEvents = (ParallelProcesses << 1),
      HasSecondaryFileSequence = (NotProcessingEvents << 1),

      // For a given input file
      FileTooOld = (HasSecondaryFileSequence << 1),
      NoEventsInFile = (FileTooOld << 1),
      EventsToBeSorted = (NoEventsInFile << 1),
      RunOrLumiNotContiguous = (EventsToBeSorted << 1),
      EventsOrLumisSelectedByID = (RunOrLumiNotContiguous << 1),
      InitialEventsSkipped = (EventsOrLumisSelectedByID << 1),
      MaxEventsTooSmall = (InitialEventsSkipped << 1),
      MaxLumisTooSmall = (MaxEventsTooSmall << 1),
      RunNumberModified = (MaxLumisTooSmall << 1),
      DuplicateEventsRemoved = (RunNumberModified << 1),

      // The remainder of these are defined here for convenience,
      // but never set in FileBlock, because they are output module specific.

      // For a given output module
      DisabledInConfigFile = (DuplicateEventsRemoved << 1),
      EventSelectionUsed = (DisabledInConfigFile << 1),

      // For given input and output files
      OutputMaxEventsTooSmall = (EventSelectionUsed << 1),
      SplitLevelMismatch = (OutputMaxEventsTooSmall << 1),
      BranchMismatch = (SplitLevelMismatch << 1)
    };

    FileBlock() :
      fileFormatVersion_(),
      tree_(nullptr), metaTree_(nullptr),
      lumiTree_(nullptr), lumiMetaTree_(nullptr),
      runTree_(nullptr), runMetaTree_(nullptr),
      whyNotFastClonable_(NoRootInputSource),
      hasNewlyDroppedBranch_(),
      fileName_(),
      branchListIndexesUnchanged_(false),
      modifiedIDs_(false),
      branchChildren_(new BranchChildren) {}

    FileBlock(FileFormatVersion const& version,
              TTree const* ev, TTree const* meta,
              TTree const* lumi, TTree const* lumiMeta,
              TTree const* run, TTree const* runMeta,
              int whyNotFastClonable,
              std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch,
              std::string const& fileName,
              bool branchListIndexesUnchanged,
              bool modifiedIDs,
              std::shared_ptr<BranchChildren const> branchChildren) :
      fileFormatVersion_(version),
      tree_(const_cast<TTree*>(ev)),
      metaTree_(const_cast<TTree*>(meta)),
      lumiTree_(const_cast<TTree*>(lumi)),
      lumiMetaTree_(const_cast<TTree*>(lumiMeta)),
      runTree_(const_cast<TTree*>(run)),
      runMetaTree_(const_cast<TTree*>(runMeta)),
      whyNotFastClonable_(whyNotFastClonable),
      hasNewlyDroppedBranch_(hasNewlyDroppedBranch),
      fileName_(fileName),
      branchListIndexesUnchanged_(branchListIndexesUnchanged),
      modifiedIDs_(modifiedIDs),
      branchChildren_(branchChildren) {}

    ~FileBlock() {}

    FileFormatVersion const& fileFormatVersion() const {return fileFormatVersion_;}
    TTree* tree() const {return tree_;}
    TTree* metaTree() const {return metaTree_;}
    TTree* lumiTree() const {return lumiTree_;}
    TTree* lumiMetaTree() const {return lumiMetaTree_;}
    TTree* runTree() const {return runTree_;}
    TTree* runMetaTree() const {return runMetaTree_;}

    int whyNotFastClonable() const {return whyNotFastClonable_;}
    std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch() const {return hasNewlyDroppedBranch_;}
    std::string const& fileName() const {return fileName_;}
    bool branchListIndexesUnchanged() const {return branchListIndexesUnchanged_;}
    bool modifiedIDs() const {return modifiedIDs_;}

    void setNotFastClonable(WhyNotFastClonable const& why) {
      whyNotFastClonable_ |= why;
    }
    BranchChildren const& branchChildren() const { return *branchChildren_; }
    void close () {runMetaTree_ = lumiMetaTree_ = metaTree_ = runTree_ = lumiTree_ = tree_ = nullptr;}

  private:
    FileFormatVersion fileFormatVersion_;
    // We use bare pointers because ROOT owns these.
    TTree* tree_;
    TTree* metaTree_;
    TTree* lumiTree_;
    TTree* lumiMetaTree_;
    TTree* runTree_;
    TTree* runMetaTree_;
    int whyNotFastClonable_;
    std::array<bool, NumBranchTypes> hasNewlyDroppedBranch_;
    std::string fileName_;
    bool branchListIndexesUnchanged_;
    bool modifiedIDs_;
    std::shared_ptr<BranchChildren const> branchChildren_;
  };
}
#endif
