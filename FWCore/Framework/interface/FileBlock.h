#ifndef FWCore_Framework_FileBlock_h
#define FWCore_Framework_FileBlock_h

/*----------------------------------------------------------------------

FileBlock: Properties of an input file.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/ProductDependencies.h"
#include "FWCore/Utilities/interface/BranchType.h"

#include <memory>
#include <array>
#include <string>
#include <utility>

namespace edm {
  /**
   * The FileBlock is intended to hold the information that is
   * independent of the IO technology (such as TTree or RNTuple). The
   * IO-technology-specific information is intended to be held by a
   * class that inherits from the FileBlock. The Source creates the
   * specific FileBlock object that gets passed around by the
   * framework via a reference to the base class, and to the
   * OutputModule that can then dynamic_cast to the specific FileBlock
   * type.
   *
   * Given the need of dynamic_cast this is a bad use of dynamic
   * polymorphism, but compared to the alternatives such as type
   * erasure it seemed the least bad option.
   */
  class FileBlock {
  public:
    // bit mask for reasons fast cloning can be disabled or not applicable
    enum WhyNotFastClonable {
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

    FileBlock()
        : fileFormatVersion_(),
          whyNotFastClonable_(NoRootInputSource),
          hasNewlyDroppedBranch_(),
          fileName_(),
          branchListIndexesUnchanged_(false),
          modifiedIDs_(false),
          productDependencies_(new ProductDependencies) {}

    FileBlock(FileFormatVersion const& version,
              int whyNotFastClonable,
              std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch,
              std::string const& fileName,
              bool branchListIndexesUnchanged,
              bool modifiedIDs,
              std::shared_ptr<ProductDependencies const> productDependencies)
        : fileFormatVersion_(version),
          whyNotFastClonable_(whyNotFastClonable),
          hasNewlyDroppedBranch_(hasNewlyDroppedBranch),
          fileName_(fileName),
          branchListIndexesUnchanged_(branchListIndexesUnchanged),
          modifiedIDs_(modifiedIDs),
          productDependencies_(std::move(productDependencies)) {}

    FileBlock(FileBlock const&) = delete;
    FileBlock& operator=(FileBlock const&) = delete;
    FileBlock(FileBlock&&) = delete;
    FileBlock& operator=(FileBlock&&) = delete;
    virtual ~FileBlock();

    FileFormatVersion const& fileFormatVersion() const { return fileFormatVersion_; }
    int whyNotFastClonable() const { return whyNotFastClonable_; }
    std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch() const { return hasNewlyDroppedBranch_; }
    std::string const& fileName() const { return fileName_; }
    bool branchListIndexesUnchanged() const { return branchListIndexesUnchanged_; }
    bool modifiedIDs() const { return modifiedIDs_; }

    void setNotFastClonable(WhyNotFastClonable const& why) { whyNotFastClonable_ |= why; }
    ProductDependencies const& productDependencies() const { return *productDependencies_; }
    virtual void close();

  private:
    FileFormatVersion fileFormatVersion_;
    int whyNotFastClonable_;
    std::array<bool, NumBranchTypes> hasNewlyDroppedBranch_;
    std::string fileName_;
    bool branchListIndexesUnchanged_;
    bool modifiedIDs_;
    std::shared_ptr<ProductDependencies const> productDependencies_;
  };
}  // namespace edm
#endif
