/*----------------------------------------------------------------------

BranchMapperWithReader:

----------------------------------------------------------------------*/
#include "BranchMapperWithReader.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "RootTree.h"

namespace edm {
  BranchMapperWithReader::BranchMapperWithReader() :
         BranchMapper(true),
         rootTree_(0),
         useCache_(false),
         infoVector_(),
         pInfoVector_(&infoVector_),
         oldProductIDToBranchIDMap_()
  { }

  BranchMapperWithReader::BranchMapperWithReader(RootTree* rootTree, bool useCache) :
         BranchMapper(true),
         rootTree_(rootTree),
         useCache_(useCache),
         infoVector_(),
         pInfoVector_(&infoVector_),
         oldProductIDToBranchIDMap_()
  { }

  void
  BranchMapperWithReader::readProvenance_() const {
    setRefCoreStreamer(0, false, false);
    if (useCache_) {
      rootTree_->fillBranchEntry(rootTree_->branchEntryInfoBranch(), pInfoVector_);
    } else {
      rootTree_->fillBranchEntryNoCache(rootTree_->branchEntryInfoBranch(), pInfoVector_);
    }
    setRefCoreStreamer(true);
    BranchMapperWithReader* me = const_cast<BranchMapperWithReader*>(this);
    for (ProductProvenanceVector::const_iterator it = infoVector_.begin(), itEnd = infoVector_.end();
      it != itEnd; ++it) {
      me->insert(*it);
    }
  }

  void
  BranchMapperWithReader::insertIntoMap(ProductID const& oldProductID, BranchID const& branchID) {
    oldProductIDToBranchIDMap_.insert(std::make_pair(oldProductID.oldID(), branchID));
  }

  BranchID
  BranchMapperWithReader::oldProductIDToBranchID_(ProductID const& oldProductID) const {
    std::map<unsigned int, BranchID>::const_iterator it = oldProductIDToBranchIDMap_.find(oldProductID.oldID());
    if (it == oldProductIDToBranchIDMap_.end()) {
      return BranchID();
    }
    return it->second;
  }
}
