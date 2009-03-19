/*----------------------------------------------------------------------
  
BranchMapperWithReader:

----------------------------------------------------------------------*/
#include "BranchMapperWithReader.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"

namespace edm {
  BranchMapperWithReader::BranchMapperWithReader() :
	 BranchMapper(true),
	 branchPtr_(0),
	 entryNumber_(0),
	 infoVector_(),
	 pInfoVector_(&infoVector_),
	 oldProductIDToBranchIDMap_()
  { }

  BranchMapperWithReader::BranchMapperWithReader(
    TBranch * branch,
    input::EntryNumber entryNumber) :
	 BranchMapper(true),
	 branchPtr_(branch),
	 entryNumber_(entryNumber),
	 infoVector_(),
	 pInfoVector_(&infoVector_),
	 oldProductIDToBranchIDMap_()
  { }

  void
  BranchMapperWithReader::readProvenance_() const {
    setRefCoreStreamer(0, false, false);
    branchPtr_->SetAddress(&pInfoVector_);
    input::getEntry(branchPtr_, entryNumber_);
    setRefCoreStreamer(true);
    BranchMapperWithReader * me = const_cast<BranchMapperWithReader*>(this);
    for (std::vector<ProductProvenance>::const_iterator it = infoVector_.begin(), itEnd = infoVector_.end();
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
