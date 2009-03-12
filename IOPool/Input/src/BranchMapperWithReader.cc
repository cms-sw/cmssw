/*----------------------------------------------------------------------
  
BranchMapperWithReader:

----------------------------------------------------------------------*/
#include "BranchMapperWithReader.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/Parentage.h"

namespace edm {
  void
  BranchMapperWithReader<EventEntryInfo>::readProvenance_() const {
    setRefCoreStreamer(0, fileFormatVersion_.value_ < 11, fileFormatVersion_.value_ < 2);
    branchPtr_->SetAddress(&pInfoVector_);
    input::getEntry(branchPtr_, entryNumber_);
    setRefCoreStreamer(true);
    BranchMapperWithReader<EventEntryInfo> * me = const_cast<BranchMapperWithReader<EventEntryInfo> *>(this);
    for (std::vector<EventEntryInfo>::const_iterator it = infoVector_.begin(), itEnd = infoVector_.end();
      it != itEnd; ++it) {
      EventEntryDescription eed;
      EntryDescriptionRegistry::instance()->getMapped(it->entryDescriptionID(), eed);
      Parentage parentage(eed.parents());
      me->insert(it->makeProductProvenance(parentage.id()));
      me->insertIntoMap(it->productID(), it->branchID());
    }
  }

  void
  BranchMapperWithReader<EventEntryInfo>::insertIntoMap(ProductID const& oldProductID, BranchID const& branchID) {
    oldProductIDToBranchIDMap_.insert(std::make_pair(oldProductID.oldID(), branchID));
  }

  BranchID
  BranchMapperWithReader<EventEntryInfo>::oldProductIDToBranchID_(ProductID const& oldProductID) const {
    std::map<unsigned int, BranchID>::const_iterator it = oldProductIDToBranchIDMap_.find(oldProductID.oldID());    
    if (it == oldProductIDToBranchIDMap_.end()) {
      return BranchID();
    }
    return it->second;
  }
}
