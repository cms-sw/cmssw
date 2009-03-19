#ifndef IOPool_Input_BranchMapperWithReader
#define IOPool_Input_BranchMapperWithReader

/*----------------------------------------------------------------------
  
BranchMapperWithReader:

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "Inputfwd.h"

#include <vector>
#include "TBranch.h"

class TBranch;
namespace edm {
  class BranchMapperWithReader : public BranchMapper {
  public:
    BranchMapperWithReader();
    BranchMapperWithReader(TBranch * branch,
                           input::EntryNumber entryNumber);

    virtual ~BranchMapperWithReader() {}
    void insertIntoMap(ProductID const& oldProductID, BranchID const& branchID);

  private:
    virtual void readProvenance_() const;
    virtual BranchID oldProductIDToBranchID_(ProductID const& oldProductID) const;

    TBranch * branchPtr_; 
    input::EntryNumber entryNumber_;
    std::vector<ProductProvenance> infoVector_;
    mutable std::vector<ProductProvenance> * pInfoVector_;
    std::map<unsigned int, BranchID> oldProductIDToBranchIDMap_;
  };
}
#endif
