#ifndef IOPool_Input_BranchMapperWithReader
#define IOPool_Input_BranchMapperWithReader

/*----------------------------------------------------------------------
  
BranchMapperWithReader:

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "Inputfwd.h"

#include <vector>

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
    ProductProvenanceVector infoVector_;
    mutable ProductProvenanceVector * pInfoVector_;
    std::map<unsigned int, BranchID> oldProductIDToBranchIDMap_;
  };
}
#endif
