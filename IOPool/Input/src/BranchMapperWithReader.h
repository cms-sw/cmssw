#ifndef IOPool_Input_BranchMapperWithReader
#define IOPool_Input_BranchMapperWithReader

/*----------------------------------------------------------------------

BranchMapperWithReader:

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchMapper.h"

#include <vector>

namespace edm {
  class RootTree;
  class BranchMapperWithReader : public BranchMapper {
  public:
    BranchMapperWithReader();
    explicit BranchMapperWithReader(RootTree* rootTree);

    virtual ~BranchMapperWithReader() {}
    void insertIntoMap(ProductID const& oldProductID, BranchID const& branchID);

  private:
    virtual void readProvenance_() const;
    virtual BranchID oldProductIDToBranchID_(ProductID const& oldProductID) const;
    virtual void reset_();

    RootTree* rootTree_;
    ProductProvenanceVector infoVector_;
    mutable ProductProvenanceVector* pInfoVector_;
    std::map<unsigned int, BranchID> oldProductIDToBranchIDMap_;
  };
}
#endif
