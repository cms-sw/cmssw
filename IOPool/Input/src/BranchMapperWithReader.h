#ifndef IOPool_Input_BranchMapperWithReader
#define IOPool_Input_BranchMapperWithReader

/*----------------------------------------------------------------------

BranchMapperWithReader:

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "Inputfwd.h"

#include <vector>

namespace edm {
  class RootTree;
  class BranchMapperWithReader : public BranchMapper {
  public:
    BranchMapperWithReader();
    BranchMapperWithReader(RootTree* rootTree, bool useCache);

    virtual ~BranchMapperWithReader() {}
    void insertIntoMap(ProductID const& oldProductID, BranchID const& branchID);

  private:
    virtual void readProvenance_() const;
    virtual BranchID oldProductIDToBranchID_(ProductID const& oldProductID) const;

    RootTree* rootTree_;
    bool useCache_;
    ProductProvenanceVector infoVector_;
    mutable ProductProvenanceVector* pInfoVector_;
    std::map<unsigned int, BranchID> oldProductIDToBranchIDMap_;
  };
}
#endif
