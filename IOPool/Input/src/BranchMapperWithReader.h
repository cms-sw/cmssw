#ifndef IOPool_Input_BranchMapperWithReader
#define IOPool_Input_BranchMapperWithReader

/*----------------------------------------------------------------------

BranchMapperWithReader:

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchMapper.h"

#include <vector>

namespace edm {
  class RootTree;
  class BranchMapperWithReader : public ProvenanceReaderBase {
  public:
    BranchMapperWithReader();
    explicit BranchMapperWithReader(RootTree* rootTree);

    virtual ~BranchMapperWithReader() {}

  private:
    virtual void readProvenance(BranchMapper const& mapper) const;

    RootTree* rootTree_;
    ProductProvenanceVector infoVector_;
    mutable ProductProvenanceVector* pInfoVector_;
  };
}
#endif
