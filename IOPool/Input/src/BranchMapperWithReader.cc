/*----------------------------------------------------------------------

BranchMapperWithReader:

----------------------------------------------------------------------*/
#include "BranchMapperWithReader.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "RootTree.h"

namespace edm {
  BranchMapperWithReader::BranchMapperWithReader() :
         ProvenanceReaderBase(),
         rootTree_(0),
         infoVector_(),
         pInfoVector_(&infoVector_) {
  }

  BranchMapperWithReader::BranchMapperWithReader(RootTree* rootTree) :
         ProvenanceReaderBase(),
         rootTree_(rootTree),
         infoVector_(),
         pInfoVector_(&infoVector_) {
  }

  void
  BranchMapperWithReader::readProvenance(BranchMapper const& mapper) const {
    rootTree_->fillBranchEntryMeta(rootTree_->branchEntryInfoBranch(), pInfoVector_);
    setRefCoreStreamer(true);
    for(ProductProvenanceVector::const_iterator it = infoVector_.begin(), itEnd = infoVector_.end();
        it != itEnd; ++it) {
      mapper.insertIntoSet(*it);
    }
  }
}
