#ifndef IOPool_Input_BranchMapperWithReader
#define IOPool_Input_BranchMapperWithReader

/*----------------------------------------------------------------------
  
BranchMapperWithReader: The mapping from per event product ID's to BranchID's.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchMapper.h"

#include <vector>
#include "TBranch.h"

class TBranch;
namespace edm {
  template <typename T>
  class BranchMapperWithReader : public BranchMapper {
  public:
    BranchMapperWithReader(TBranch * branch, input::EntryNumber entryNumber);

    virtual ~BranchMapperWithReader() {}

  private:
    virtual void readProvenance_() const;

    TBranch * branchPtr_; 
    input::EntryNumber entryNumber_;
    std::vector<T> infoVector_;
    mutable std::vector<T> * pInfoVector_;
  };
  
  template <typename T>
  BranchMapperWithReader<T>::BranchMapperWithReader(TBranch * branch, input::EntryNumber entryNumber) :
	 BranchMapper(true),
	 branchPtr_(branch), entryNumber_(entryNumber),
	 infoVector_(), pInfoVector_(&infoVector_)
  { }

  template <typename T>
  inline
  void
  BranchMapperWithReader<T>::readProvenance_() const {
    branchPtr_->SetAddress(&pInfoVector_);
    input::getEntry(branchPtr_, entryNumber_);
    BranchMapperWithReader<T> * me = const_cast<BranchMapperWithReader<T> *>(this);
    for (typename std::vector<T>::const_iterator it = infoVector_.begin(), itEnd = infoVector_.end();
      it != itEnd; ++it) {
      me->insert(it->makeEntryInfo());
    }
  }

}
#endif
