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
  template <typename T>
  class BranchMapperWithReader : public BranchMapper {
  public:
    BranchMapperWithReader(TBranch * branch,
                           input::EntryNumber entryNumber,
                           FileFormatVersion const& fileFormatVersion);

    virtual ~BranchMapperWithReader() {}

  private:
    virtual void readProvenance_() const;

    TBranch * branchPtr_; 
    input::EntryNumber entryNumber_;
    FileFormatVersion fileFormatVersion_;
    std::vector<T> infoVector_;
    mutable std::vector<T> * pInfoVector_;
  };
  
  template <typename T>
  BranchMapperWithReader<T>::BranchMapperWithReader(
    TBranch * branch,
    input::EntryNumber entryNumber,
    FileFormatVersion const& fileFormatVersion) :
	 BranchMapper(true),
	 branchPtr_(branch),
	 entryNumber_(entryNumber),
	 fileFormatVersion_(fileFormatVersion),
	 infoVector_(),
	 pInfoVector_(&infoVector_)
  { }

  template <typename T>
  void
  BranchMapperWithReader<T>::readProvenance_() const {
    branchPtr_->SetAddress(&pInfoVector_);
    input::getEntry(branchPtr_, entryNumber_);
    BranchMapperWithReader<T> * me = const_cast<BranchMapperWithReader<T> *>(this);
    for (typename std::vector<T>::const_iterator it = infoVector_.begin(), itEnd = infoVector_.end();
      it != itEnd; ++it) {
      me->insert(it->makeProductProvenance());
    }
  }

  // Backward compatibility
  template <>
  class BranchMapperWithReader<EventEntryInfo> : public BranchMapper {
  public:
    BranchMapperWithReader(
      TBranch * branch,
      input::EntryNumber entryNumber,
      FileFormatVersion const& fileFormatVersion) :
	 BranchMapper(true),
	 branchPtr_(branch),
         entryNumber_(entryNumber),
	 fileFormatVersion_(fileFormatVersion),
	 infoVector_(), pInfoVector_(&infoVector_), oldProductIDToBranchIDMap_()
  { }

    virtual ~BranchMapperWithReader() {}

    void insertIntoMap(ProductID const& oldProductID, BranchID const& branchID);

  private:
    virtual void readProvenance_() const;
    virtual BranchID oldProductIDToBranchID_(ProductID const& oldProductID) const;

    TBranch * branchPtr_; 
    input::EntryNumber entryNumber_;
    FileFormatVersion fileFormatVersion_;
    std::vector<EventEntryInfo> infoVector_;
    mutable std::vector<EventEntryInfo> * pInfoVector_;
    std::map<unsigned int, BranchID> oldProductIDToBranchIDMap_;
  };
}
#endif
