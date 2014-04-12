#ifndef IOPool_Input_RootDelayedReader_h
#define IOPool_Input_RootDelayedReader_h

/*----------------------------------------------------------------------

RootDelayedReader.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchKey.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "RootTree.h"

#include <map>
#include <memory>
#include <string>

namespace edm {
  class InputFile;
  class RootTree;

  //------------------------------------------------------------
  // Class RootDelayedReader: pretends to support file reading.
  //

  class RootDelayedReader : public DelayedReader {
  public:
    typedef roottree::BranchInfo BranchInfo;
    typedef roottree::BranchMap BranchMap;
    typedef roottree::BranchMap::const_iterator iterator;
    typedef roottree::EntryNumber EntryNumber;
    RootDelayedReader(
      RootTree const& tree,
      boost::shared_ptr<InputFile> filePtr,
      InputType inputType);

    virtual ~RootDelayedReader();

    RootDelayedReader(RootDelayedReader const&) = delete; // Disallow copying and moving
    RootDelayedReader& operator=(RootDelayedReader const&) = delete; // Disallow copying and moving

  private:
    virtual WrapperOwningHolder getProduct_(BranchKey const& k, 
                                            WrapperInterfaceBase const* interface,
                                            EDProductGetter const* ep) const override;
    virtual void mergeReaders_(DelayedReader* other) {nextReader_ = other;}
    virtual void reset_() {nextReader_ = 0;}
    BranchMap const& branches() const {return tree_.branches();}
    iterator branchIter(BranchKey const& k) const {return branches().find(k);}
    bool found(iterator const& iter) const {return iter != branches().end();}
    BranchInfo const& getBranchInfo(iterator const& iter) const {return iter->second; }
    // NOTE: filePtr_ appears to be unused, but is needed to prevent
    // the file containing the branch from being reclaimed.
    RootTree const& tree_;
    boost::shared_ptr<InputFile> filePtr_;
    DelayedReader* nextReader_;
    InputType inputType_;
  }; // class RootDelayedReader
  //------------------------------------------------------------
}
#endif
