#ifndef IOPool_Input_RootDelayedReader_h
#define IOPool_Input_RootDelayedReader_h

/*----------------------------------------------------------------------

RootDelayedReader.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchKey.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "RootTree.h"

#include <map>
#include <memory>
#include <string>
#include <exception>

class TClass;
namespace edm {
  class InputFile;
  class RootTree;
  class SharedResourcesAcquirer;
  class Exception;

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
      std::shared_ptr<InputFile> filePtr,
      InputType inputType);

    virtual ~RootDelayedReader();

    RootDelayedReader(RootDelayedReader const&) = delete; // Disallow copying and moving
    RootDelayedReader& operator=(RootDelayedReader const&) = delete; // Disallow copying and moving

    virtual signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal() const override final {
      return preEventReadFromSourceSignal_;
    }
    virtual signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal() const override final {
      return postEventReadFromSourceSignal_;
    }

    void setSignals(signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
                    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource) {
      preEventReadFromSourceSignal_ =preEventReadSource;
      postEventReadFromSourceSignal_ = postEventReadSource;
    }

  private:
    virtual std::unique_ptr<WrapperBase> getProduct_(BranchKey const& k, EDProductGetter const* ep) override;
    virtual void mergeReaders_(DelayedReader* other) override {nextReader_ = other;}
    virtual void reset_() override {nextReader_ = nullptr;}
    std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> sharedResources_() const override;

    BranchMap const& branches() const {return tree_.branches();}
    iterator branchIter(BranchKey const& k) const {return branches().find(k);}
    bool found(iterator const& iter) const {return iter != branches().end();}
    BranchInfo const& getBranchInfo(iterator const& iter) const {return iter->second; }
    // NOTE: filePtr_ appears to be unused, but is needed to prevent
    // the file containing the branch from being reclaimed.
    RootTree const& tree_;
    edm::propagate_const<std::shared_ptr<InputFile>> filePtr_;
    edm::propagate_const<DelayedReader*> nextReader_;
    std::unique_ptr<SharedResourcesAcquirer> resourceAcquirer_; // We do not use propagate_const because the acquirer is itself mutable.
    std::shared_ptr<std::recursive_mutex> mutex_;
    InputType inputType_;
    edm::propagate_const<TClass*> wrapperBaseTClass_;
    
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal_ = nullptr;
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal_ = nullptr;

    //If a fatal exception happens we need to make a copy so we can
    // rethrow that exception on other threads. This avoids TTree
    // non-exception safety problems on later calls to TTree.
    mutable std::exception_ptr lastException_;
  }; // class RootDelayedReader
  //------------------------------------------------------------
}
#endif
