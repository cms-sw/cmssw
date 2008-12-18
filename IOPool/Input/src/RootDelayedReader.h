#ifndef IOPool_Input_RootDelayedReader_h
#define IOPool_Input_RootDelayedReader_h

/*----------------------------------------------------------------------

RootDelayedReader.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include <memory>
#include <map>
#include <string>
#include "boost/utility.hpp"
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/BranchKey.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "Inputfwd.h"

class TFile;
namespace edm {

  //------------------------------------------------------------
  // Class RootDelayedReader: pretends to support file reading.
  //

  class RootDelayedReader : public DelayedReader, private boost::noncopyable {
  public:
    typedef input::BranchMap BranchMap;
    typedef input::EntryNumber EntryNumber;
    typedef input::BranchInfo BranchInfo;
    typedef input::BranchMap::const_iterator iterator;
    RootDelayedReader(EntryNumber const& entry,
      boost::shared_ptr<BranchMap const> bMap,
      boost::shared_ptr<TFile const> filePtr,
      bool oldFormat);

    virtual ~RootDelayedReader();

  private:
    virtual std::auto_ptr<EDProduct> getProduct_(BranchKey const& k, EDProductGetter const* ep) const;
    virtual void mergeReaders_(boost::shared_ptr<DelayedReader> other) {nextReader_ = other;}
    BranchMap const& branches() const {return *branches_;}
    iterator branchIter(BranchKey const& k) const {return branches().find(k);}
    bool found(iterator const& iter) const {return iter != branches().end();}
    BranchInfo const& getBranchInfo(iterator const& iter) const {return iter->second; }
    EntryNumber const entryNumber_;
    boost::shared_ptr<BranchMap const> branches_;
    // NOTE: filePtr_ appears to be unused, but is needed to prevent
    // the TFile containing the branch from being reclaimed.
    boost::shared_ptr<TFile const> filePtr_;
    boost::shared_ptr<DelayedReader> nextReader_;
    bool customStreamers_;
    bool oldFormat_;
  }; // class RootDelayedReader
  //------------------------------------------------------------
}
#endif
