#ifndef Input_RootDelayedReader_h
#define Input_RootDelayedReader_h

/*----------------------------------------------------------------------

RootDelayedReader.h // used by ROOT input sources

$Id: RootDelayedReader.h,v 1.1 2006/01/06 02:35:45 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <map>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "IOPool/Input/src/Inputfwd.h"
#include "TBranch.h"

namespace edm {

  //------------------------------------------------------------
  // Class RootDelayedReader: pretends to support file reading.
  //

  class RootDelayedReader : public DelayedReader {
  public:
    typedef input::BranchMap BranchMap;
    typedef input::EntryNumber EntryNumber;
    RootDelayedReader(EntryNumber const& entry, BranchMap const& bMap) : entryNumber_(entry), branches_(&bMap) {}
    virtual ~RootDelayedReader();
  private:
    RootDelayedReader(RootDelayedReader const&); // disable copy construction
    RootDelayedReader & operator=(RootDelayedReader const&); // disable assignment
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k, EDProductGetter const* ep) const;
    BranchMap const& branches() const {return *branches_;}
    EntryNumber const entryNumber_;
    BranchMap const* branches_;
  }; // class RootDelayedReader
  //------------------------------------------------------------
}
#endif
