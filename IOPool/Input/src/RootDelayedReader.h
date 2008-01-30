#ifndef IOPool_Input_RootDelayedReader_h
#define IOPool_Input_RootDelayedReader_h

/*----------------------------------------------------------------------

RootDelayedReader.h // used by ROOT input sources

$Id: RootDelayedReader.h,v 1.13 2007/09/18 23:22:37 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <map>
#include <string>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "Inputfwd.h"

class TFile;
namespace edm {

  //------------------------------------------------------------
  // Class RootDelayedReader: pretends to support file reading.
  //

  class RootDelayedReader : public DelayedReader {
  public:
    typedef input::BranchMap BranchMap;
    typedef input::EntryNumber EntryNumber;
    RootDelayedReader(EntryNumber const& entry,
      boost::shared_ptr<BranchMap const> bMap,
      boost::shared_ptr<TFile const> filePtr,
      FileFormatVersion const& fileFormatVersion);

    virtual ~RootDelayedReader();
  private:
    RootDelayedReader(RootDelayedReader const&); // disable copy construction
    RootDelayedReader & operator=(RootDelayedReader const&); // disable assignment
    virtual std::auto_ptr<EDProduct> getProduct(BranchKey const& k, EDProductGetter const* ep) const;
    virtual std::auto_ptr<EntryDescription> getProvenance(BranchKey const& k) const;
    BranchMap const& branches() const {return *branches_;}
    EntryNumber const entryNumber_;
    boost::shared_ptr<BranchMap const> branches_;
    // NOTE: filePtr_ appears to be unused, but is needed
    // to prevent the TFile containing the branch frong reclaimed.
    boost::shared_ptr<TFile const> filePtr_;
    FileFormatVersion fileFormatVersion_;
  }; // class RootDelayedReader
  //------------------------------------------------------------
}
#endif
