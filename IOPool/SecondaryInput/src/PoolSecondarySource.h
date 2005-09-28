#ifndef Input_PoolSecondarySource_h
#define Input_PoolSecondarySource_h

/*----------------------------------------------------------------------

PoolSecondarySource: This is a SecondaryInputSource

$Id: PoolSecondarySource.h,v 1.11 2005/09/15 16:33:09 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <map>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/SecondaryInputSource.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "TBranch.h"

// forwards
namespace seal { class Status; }

namespace edm {

  class ParameterSet;
  class InputSourceDescription;
  class PoolSecondarySource : public SecondaryInputSource {
  public:
    typedef std::map<BranchKey, std::pair<std::string, TBranch *> > BranchMap;
    typedef Long64_t EntryNumber;
  private:
    //------------------------------------------------------------
    // Nested class PoolDelayedReader: pretends to support file reading.
    //

    class PoolDelayedReader : public DelayedReader {
    public:
      PoolDelayedReader(EntryNumber const& entry, PoolSecondarySource const& serv) : entryNumber_(entry), inputSource(&serv) {}
      virtual ~PoolDelayedReader();
      virtual std::auto_ptr<EDProduct> get(BranchKey const& k) const;
      BranchMap const& branches() const {return inputSource->branches_;}
    private:
      EntryNumber const entryNumber_;
      PoolSecondarySource const* inputSource;
    }; // class PoolSecondarySource::PoolDelayedReader
    //------------------------------------------------------------

  public:
    friend class PoolDelayedReader;
    explicit PoolSecondarySource(ParameterSet const& pset, InputSourceDescription const&);
    virtual ~PoolSecondarySource();

  private:
    std::map<ProductID, ProductDescription> productMap;
    std::string const file_;
    EntryNumber entries_;
    BranchMap branches_;
    TBranch *auxBranch_;
    TBranch *provBranch_;
    boost::shared_ptr<ProductRegistry> pReg_;

    virtual void read(int idx, int number, std::vector<EventPrincipal*>& result);
    void init();
  }; // class PoolSecondarySource
}
#endif
