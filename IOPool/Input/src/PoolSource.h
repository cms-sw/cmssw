#ifndef Input_PoolSource_h
#define Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

$Id: PoolSource.h,v 1.11 2005/09/15 16:33:09 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <map>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/RandomAccessInputSource.h"
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
  class PoolSource : public RandomAccessInputSource {
  public:
    typedef std::map<BranchKey, std::pair<std::string, TBranch *> > BranchMap;
    typedef Long64_t EntryNumber;
  private:
    //------------------------------------------------------------
    // Nested class PoolDelayedReader: pretends to support file reading.
    //

    class PoolDelayedReader : public DelayedReader {
    public:
      PoolDelayedReader(EntryNumber const& entry, PoolSource const& serv) : entryNumber_(entry), inputSource(&serv) {}
      virtual ~PoolDelayedReader();
      virtual std::auto_ptr<EDProduct> get(BranchKey const& k) const;
      BranchMap const& branches() const {return inputSource->branches_;}
    private:
      EntryNumber const entryNumber_;
      PoolSource const* inputSource;
    }; // class PoolSource::PoolDelayedReader
    //------------------------------------------------------------

  public:
    friend class PoolDelayedReader;
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolSource();

  private:
    std::map<ProductID, ProductDescription> productMap;
    std::string const file_;
    EntryNumber remainingEvents_;
    EntryNumber entryNumber_;
    EntryNumber entries_;
    EventID eventID_;
    BranchMap branches_;
    TBranch *auxBranch_;
    TBranch *provBranch_;

    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> read(EventID const& id);
    virtual void skip(int offset);
    void init();
  }; // class PoolSource
}
#endif
