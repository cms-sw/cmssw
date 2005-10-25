#ifndef Input_PoolSource_h
#define Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

$Id: PoolSource.h,v 1.3 2005/10/03 19:00:29 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <map>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/RandomAccessInputSource.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "TBranch.h"

#include "boost/shared_ptr.hpp"

// forwards
namespace seal { class Status; }

namespace edm {

  class ParameterSet;
  class PoolRASource : public RandomAccessInputSource {
  public:
    typedef Long64_t EntryNumber;
    class PoolFile;
    class PoolDelayedReader;

    explicit PoolRASource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolRASource();

  private:
    std::map<ProductID, BranchDescription> productMap_;
    std::string const file_;
    boost::shared_ptr<PoolFile> poolFile_;
    EntryNumber remainingEvents_;
    EventID eventID_;

    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> read(EventID const& id);
    virtual void skip(int offset);
    void init();
  }; // class PoolRASource

  //------------------------------------------------------------
  // Nested class PoolFile: supports file reading.

  class PoolRASource::PoolFile {
  public:
    typedef std::map<BranchKey, std::pair<std::string, TBranch *> > BranchMap;
    BranchMap const& branches() const {return branches_;}
    explicit PoolFile(std::string const& fileName);
    ~PoolFile() {}
    bool next() {return ++entryNumber_ < entries_;} 
    ProductRegistry const& productRegistry() const {return productRegistry_;}
    TBranch *auxBranch() {return auxBranch_;}
    TBranch *provBranch() {return provBranch_;}
    EntryNumber & entryNumber() {return entryNumber_;}

  private:
    PoolFile(PoolFile const&); // disable copy construction
    PoolFile & operator=(PoolFile const&); // disable assignment
    std::string const file_;
    EntryNumber entryNumber_;
    EntryNumber entries_;
    ProductRegistry productRegistry_;
    BranchMap branches_;
    TBranch *auxBranch_;
    TBranch *provBranch_;
  }; // class PoolRASource::PoolFile


  //------------------------------------------------------------
  // Nested class PoolDelayedReader: pretends to support file reading.
  //

  class PoolRASource::PoolDelayedReader : public DelayedReader {
  public:
    PoolDelayedReader(EntryNumber const& entry, PoolRASource const& serv) : entryNumber_(entry), inputSource(serv) {}
    virtual ~PoolDelayedReader();
  private:
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k) const;
    PoolFile::BranchMap const& branches() const {return inputSource.poolFile_->branches();}
    EntryNumber const entryNumber_;
    PoolRASource const& inputSource;
  }; // class PoolRASource::PoolDelayedReader
  //------------------------------------------------------------

}
#endif
