#ifndef Input_PoolSource_h
#define Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

$Id: PoolSource.h,v 1.8 2005/11/01 23:24:13 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <map>
#include <vector>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/RandomAccessInputSource.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "IOPool/Common/interface/PoolCatalog.h"
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
    PoolRASource(PoolRASource const&); // disable copy construction
    PoolRASource & operator=(PoolRASource const&); // disable assignment
    virtual std::auto_ptr<EventPrincipal> read();
    virtual std::auto_ptr<EventPrincipal> read(EventID const& id);
    virtual void skip(int offset);
    void init(std::string const& file);
    bool next();

    PoolCatalog catalog_;
    std::map<ProductID, BranchDescription> productMap_;
    std::string const file_;
    std::vector<std::string> const files_;
    std::vector<std::string>::const_iterator fileIter_;
    boost::shared_ptr<PoolFile> poolFile_;
    EntryNumber remainingEvents_;
    EventID eventID_;
  }; // class PoolRASource

  //------------------------------------------------------------
  // Nested class PoolFile: supports file reading.

  class PoolRASource::PoolFile {
  public:
    typedef std::map<BranchKey, std::pair<std::string, TBranch *> > BranchMap;
    BranchMap const& branches() const {return branches_;}
    explicit PoolFile(std::string const& fileName);
    ~PoolFile();
    bool next() {return ++entryNumber_ < entries_;} 
    ProductRegistry const& productRegistry() const {return *productRegistry_;}
    boost::shared_ptr<ProductRegistry> productRegistrySharedPtr() const {return productRegistry_;}
    TBranch *auxBranch() {return auxBranch_;}
    TBranch *provBranch() {return provBranch_;}
    EntryNumber & entryNumber() {return entryNumber_;}

  private:
    PoolFile(PoolFile const&); // disable copy construction
    PoolFile & operator=(PoolFile const&); // disable assignment
    std::string const file_;
    EntryNumber entryNumber_;
    EntryNumber entries_;
    boost::shared_ptr<ProductRegistry> productRegistry_;
// We use bare pointers for pointers to ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore,using shared pointers here will do no good.
    BranchMap branches_;
    TBranch *auxBranch_;
    TBranch *provBranch_;
    TFile *filePtr_;
  }; // class PoolRASource::PoolFile


  //------------------------------------------------------------
  // Nested class PoolDelayedReader: pretends to support file reading.
  //

  class PoolRASource::PoolDelayedReader : public DelayedReader {
  public:
    PoolDelayedReader(EntryNumber const& entry, PoolRASource const& serv) : entryNumber_(entry), inputSource(serv) {}
    virtual ~PoolDelayedReader();
  private:
    PoolDelayedReader(PoolDelayedReader const&); // disable copy construction
    PoolDelayedReader & operator=(PoolDelayedReader const&); // disable assignment
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k, EventPrincipal const* ep) const;
    PoolFile::BranchMap const& branches() const {return inputSource.poolFile_->branches();}
    EntryNumber const entryNumber_;
    PoolRASource const& inputSource;
  }; // class PoolRASource::PoolDelayedReader
  //------------------------------------------------------------

}
#endif
