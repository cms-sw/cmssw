#ifndef Input_PoolSecondarySource_h
#define Input_PoolSecondarySource_h

/*----------------------------------------------------------------------

PoolSecondarySource: This is a SecondaryInputSource

$Id: PoolSecondarySource.h,v 1.5 2005/10/27 23:55:34 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <map>
#include <vector>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/SecondaryInputSource.h"
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
  class PoolSecondarySource : public SecondaryInputSource {
  public:
    typedef Long64_t EntryNumber;
    class PoolFile;
    class PoolDelayedReader;
    explicit PoolSecondarySource(ParameterSet const& pset);
    virtual ~PoolSecondarySource();

  private:
    PoolSecondarySource(PoolSecondarySource const&); // disable copy construction
    PoolSecondarySource & operator=(PoolSecondarySource const&); // disable assignment
    virtual void read(int idx, int number, std::vector<EventPrincipal*>& result);
    void init(std::string const& file);
    bool next();

    PoolCatalog catalog_;
    std::map<ProductID, BranchDescription> productMap_;
    std::string const file_;
    std::vector<std::string> const files_;
    std::vector<std::string>::const_iterator fileIter_;
    boost::shared_ptr<PoolFile> poolFile_;
    boost::shared_ptr<ProductRegistry> pReg_;
  }; // class PoolSecondarySource


  //------------------------------------------------------------
  // Nested class PoolFile: supports file reading.

  class PoolSecondarySource::PoolFile {
  public:
    typedef std::map<BranchKey, std::pair<std::string, TBranch *> > BranchMap;
    BranchMap const& branches() const {return branches_;}
    explicit PoolFile(std::string const& fileName);
    ~PoolFile();
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
// We use bare pointers for pointers to ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore,using shared pointers here will do no good.
    BranchMap branches_;
    TBranch *auxBranch_;
    TBranch *provBranch_;
    TFile *filePtr_;
  }; // class PoolSecondarySource::PoolFile

  //------------------------------------------------------------
  // Nested class PoolDelayedReader: pretends to support file reading.
  //

  class PoolSecondarySource::PoolDelayedReader : public DelayedReader {
  public:
    PoolDelayedReader(EntryNumber const& entry, PoolSecondarySource const& serv) : entryNumber_(entry), inputSource(serv) {}
    virtual ~PoolDelayedReader();
  private:
    PoolDelayedReader(PoolDelayedReader const&); // disable copy construction
    PoolDelayedReader & operator=(PoolDelayedReader const&); // disable assignment
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k) const;
    PoolFile::BranchMap const& branches() const {return inputSource.poolFile_->branches();}
    EntryNumber const entryNumber_;
    PoolSecondarySource const& inputSource;
  }; // class PoolSecondarySource::PoolDelayedReader
  //------------------------------------------------------------

}
#endif
