#ifndef Input_RootFile_h
#define Input_RootFile_h

/*----------------------------------------------------------------------

RootFile.h // used by ROOT input sources

$Id: RootFile.h,v 1.13 2006/09/24 17:11:11 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"

#include "IOPool/Input/src/Inputfwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/EventAux.h"
#include "TBranch.h"
#include "TFile.h"

namespace edm {

  //------------------------------------------------------------
  // Class RootFile: supports file reading.

  class RootFile {
  public:
    typedef input::BranchMap BranchMap;
    typedef input::EntryNumber EntryNumber;
    typedef std::map<ProductID, BranchDescription> ProductMap;
    BranchMap const& branches() const {return *branches_;}
    explicit RootFile(std::string const& fileName,
		      std::string const& catalogName,
		      std::string const& logicalFileName = std::string());
    ~RootFile();
    void open();
    void close();
    bool next() {return ++entryNumber_ < entries_;} 
    bool previous() {return --entryNumber_ >= 0;} 
    std::auto_ptr<EventPrincipal> read(ProductRegistry const& pReg);
    ProductRegistry const& productRegistry() const {return *productRegistry_;}
    boost::shared_ptr<ProductRegistry> productRegistrySharedPtr() const {return productRegistry_;}
    TBranch *auxBranch() {return auxBranch_;}
    EventAux const& eventAux() {return eventAux_;}
    EventID const& eventID() {return eventAux().id();}
    EntryNumber const& entryNumber() const {return entryNumber_;}
    EntryNumber const& entries() const {return entries_;}
    void setEntryNumber(EntryNumber theEntryNumber) {entryNumber_ = theEntryNumber;}
    EntryNumber getEntryNumber(EventID const& eventID) const;

  private:
    RootFile(RootFile const&); // disable copy construction
    RootFile & operator=(RootFile const&); // disable assignment
    std::string const file_;
    std::string const logicalFile_;
    std::string const catalog_;
    std::vector<std::string> branchNames_;
    std::vector<BranchEntryDescription> eventProvenance_;
    std::vector<BranchEntryDescription *> eventProvenancePtrs_;
    JobReport::Token reportToken_;
    EventAux eventAux_;
    EntryNumber entryNumber_;
    EntryNumber entries_;
    boost::shared_ptr<ProductRegistry> productRegistry_;
    boost::shared_ptr<BranchMap> branches_;
    ProductMap productMap_;
    boost::shared_ptr<LuminosityBlockPrincipal const> luminosityBlockPrincipal_;
// We use bare pointers for pointers to ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore,using shared pointers here will do no good.
    TTree *eventTree_;
    TTree *eventMetaTree_;
    TBranch *auxBranch_;
    boost::shared_ptr<TFile> filePtr_;
  }; // class RootFile


}
#endif
