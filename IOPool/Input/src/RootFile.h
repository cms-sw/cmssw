#ifndef Input_RootFile_h
#define Input_RootFile_h

/*----------------------------------------------------------------------

RootFile.h // used by ROOT input sources

$Id: RootFile.h,v 1.14 2006/12/14 04:30:59 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"
#include "boost/array.hpp"

#include "IOPool/Input/src/Inputfwd.h"
#include "IOPool/Input/src/RootTree.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/EventAux.h"
#include "DataFormats/Common/interface/LuminosityBlockAux.h"
#include "DataFormats/Common/interface/RunAux.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "TBranch.h"
#include "TFile.h"

namespace edm {

  //------------------------------------------------------------
  // Class RootFile: supports file reading.

  class RootFile {
  public:
    typedef RootTree::BranchMap BranchMap;
    typedef RootTree::ProductMap ProductMap;
    typedef boost::array<RootTree *, EndBranchType> RootTreePtrArray;
    explicit RootFile(std::string const& fileName,
		      std::string const& catalogName,
		      std::string const& logicalFileName = std::string());
    ~RootFile();
    void open();
    void close();
    std::auto_ptr<EventPrincipal> read(ProductRegistry const& pReg);
    ProductRegistry const& productRegistry() const {return *productRegistry_;}
    boost::shared_ptr<ProductRegistry> productRegistrySharedPtr() const {return productRegistry_;}
    EventAux const& eventAux() {return eventAux_;}
    LuminosityBlockAux const& luminosityBlockAux() {return lumiAux_;}
    RunAux const& runAux() {return runAux_;}
    EventID const& eventID() {return eventAux().id();}
    RootTreePtrArray & treePointers() {return treePointers_;}
    RootTree & eventTree() {return eventTree_;}
    RootTree & lumiTree() {return lumiTree_;}
    RootTree & runTree() {return runTree_;}
    BranchMap const& branches() const {return *branches_;}

  private:
    RootFile(RootFile const&); // disable copy construction
    RootFile & operator=(RootFile const&); // disable assignment
    std::string const file_;
    std::string const logicalFile_;
    std::string const catalog_;
    boost::shared_ptr<TFile> filePtr_;
    JobReport::Token reportToken_;
    EventAux eventAux_;
    LuminosityBlockAux lumiAux_;
    RunAux runAux_;
    RootTree eventTree_;
    RootTree lumiTree_;
    RootTree runTree_;
    RootTreePtrArray treePointers_;
    boost::shared_ptr<ProductRegistry> productRegistry_;
    boost::shared_ptr<BranchMap> branches_;
    ProductMap products_;
    boost::shared_ptr<LuminosityBlockPrincipal const> luminosityBlockPrincipal_;
  }; // class RootFile

}
#endif
