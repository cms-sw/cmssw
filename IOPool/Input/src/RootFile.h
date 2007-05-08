#ifndef Input_RootFile_h
#define Input_RootFile_h

/*----------------------------------------------------------------------

RootFile.h // used by ROOT input sources

$Id: RootFile.h,v 1.1 2007/04/16 19:43:51 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"
#include "boost/array.hpp"

#include "Inputfwd.h"
#include "RootTree.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "TBranch.h"
#include "TFile.h"

namespace edm {

  //------------------------------------------------------------
  // Class RootFile: supports file reading.

  class RootFile {
  public:
    typedef boost::array<RootTree *, EndBranchType> RootTreePtrArray;
    explicit RootFile(std::string const& fileName,
		      std::string const& catalogName,
		      ProcessConfiguration const& processConfiguration,
		      std::string const& logicalFileName = std::string());
    ~RootFile();
    void open();
    void close();
    std::auto_ptr<EventPrincipal> read(ProductRegistry const& pReg);
    ProductRegistry const& productRegistry() const {return *productRegistry_;}
    boost::shared_ptr<ProductRegistry> productRegistrySharedPtr() const {return productRegistry_;}
    EventAuxiliary const& eventAux() {return eventAux_;}
    LuminosityBlockAuxiliary const& luminosityBlockAux() {return lumiAux_;}
    RunAuxiliary const& runAux() {return runAux_;}
    EventID const& eventID() {return eventAux().id();}
    RootTreePtrArray & treePointers() {return treePointers_;}
    RootTree & eventTree() {return eventTree_;}
    RootTree & lumiTree() {return lumiTree_;}
    RootTree & runTree() {return runTree_;}

  private:
    void validateFile();
    boost::shared_ptr<RunPrincipal> readRun(ProductRegistry const& pReg, RunNumber_t const& runNumber);
    boost::shared_ptr<LuminosityBlockPrincipal> readLumi(ProductRegistry const& pReg,
							 RunNumber_t const& runNumber,
							 LuminosityBlockNumber_t const& lumiID,
							 bool isNewRun);
    RootFile(RootFile const&); // disable copy construction
    RootFile & operator=(RootFile const&); // disable assignment
    std::string const file_;
    std::string const logicalFile_;
    std::string const catalog_;
    ProcessConfiguration const& processConfiguration_;
    boost::shared_ptr<TFile> filePtr_;
    FileFormatVersion fileFormatVersion_;
    JobReport::Token reportToken_;
    EventAuxiliary eventAux_;
    LuminosityBlockAuxiliary lumiAux_;
    RunAuxiliary runAux_;
    RootTree eventTree_;
    RootTree lumiTree_;
    RootTree runTree_;
    RootTreePtrArray treePointers_;
    boost::shared_ptr<ProductRegistry> productRegistry_;
    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal_;
  }; // class RootFile

}
#endif
