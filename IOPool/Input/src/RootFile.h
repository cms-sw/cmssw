#ifndef IOPool_Input_RootFile_h
#define IOPool_Input_RootFile_h

/*----------------------------------------------------------------------

RootFile.h // used by ROOT input sources

$Id: RootFile.h,v 1.39 2007/11/03 06:53:02 wmtan Exp $

----------------------------------------------------------------------*/

#include <map>
#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"
#include "boost/array.hpp"

#include "RootTree.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
class TFile;

namespace edm {

  //------------------------------------------------------------
  // Class RootFile: supports file reading.

  class RootFile {
  public:
    typedef boost::array<RootTree *, NumBranchTypes> RootTreePtrArray;
    explicit RootFile(std::string const& fileName,
		      std::string const& catalogName,
		      ProcessConfiguration const& processConfiguration,
		      std::string const& logicalFileName = std::string());
    ~RootFile();
    void open();
    void close(bool reallyClose);
    std::auto_ptr<EventPrincipal> readEvent(
	boost::shared_ptr<ProductRegistry const> pReg,
	boost::shared_ptr<LuminosityBlockPrincipal> lbp);
    boost::shared_ptr<LuminosityBlockPrincipal> readLumi(
	boost::shared_ptr<ProductRegistry const> pReg,
	boost::shared_ptr<RunPrincipal> rp);
    boost::shared_ptr<RunPrincipal> readRun(boost::shared_ptr<ProductRegistry const> pReg);
    boost::shared_ptr<ProductRegistry const> productRegistry() const {return productRegistry_;}
    EventAuxiliary const& eventAux() const {return eventAux_;}
    LuminosityBlockAuxiliary const& luminosityBlockAux() {return lumiAux_;}
    RunAuxiliary const& runAux() const {return runAux_;}
    EventID const& eventID() const {return eventAux().id();}
    RootTreePtrArray & treePointers() {return treePointers_;}
    RootTree & eventTree() {return eventTree_;}
    RootTree const& eventTree() const {return eventTree_;}
    RootTree & lumiTree() {return lumiTree_;}
    RootTree const& lumiTree() const {return lumiTree_;}
    RootTree & runTree() {return runTree_;}
    RootTree const & runTree() const {return runTree_;}
    void forceRunNumber(RunNumber_t const& run) {forcedRunNumber_ = run;}
    FileFormatVersion fileFormatVersion() const {return fileFormatVersion_;}
    bool fastClonable() const {return fileFormatVersion().value_ >= 3;}
    boost::shared_ptr<FileBlock> createFileBlock(bool isFastClonable) const;

  private:
    void validateFile();
    void fillEventAuxiliary();
    void overrideRunNumber(RunID & id);
    void overrideRunNumber(LuminosityBlockID & id);
    void overrideRunNumber(EventID & id, bool isRealData);
    std::string const& newBranchToOldBranch(std::string const& newBranch) const;
    RootFile(RootFile const&); // disable copy construction
    RootFile & operator=(RootFile const&); // disable assignment
    std::string const file_;
    std::string const logicalFile_;
    std::string const catalog_;
    ProcessConfiguration const& processConfiguration_;
    boost::shared_ptr<TFile> filePtr_;
    FileFormatVersion fileFormatVersion_;
    FileID fid_;
    JobReport::Token reportToken_;
    EventAuxiliary eventAux_;
    LuminosityBlockAuxiliary lumiAux_;
    RunAuxiliary runAux_;
    RootTree eventTree_;
    RootTree lumiTree_;
    RootTree runTree_;
    RootTreePtrArray treePointers_;
    boost::shared_ptr<ProductRegistry const> productRegistry_;
    RunNumber_t forcedRunNumber_;
    int forcedRunNumberOffset_;
    std::map<std::string, std::string> newBranchToOldBranch_;
    std::vector<std::string> sortedNewBranchNames_;
    std::vector<std::string> oldBranchNames_;
  }; // class RootFile

}
#endif
