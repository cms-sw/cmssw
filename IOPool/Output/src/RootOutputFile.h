#ifndef IOPool_Output_RootOutputFile_h
#define IOPool_Output_RootOutputFile_h

//////////////////////////////////////////////////////////////////////
//
// Class RootOutputFile
//
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <map>
#include <string>
#include <vector>

#include "boost/array.hpp"
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/Selections.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "IOPool/Output/interface/PoolOutputModule.h"
#include "IOPool/Output/src/RootOutputTree.h"

class TTree;
class TFile;
#include "TROOT.h"

namespace edm {
  class PoolOutputModule;
  class History;

  class RootOutputFile {
  public:
    typedef PoolOutputModule::OutputItem OutputItem;
    typedef PoolOutputModule::OutputItemList OutputItemList;
    typedef boost::array<RootOutputTree*, NumBranchTypes> RootOutputTreePtrArray;
    explicit RootOutputFile(PoolOutputModule* om, std::string const& fileName,
                            std::string const& logicalFileName);
    ~RootOutputFile() {}
    void writeOne(EventPrincipal const& e);
    //void endFile();
    void writeLuminosityBlock(LuminosityBlockPrincipal const& lb);
    void writeRun(RunPrincipal const& r);
    void writeFileFormatVersion();
    void writeFileIdentifier();
    void writeFileIndex();
    void writeEventHistory();
    void writeProcessConfigurationRegistry();
    void writeProcessHistoryRegistry();
    void writeParameterSetRegistry();
    void writeProductDescriptionRegistry();
    void writeParentageRegistry();
    void writeBranchIDListRegistry();
    void writeProductDependencies();

    void finishEndFile();
    void beginInputFile(FileBlock const& fb, int remainingEvents);
    void respondToCloseInputFile(FileBlock const& fb);
    bool shouldWeCloseFile() const;

  private:

    //-------------------------------
    // Local types
    //

    //-------------------------------
    // Private functions

    void setBranchAliases(TTree* tree, Selections const& branches) const;

    void fillBranches(BranchType const& branchType,
		      Principal const& principal,
		      std::vector<ProductProvenance>* productProvenanceVecPtr);

     void insertAncestors(ProductProvenance const& iGetParents,
                          Principal const& principal,
                          bool produced,
                          std::set<ProductProvenance>& oToFill);
        
    //-------------------------------
    // Member data

    std::string file_;
    std::string logicalFile_;
    JobReport::Token reportToken_;
    PoolOutputModule const* om_;
    int whyNotFastClonable_;
    boost::shared_ptr<TFile> filePtr_;
    FileID fid_;
    FileIndex fileIndex_;
    FileIndex::EntryNumber_t eventEntryNumber_;
    FileIndex::EntryNumber_t lumiEntryNumber_;
    FileIndex::EntryNumber_t runEntryNumber_;
    TTree* metaDataTree_;
    TTree* parentageTree_;
    TTree* eventHistoryTree_;
    EventAuxiliary const*           pEventAux_;
    LuminosityBlockAuxiliary const* pLumiAux_;
    RunAuxiliary const*             pRunAux_;
    ProductProvenanceVector         eventEntryInfoVector_;
    ProductProvenanceVector	    lumiEntryInfoVector_;
    ProductProvenanceVector         runEntryInfoVector_;
    ProductProvenanceVector*       pEventEntryInfoVector_;
    ProductProvenanceVector*       pLumiEntryInfoVector_;
    ProductProvenanceVector*       pRunEntryInfoVector_;
    History const*                  pHistory_;
    RootOutputTree eventTree_;
    RootOutputTree lumiTree_;
    RootOutputTree runTree_;
    RootOutputTreePtrArray treePointers_;
    bool dataTypeReported_;
    std::set<ParentageID> parentageIDs_;
    std::set<BranchID> branchesWithStoredHistory_;
  };
   
}

#endif
