#ifndef IOPool_Output_RootOutputFile_h
#define IOPool_Output_RootOutputFile_h

//////////////////////////////////////////////////////////////////////
//
// Class RootOutputFile
//
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <array>
#include <map>
#include <string>
#include <vector>

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/StoredProductProvenance.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"
#include "IOPool/Output/interface/PoolOutputModule.h"
#include "IOPool/Output/src/RootOutputTree.h"

class TTree;
class TFile;
class TClass;

namespace edm {
  class ModuleCallingContext;
  class PoolOutputModule;


  class RootOutputFile {
  public:
    typedef PoolOutputModule::OutputItem OutputItem;
    typedef PoolOutputModule::OutputItemList OutputItemList;
    typedef std::array<RootOutputTree*, NumBranchTypes> RootOutputTreePtrArray;
    explicit RootOutputFile(PoolOutputModule* om, std::string const& fileName,
                            std::string const& logicalFileName);
    ~RootOutputFile() {}
    void writeOne(EventPrincipal const& e, ModuleCallingContext const*);
    //void endFile();
    void writeLuminosityBlock(LuminosityBlockPrincipal const& lb, ModuleCallingContext const*);
    void writeRun(RunPrincipal const& r, ModuleCallingContext const*);
    void writeFileFormatVersion();
    void writeFileIdentifier();
    void writeIndexIntoFile();
    void writeProcessHistoryRegistry();
    void writeParameterSetRegistry();
    void writeProductDescriptionRegistry();
    void writeParentageRegistry();
    void writeBranchIDListRegistry();
    void writeThinnedAssociationsHelper();
    void writeProductDependencies();

    void finishEndFile();
    void beginInputFile(FileBlock const& fb, int remainingEvents);
    void respondToCloseInputFile(FileBlock const& fb);
    bool shouldWeCloseFile() const;

    std::string const& fileName() const {return file_;}

  private:

    //-------------------------------
    // Local types
    //

    //-------------------------------
    // Private functions

    void setBranchAliases(TTree* tree, SelectedProducts const& branches) const;

    void fillBranches(BranchType const& branchType,
                      Principal const& principal,
                      StoredProductProvenanceVector* productProvenanceVecPtr,
                      ModuleCallingContext const*);

     void insertAncestors(ProductProvenance const& iGetParents,
                          EventPrincipal const& principal,
                          bool produced,
                          std::set<StoredProductProvenance>& oToFill,
                          ModuleCallingContext const*);

    bool insertProductProvenance(const ProductProvenance&,
                                 std::set<StoredProductProvenance>& oToInsert);
    //-------------------------------
    // Member data

    std::string file_;
    std::string logicalFile_;
    JobReport::Token reportToken_;
    PoolOutputModule* om_;
    int whyNotFastClonable_;
    bool canFastCloneAux_;
    std::shared_ptr<TFile> filePtr_;
    FileID fid_;
    IndexIntoFile::EntryNumber_t eventEntryNumber_;
    IndexIntoFile::EntryNumber_t lumiEntryNumber_;
    IndexIntoFile::EntryNumber_t runEntryNumber_;
    IndexIntoFile indexIntoFile_;
    TTree* metaDataTree_;
    TTree* parameterSetsTree_;
    TTree* parentageTree_;
    LuminosityBlockAuxiliary  lumiAux_;
    RunAuxiliary              runAux_;
    EventAuxiliary const*           pEventAux_;
    LuminosityBlockAuxiliary const* pLumiAux_;
    RunAuxiliary const*             pRunAux_;
    StoredProductProvenanceVector eventEntryInfoVector_;
    StoredProductProvenanceVector*        pEventEntryInfoVector_;
    BranchListIndexes const*        pBranchListIndexes_;
    EventSelectionIDVector const*   pEventSelectionIDs_;
    RootOutputTree eventTree_;
    RootOutputTree lumiTree_;
    RootOutputTree runTree_;
    RootOutputTreePtrArray treePointers_;
    bool dataTypeReported_;
    ProcessHistoryRegistry processHistoryRegistry_;
    std::map<ParentageID,unsigned int> parentageIDs_;
    std::set<BranchID> branchesWithStoredHistory_;
    TClass* wrapperBaseTClass_;
  };

}

#endif
