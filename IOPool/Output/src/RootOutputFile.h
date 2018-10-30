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
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/StoredProductProvenance.h"
#include "DataFormats/Provenance/interface/StoredMergeableRunProductMetadata.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"
#include "IOPool/Output/interface/PoolOutputModule.h"
#include "IOPool/Output/src/RootOutputTree.h"

class TTree;
class TFile;
class TClass;

namespace edm {
  class OccurrenceForOutput;
  class PoolOutputModule;

  class RootOutputFile {
  public:
    typedef PoolOutputModule::OutputItem OutputItem;
    typedef PoolOutputModule::OutputItemList OutputItemList;
    typedef std::array<edm::propagate_const<RootOutputTree*>, NumBranchTypes> RootOutputTreePtrArray;
    explicit RootOutputFile(PoolOutputModule* om, std::string const& fileName,
                            std::string const& logicalFileName,
                            std::vector<std::string> const& processesWithSelectedMergeableRunProducts);
    ~RootOutputFile() {}
    void writeOne(EventForOutput const& e);
    //void endFile();
    void writeLuminosityBlock(LuminosityBlockForOutput const& lb);
    void writeRun(RunForOutput const& r);
    void writeFileFormatVersion();
    void writeFileIdentifier();
    void writeIndexIntoFile();
    void writeStoredMergeableRunProductMetadata();
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
                      OccurrenceForOutput const& occurrence,
                      StoredProductProvenanceVector* productProvenanceVecPtr = nullptr,
                      ProductProvenanceRetriever const* provRetriever = nullptr);

     void insertAncestors(ProductProvenance const& iGetParents,
                          ProductProvenanceRetriever const* iMapper,
                          bool produced,
                          std::set<BranchID> const& producedBranches,
                          std::set<StoredProductProvenance>& oToFill);

    bool insertProductProvenance(const ProductProvenance&,
                                 std::set<StoredProductProvenance>& oToInsert);

    std::shared_ptr<TFile const> filePtr() const {return get_underlying_safe(filePtr_);}
    std::shared_ptr<TFile>& filePtr() {return get_underlying_safe(filePtr_);}
    StoredProductProvenanceVector const* pEventEntryInfoVector() const {return get_underlying_safe(pEventEntryInfoVector_);}
    StoredProductProvenanceVector*& pEventEntryInfoVector() {return get_underlying_safe(pEventEntryInfoVector_);}

    //-------------------------------
    // Member data

    std::string file_;
    std::string logicalFile_;
    JobReport::Token reportToken_;
    edm::propagate_const<PoolOutputModule*> om_;
    int whyNotFastClonable_;
    bool canFastCloneAux_;
    edm::propagate_const<std::shared_ptr<TFile>> filePtr_;
    FileID fid_;
    IndexIntoFile::EntryNumber_t eventEntryNumber_;
    IndexIntoFile::EntryNumber_t lumiEntryNumber_;
    IndexIntoFile::EntryNumber_t runEntryNumber_;
    IndexIntoFile indexIntoFile_;
    StoredMergeableRunProductMetadata storedMergeableRunProductMetadata_;
    unsigned long nEventsInLumi_;
    edm::propagate_const<TTree*> metaDataTree_;
    edm::propagate_const<TTree*> parameterSetsTree_;
    edm::propagate_const<TTree*> parentageTree_;
    LuminosityBlockAuxiliary  lumiAux_;
    RunAuxiliary              runAux_;
    EventAuxiliary const*           pEventAux_;
    LuminosityBlockAuxiliary const* pLumiAux_;
    RunAuxiliary const*             pRunAux_;
    StoredProductProvenanceVector eventEntryInfoVector_;
    edm::propagate_const<StoredProductProvenanceVector*> pEventEntryInfoVector_;
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
    edm::propagate_const<TClass*> wrapperBaseTClass_;
  };

}

#endif
