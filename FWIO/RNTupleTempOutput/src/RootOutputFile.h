#ifndef FWIO_RNTupleTempOutput_RootOutputFile_h
#define FWIO_RNTupleTempOutput_RootOutputFile_h

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
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/propagate_const.h"
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
#include "DataFormats/Provenance/interface/CompactEventAuxiliaryVector.h"
#include "FWIO/RNTupleTempOutput/interface/RNTupleTempOutputModule.h"
#include "FWIO/RNTupleTempOutput/src/RootOutputRNTuple.h"

class TFile;
class TClass;

#include "ROOT/RFieldBase.hxx"
#include "ROOT/REntry.hxx"

namespace edm {
  class OccurrenceForOutput;
}
namespace edm::rntuple_temp {
  class RNTupleTempOutputModule;

  class RootOutputFile {
  public:
    using OutputItem = RNTupleTempOutputModule::OutputItem;
    using OutputItemList = RNTupleTempOutputModule::OutputItemList;
    explicit RootOutputFile(RNTupleTempOutputModule* om,
                            std::string const& fileName,
                            std::string const& logicalFileName,
                            std::vector<std::string> const& processesWithSelectedMergeableRunProducts,
                            std::string const& overrideGUID);
    ~RootOutputFile() {}
    void writeOne(EventForOutput const& e);
    //void endFile();
    void writeLuminosityBlock(LuminosityBlockForOutput const& lb);
    void writeRun(RunForOutput const& r);
    void writeProcessBlock(ProcessBlockForOutput const&);
    void writeParameterSetRegistry();
    void writeParentageRegistry();
    void writeMetaData(ProductRegistry const&);

    void finishEndFile();
    void beginInputFile(FileBlock const& fb, int remainingEvents);
    void respondToCloseInputFile(FileBlock const& fb);
    bool shouldWeCloseFile() const;

    std::string const& fileName() const { return file_; }

  private:
    std::unique_ptr<ROOT::RFieldBase> setupFileFormatVersion();
    std::unique_ptr<ROOT::RFieldBase> setupFileIdentifier();
    std::unique_ptr<ROOT::RFieldBase> setupIndexIntoFile();
    std::unique_ptr<ROOT::RFieldBase> setupStoredMergeableRunProductMetadata();
    std::unique_ptr<ROOT::RFieldBase> setupProcessHistoryRegistry();
    std::unique_ptr<ROOT::RFieldBase> setupProductDescriptionRegistry();
    std::unique_ptr<ROOT::RFieldBase> setupBranchIDListRegistry();
    std::unique_ptr<ROOT::RFieldBase> setupThinnedAssociationsHelper();
    std::unique_ptr<ROOT::RFieldBase> setupProductDependencies();
    std::unique_ptr<ROOT::RFieldBase> setupProcessBlockHelper();

    void writeFileFormatVersion(ROOT::REntry&);
    void writeFileIdentifier(ROOT::REntry&);
    void writeIndexIntoFile(ROOT::REntry&);
    void writeStoredMergeableRunProductMetadata(ROOT::REntry&);
    void writeProcessHistoryRegistry(ROOT::REntry&);
    void writeProductDescriptionRegistry(ROOT::REntry&, ProductRegistry const&);
    void writeBranchIDListRegistry(ROOT::REntry&);
    void writeThinnedAssociationsHelper(ROOT::REntry&);
    void writeProductDependencies(ROOT::REntry&);
    void writeProcessBlockHelper(ROOT::REntry&);

    void fillBranches(BranchType const& branchType,
                      OccurrenceForOutput const& occurrence,
                      unsigned int ttreeIndex,
                      StoredProductProvenanceVector* productProvenanceVecPtr = nullptr,
                      ProductProvenanceRetriever const* provRetriever = nullptr);

    void insertAncestors(ProductProvenance const& iGetParents,
                         ProductProvenanceRetriever const* iMapper,
                         bool produced,
                         std::set<BranchID> const& producedBranches,
                         std::set<StoredProductProvenance>& oToFill);

    bool insertProductProvenance(const ProductProvenance&, std::set<StoredProductProvenance>& oToInsert);

    std::shared_ptr<TFile const> filePtr() const { return get_underlying_safe(filePtr_); }
    std::shared_ptr<TFile>& filePtr() { return get_underlying_safe(filePtr_); }

    //-------------------------------
    // Member data

    std::string file_;
    std::string logicalFile_;
    JobReport::Token reportToken_;
    edm::propagate_const<RNTupleTempOutputModule*> om_;
    edm::propagate_const<std::shared_ptr<TFile>> filePtr_;
    FileID fid_;
    IndexIntoFile::EntryNumber_t eventEntryNumber_;
    IndexIntoFile::EntryNumber_t lumiEntryNumber_;
    IndexIntoFile::EntryNumber_t runEntryNumber_;
    IndexIntoFile indexIntoFile_;
    StoredMergeableRunProductMetadata storedMergeableRunProductMetadata_;
    unsigned long nEventsInLumi_;
    LuminosityBlockAuxiliary lumiAux_;
    RunAuxiliary runAux_;
    EventAuxiliary const* pEventAux_;
    LuminosityBlockAuxiliary const* pLumiAux_;
    RunAuxiliary const* pRunAux_;
    StoredProductProvenanceVector eventEntryInfoVector_;
    StoredProductProvenanceVector const* pEventEntryInfoVector_;
    BranchListIndexes const* pBranchListIndexes_;
    EventToProcessBlockIndexes const* pEventToProcessBlockIndexes_;
    EventSelectionIDVector const* pEventSelectionIDs_;
    RootOutputRNTuple eventRNTuple_;
    RootOutputRNTuple lumiRNTuple_;
    RootOutputRNTuple runRNTuple_;
    std::vector<edm::propagate_const<std::unique_ptr<RootOutputRNTuple>>> processBlockRNTuples_;
    std::vector<edm::propagate_const<RootOutputRNTuple*>> treePointers_;
    bool dataTypeReported_;
    ProcessHistoryRegistry processHistoryRegistry_;
    std::map<ParentageID, unsigned int> parentageIDs_;
    std::set<BranchID> branchesWithStoredHistory_;
    edm::propagate_const<TClass*> wrapperBaseTClass_;
  };

}  // namespace edm::rntuple_temp

#endif
