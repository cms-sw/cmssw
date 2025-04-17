#if !defined(FWIO_RNTuple_RNTupleOutputFile_h)
#define FWIO_RNTuple_RNTupleOutputFile_h

#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/FileBlock.h"

#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/ProductDependencies.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"
#include "DataFormats/Provenance/interface/StoredProductProvenance.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"

#include "TFile.h"
#include "ROOT/RNTuple.hxx"
#include "ROOT/RNTupleWriter.hxx"
#include <string>
#include <optional>
#include <map>
#include <set>
#include <array>

namespace edm {
  namespace rntuple {
    enum class CompressionAlgos { kLZMA, kZSTD, kZLIB, kLZ4 };
  }

  class RNTupleOutputFile {
  public:
    struct Config {
      ParameterSetID selectorConfig;
      std::vector<bool> streamerProduct;
      std::vector<std::string> doNotSplitSubFields;
      rntuple::CompressionAlgos compressionAlgo = rntuple::CompressionAlgos::kZSTD;
      int compressionLevel = 4;
      unsigned long long approxZippedClusterSize;
      unsigned long long maxUnzippedClusterSize;
      unsigned long long initialUnzippedPageSize;
      unsigned long long maxUnzippedPageSize;
      unsigned long long pageBufferBudget;
      bool useBufferedWrite;
      bool useDirectIO;

      bool wantAllEvents;
      bool dropMetaData = false;
    };

    explicit RNTupleOutputFile(std::string const& iFileName,
                               FileBlock const& iFileBlock,
                               SelectedProductsForBranchType const& iSelected,
                               Config const&,
                               bool anyProductProduced);
    ~RNTupleOutputFile();

    void write(EventForOutput const& e);
    void writeLuminosityBlock(LuminosityBlockForOutput const&);
    void writeRun(RunForOutput const&);
    void reallyCloseFile(BranchIDLists const& iBranchIDLists,
                         ThinnedAssociationsHelper const& iThinnedHelper,
                         ProductRegistry const& iReg);
    void openFile(FileBlock const& fb);

    struct Product {
      Product(EDGetToken iGet, ProductDescription const* iDesc, ROOT::RFieldToken iField)
          : get_(iGet), desc_(iDesc), field_(iField) {}

      EDGetToken get_;
      ProductDescription const* desc_;
      ROOT::RFieldToken field_;
    };

  private:
    void setupRuns(SelectedProducts const&, Config const&);
    void setupLumis(SelectedProducts const&, Config const&);
    std::unique_ptr<ROOT::RNTupleModel> setupCommonModels(SelectedProducts const&,
                                                          std::string const& iAuxName,
                                                          std::string const& iAuxType);
    void setupEvents(SelectedProducts const&, Config const&, bool anyProductProduced);
    void setupPSets(Config const&);
    void setupParentage(Config const&);
    void setupMetaData(Config const&);

    void fillPSets();
    void fillParentage();
    void fillMetaData(BranchIDLists const& iBranchIDLists,
                      ThinnedAssociationsHelper const& iThinnedHelper,
                      ProductRegistry const&);

    void setupDataProducts(SelectedProducts const&,
                           std::vector<bool> const&,
                           std::vector<std::string> const&,
                           ROOT::RNTupleModel&);
    //Can't call until the model is frozen
    std::vector<Product> associateDataProducts(SelectedProducts const&, ROOT::RNTupleModel const&);

    std::vector<std::unique_ptr<edm::WrapperBase>> writeDataProducts(std::vector<Product> const& iProduct,
                                                                     OccurrenceForOutput const& iOccurence,
                                                                     ROOT::REntry&);
    std::vector<StoredProductProvenance> writeDataProductProvenance(std::vector<Product> const& iProduct,
                                                                    EventForOutput const& iEvent);
    bool insertProductProvenance(ProductProvenance const& iProv, std::set<StoredProductProvenance>& oToKeep);
    void insertAncestorsProvenance(ProductProvenance const& iProv,
                                   ProductProvenanceRetriever const&,
                                   std::set<StoredProductProvenance>& oToKeep);
    TFile file_;
    std::unique_ptr<ROOT::RNTupleWriter> events_;
    std::optional<ROOT::RFieldToken> eventAuxField_;
    std::optional<ROOT::RFieldToken> eventProvField_;
    std::optional<ROOT::RFieldToken> eventSelField_;
    std::optional<ROOT::RFieldToken> branchListField_;

    std::unique_ptr<ROOT::RNTupleWriter> runs_;
    std::optional<ROOT::RFieldToken> runAuxField_;

    std::unique_ptr<ROOT::RNTupleWriter> lumis_;
    std::optional<ROOT::RFieldToken> lumiAuxField_;

    std::unique_ptr<ROOT::RNTupleWriter> parameterSets_;
    std::unique_ptr<ROOT::RNTupleWriter> parentage_;
    std::unique_ptr<ROOT::RNTupleWriter> metaData_;

    std::map<ParentageID, unsigned int> parentageIDs_;
    ProcessHistoryRegistry processHistoryRegistry_;
    std::set<BranchID> branchesWithStoredHistory_;

    IndexIntoFile::EntryNumber_t eventEntryNumber_ = 0LL;
    IndexIntoFile::EntryNumber_t lumiEntryNumber_ = 0LL;
    IndexIntoFile::EntryNumber_t runEntryNumber_ = 0LL;
    IndexIntoFile indexIntoFile_;
    ProductDependencies productDependencies_;

    std::array<std::vector<Product>, NumBranchTypes> products_;
    TClass const* wrapperBaseTClass_;

    ParameterSetID selectorConfig_;
    bool extendSelectorConfig_ = true;
    bool dropMetaData_ = false;
  };
}  // namespace edm
#endif
