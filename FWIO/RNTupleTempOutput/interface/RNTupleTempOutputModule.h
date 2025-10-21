#ifndef FWIO_RNTupleTempOutput_RNTupleTempOutputModule_h
#define FWIO_RNTupleTempOutput_RNTupleTempOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// Class RNTupleTempOutputModule. Output module to POOL file
//
// Oringinal Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <array>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <regex>

#include "IOPool/Common/interface/RootServiceChecker.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "DataFormats/Provenance/interface/ProductDependencies.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

namespace edm {

  class EDGetToken;
  class ModuleCallingContext;
  class ParameterSet;
  class ConfigurationDescriptions;
  class ProductProvenanceRetriever;
}  // namespace edm
namespace edm::rntuple_temp {
  class RootOutputFile;

  class RNTupleTempOutputModule : public one::OutputModule<WatchInputFiles> {
  public:
    enum DropMetaData { DropNone, DropDroppedPrior, DropPrior, DropAll };
    struct Optimizations {
      unsigned long long approxZippedClusterSize;
      unsigned long long maxUnzippedClusterSize;
      unsigned long long initialUnzippedPageSize;
      unsigned long long maxUnzippedPageSize;
      unsigned long long pageBufferBudget;
      bool useBufferedWrite;
      bool useDirectIO;
    };
    explicit RNTupleTempOutputModule(ParameterSet const& ps);
    ~RNTupleTempOutputModule() override;
    RNTupleTempOutputModule(RNTupleTempOutputModule const&) = delete;             // Disallow copying and moving
    RNTupleTempOutputModule& operator=(RNTupleTempOutputModule const&) = delete;  // Disallow copying and moving
    std::string const& fileName() const { return fileName_; }
    std::string const& logicalFileName() const { return logicalFileName_; }
    int compressionLevel() const { return compressionLevel_; }
    std::string const& compressionAlgorithm() const { return compressionAlgorithm_; }
    Optimizations const& optimizations() const { return optimizations_; }
    bool mergeJob() const { return mergeJob_; }
    DropMetaData const& dropMetaData() const { return dropMetaData_; }
    std::string const& catalog() const { return catalog_; }
    std::string const& moduleLabel() const { return moduleLabel_; }
    unsigned int maxFileSize() const { return maxFileSize_; }
    int inputFileCount() const { return inputFileCount_; }

    std::string const& currentFileName() const;

    static void fillDescription(ParameterSetDescription& desc);
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

    using OutputModule::selectorConfig;

    struct AuxItem {
      AuxItem();
      ~AuxItem() {}
      int basketSize_;
    };
    using AuxItemArray = std::array<AuxItem, numberOfRunLumiEventProductTrees>;
    AuxItemArray const& auxItems() const { return auxItems_; }

    struct OutputItem {
      explicit OutputItem(ProductDescription const* bd, EDGetToken const& token, bool streamerProduct);

      BranchID branchID() const { return productDescription_->branchID(); }
      std::string const& branchName() const { return productDescription_->branchName(); }

      bool operator<(OutputItem const& rh) const { return *productDescription_ < *rh.productDescription_; }

      ProductDescription const* productDescription() const { return productDescription_; }
      EDGetToken token() const { return token_; }
      void const* const product() const { return product_; }
      void const*& product() { return product_; }
      void const** productPtr() { return &product_; }
      void setProduct(void const* iProduct) { product_ = iProduct; }

      bool streamerProduct() const { return streamerProduct_; }

    private:
      ProductDescription const* productDescription_;
      EDGetToken token_;
      void const* product_;
      bool streamerProduct_;
    };

    using OutputItemList = std::vector<OutputItem>;

    struct AliasForBranch {
      AliasForBranch(std::string const& iBranchName, std::string const& iAlias)
          : branch_{convert(iBranchName)}, alias_{iAlias} {}

      bool match(std::string const& iBranchName) const;
      std::regex convert(std::string const& iGlobBranchExpression) const;

      std::regex branch_;
      std::string alias_;
    };

    struct SetStreamerForDataProduct {
      SetStreamerForDataProduct(std::string const& iName, bool iUseStreamer)
          : branch_(convert(iName)), useStreamer_(iUseStreamer) {}
      bool match(std::string const& iName) const;
      std::regex convert(std::string const& iGlobBranchExpression) const;

      std::regex branch_;
      bool useStreamer_;
    };

    std::vector<OutputItemList> const& selectedOutputItemList() const { return selectedOutputItemList_; }

    std::vector<OutputItemList>& selectedOutputItemList() { return selectedOutputItemList_; }

    ProductDependencies const& productDependencies() const { return productDependencies_; }

    std::vector<AliasForBranch> const& aliasForBranches() const { return aliasForBranches_; }

    std::vector<std::string> const& noSplitSubFields() const { return noSplitSubFields_; }
    bool allProductsUseStreamer() const { return allProductsUseStreamer_; }

  protected:
    ///allow inheriting classes to override but still be able to call this method in the overridden version
    bool shouldWeCloseFile() const override;
    void write(EventForOutput const& e) override;

    virtual std::pair<std::string, std::string> physicalAndLogicalNameForNewFile();
    virtual void doExtrasAfterCloseFile();

  private:
    void preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                      ModuleCallingContext const& iModuleCallingContext,
                                      Principal const& iPrincipal) const noexcept override;

    void openFile(FileBlock const& fb) override;
    void respondToOpenInputFile(FileBlock const& fb) override;
    void respondToCloseInputFile(FileBlock const& fb) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) override;
    void writeRun(RunForOutput const&) override;
    void writeProcessBlock(ProcessBlockForOutput const&) override;
    bool isFileOpen() const override;
    void reallyOpenFile();
    void reallyCloseFile() override;
    void beginJob() override;
    void initialRegistry(edm::ProductRegistry const& iReg) override;

    void setProcessesWithSelectedMergeableRunProducts(std::set<std::string> const&) override;

    using BranchParents = std::map<BranchID, std::set<ParentageID>>;
    void updateBranchParentsForOneBranch(ProductProvenanceRetriever const* provRetriever, BranchID const& branchID);
    void updateBranchParents(EventForOutput const& e);
    void fillDependencyGraph();

    void startEndFile();
    void writeMetaData();

    void writeParameterSetRegistry();
    void writeParentageRegistry();
    void finishEndFile();

    void fillSelectedItemList(BranchType branchtype, std::string const& processName, OutputItemList&);
    void beginInputFile(FileBlock const& fb);

    RootServiceChecker rootServiceChecker_;
    AuxItemArray auxItems_;
    std::vector<OutputItemList> selectedOutputItemList_;
    std::vector<AliasForBranch> aliasForBranches_;
    std::unique_ptr<edm::ProductRegistry const> reg_;
    std::string const fileName_;
    std::string const logicalFileName_;
    std::string const catalog_;
    unsigned int const maxFileSize_;
    int const compressionLevel_;
    std::string const compressionAlgorithm_;
    Optimizations const optimizations_;
    DropMetaData dropMetaData_;
    std::string const moduleLabel_;
    bool initializedFromInput_;
    int outputFileCount_;
    int inputFileCount_;
    BranchParents branchParents_;
    ProductDependencies productDependencies_;
    std::vector<BranchID> producedBranches_;
    bool overrideInputFileSplitLevels_;
    bool mergeJob_;
    edm::propagate_const<std::unique_ptr<RootOutputFile>> rootOutputFile_;
    std::string statusFileName_;
    std::string overrideGUID_;
    std::vector<std::string> processesWithSelectedMergeableRunProducts_;

    std::vector<std::string> noSplitSubFields_;
    std::vector<SetStreamerForDataProduct> overrideStreamer_;
    bool allProductsUseStreamer_;
  };
}  // namespace edm::rntuple_temp

#endif
