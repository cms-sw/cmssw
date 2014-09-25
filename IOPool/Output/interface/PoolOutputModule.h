#ifndef IOPool_Output_PoolOutputModule_h
#define IOPool_Output_PoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// Class PoolOutputModule. Output module to POOL file
//
// Oringinal Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <array>
#include <memory>
#include <string>

#include "IOPool/Common/interface/RootServiceChecker.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/OutputModule.h"

class TTree;
namespace edm {

  class ModuleCallingContext;
  class ParameterSet;
  class RootOutputFile;
  class ConfigurationDescriptions;

  class PoolOutputModule : public one::OutputModule<WatchInputFiles> {
  public:
    enum DropMetaData { DropNone, DropDroppedPrior, DropPrior, DropAll };
    explicit PoolOutputModule(ParameterSet const& ps);
    virtual ~PoolOutputModule();
    PoolOutputModule(PoolOutputModule const&) = delete; // Disallow copying and moving
    PoolOutputModule& operator=(PoolOutputModule const&) = delete; // Disallow copying and moving
    std::string const& fileName() const {return fileName_;}
    std::string const& logicalFileName() const {return logicalFileName_;}
    int const& compressionLevel() const {return compressionLevel_;}
    std::string const& compressionAlgorithm() const {return compressionAlgorithm_;}
    int const& basketSize() const {return basketSize_;}
    int eventAutoFlushSize() const {return eventAutoFlushSize_;}
    int const& splitLevel() const {return splitLevel_;}
    std::string const& basketOrder() const {return basketOrder_;}
    int const& treeMaxVirtualSize() const {return treeMaxVirtualSize_;}
    bool const& overrideInputFileSplitLevels() const {return overrideInputFileSplitLevels_;}
    DropMetaData const& dropMetaData() const {return dropMetaData_;}
    std::string const& catalog() const {return catalog_;}
    std::string const& moduleLabel() const {return moduleLabel_;}
    unsigned int const& maxFileSize() const {return maxFileSize_;}
    int const& inputFileCount() const {return inputFileCount_;}
    int const& whyNotFastClonable() const {return whyNotFastClonable_;}

    std::string const& currentFileName() const;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

    using OutputModule::selectorConfig;

    struct AuxItem {
      AuxItem();
      ~AuxItem() {}
      int basketSize_;
    };
    typedef std::array<AuxItem, NumBranchTypes> AuxItemArray;
    AuxItemArray const& auxItems() const {return auxItems_;}

    struct OutputItem {
      class Sorter {
      public:
        explicit Sorter(TTree* tree);
        bool operator() (OutputItem const& lh, OutputItem const& rh) const;
      private:
        std::shared_ptr<std::map<std::string, int> > treeMap_;
      };

      OutputItem();

      explicit OutputItem(BranchDescription const* bd, int splitLevel, int basketSize);

      ~OutputItem() {}

      BranchID branchID() const { return branchDescription_->branchID(); }
      std::string const& branchName() const { return branchDescription_->branchName(); }

      bool operator <(OutputItem const& rh) const {
        return *branchDescription_ < *rh.branchDescription_;
      }

      BranchDescription const* branchDescription_;
      mutable void const* product_;
      int splitLevel_;
      int basketSize_;
    };

    typedef std::vector<OutputItem> OutputItemList;

    typedef std::array<OutputItemList, NumBranchTypes> OutputItemListArray;

    OutputItemListArray const& selectedOutputItemList() const {return selectedOutputItemList_;}

  protected:
    ///allow inheriting classes to override but still be able to call this method in the overridden version
    virtual bool shouldWeCloseFile() const override;
    virtual void write(EventPrincipal const& e, ModuleCallingContext const*) override;
  private:
    virtual void openFile(FileBlock const& fb) override;
    virtual void respondToOpenInputFile(FileBlock const& fb) override;
    virtual void respondToCloseInputFile(FileBlock const& fb) override;
    virtual void writeLuminosityBlock(LuminosityBlockPrincipal const& lb, ModuleCallingContext const*) override;
    virtual void writeRun(RunPrincipal const& r, ModuleCallingContext const*) override;
    virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) override;
    virtual bool isFileOpen() const override;
    virtual void reallyOpenFile() override;
    virtual void reallyCloseFile() override;
    virtual void beginJob() override;

    void startEndFile();
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

    void fillSelectedItemList(BranchType branchtype, TTree* theInputTree);
    void beginInputFile(FileBlock const& fb);

    RootServiceChecker rootServiceChecker_;
    AuxItemArray auxItems_;
    OutputItemListArray selectedOutputItemList_;
    std::string const fileName_;
    std::string const logicalFileName_;
    std::string const catalog_;
    unsigned int const maxFileSize_;
    int const compressionLevel_;
    std::string const compressionAlgorithm_;
    int const basketSize_;
    int const eventAutoFlushSize_;
    int const splitLevel_;
    std::string basketOrder_;
    int const treeMaxVirtualSize_;
    int whyNotFastClonable_;
    DropMetaData dropMetaData_;
    std::string const moduleLabel_;
    bool initializedFromInput_;
    int outputFileCount_;
    int inputFileCount_;
    unsigned int childIndex_;
    unsigned int numberOfDigitsInIndex_;
    bool overrideInputFileSplitLevels_;
    std::unique_ptr<RootOutputFile> rootOutputFile_;
    std::string statusFileName_;
  };
}

#endif
