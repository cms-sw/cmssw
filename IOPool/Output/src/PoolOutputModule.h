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

#include <string>
#include "boost/scoped_ptr.hpp"

#include "IOPool/Common/interface/RootServiceChecker.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"

class TTree;
namespace edm {
  class ParameterSet;
  class RootOutputFile;

  class PoolOutputModule : public OutputModule {
  public:
    friend class RootOutputFile;
    explicit PoolOutputModule(ParameterSet const& ps);
    virtual ~PoolOutputModule();
    std::string const& fileName() const {return fileName_;}
    std::string const& logicalFileName() const {return logicalFileName_;}
    int const& compressionLevel() const {return compressionLevel_;}
    int const& basketSize() const {return basketSize_;}
    int const& splitLevel() const {return splitLevel_;}
    int const& treeMaxVirtualSize() const {return treeMaxVirtualSize_;}
    bool const& fastCloning() const {return fastCloning_;}
    bool const& dropMetaData() const {return dropMetaDataForDroppedData_;}

    struct OutputItem {
      class Sorter {
      public:
        explicit Sorter(TTree * tree);
        bool operator() (OutputItem const& lh, OutputItem const& rh) const;
      private:
        std::map<std::string, int> treeMap_;
      };

      OutputItem() : branchDescription_(0), product_(0) {}

      explicit OutputItem(BranchDescription const* bd) :
	branchDescription_(bd), product_(0) {}

      ~OutputItem() {}

      BranchID branchID() const { return branchDescription_->branchID(); }
      std::string const& branchName() const { return branchDescription_->branchName(); }

      BranchDescription const* branchDescription_;
      mutable void const* product_;

      bool operator <(OutputItem const& rh) const {
        return *branchDescription_ < *rh.branchDescription_;
      }
    };

    typedef std::vector<OutputItem> OutputItemList;

    typedef boost::array<OutputItemList, NumBranchTypes> OutputItemListArray;

    OutputItemListArray const& selectedOutputItemList() const {return selectedOutputItemList_;}

  private:
    virtual void openFile(FileBlock const& fb);
    virtual void respondToOpenInputFile(FileBlock const& fb);
    virtual void respondToCloseInputFile(FileBlock const& fb);
    virtual void write(EventPrincipal const& e);
    virtual void writeLuminosityBlock(LuminosityBlockPrincipal const& lb);
    virtual void writeRun(RunPrincipal const& r);

    virtual bool isFileOpen() const;
    virtual bool shouldWeCloseFile() const;
    virtual void doOpenFile();


    virtual void startEndFile();
    virtual void writeFileFormatVersion();
    virtual void writeFileIdentifier();
    virtual void writeFileIndex();
    virtual void writeEventHistory();
    virtual void writeProcessConfigurationRegistry();
    virtual void writeProcessHistoryRegistry();
    virtual void writeModuleDescriptionRegistry();
    virtual void writeParameterSetRegistry();
    virtual void writeProductDescriptionRegistry();
    virtual void writeProductDependencies();
    virtual void writeEntryDescriptions();
    virtual void finishEndFile();

    void fillSelectedItemList(BranchType branchtype, TTree *theTree);

    RootServiceChecker rootServiceChecker_;
    OutputItemListArray selectedOutputItemList_;
    std::string const fileName_;
    std::string const logicalFileName_;
    std::string const catalog_;
    unsigned int const maxFileSize_;
    int const compressionLevel_;
    int const basketSize_;
    int const splitLevel_;
    int const treeMaxVirtualSize_;
    bool fastCloning_;
    bool dropMetaDataForDroppedData_;
    std::string const moduleLabel_;
    int outputFileCount_;
    int inputFileCount_;
    boost::scoped_ptr<RootOutputFile> rootOutputFile_;
  };
}

#endif
