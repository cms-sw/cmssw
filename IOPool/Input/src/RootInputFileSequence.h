#ifndef IOPool_Input_RootInputFileSequence_h
#define IOPool_Input_RootInputFileSequence_h

/*----------------------------------------------------------------------

RootInputFileSequence: This is an InputSource. initTheFile tries to open
a file using a list of PFN names constructed from multiple data catalogs
in site-local-config.xml. These are accessed via FileCatalogItem iterator
fileIter_.

----------------------------------------------------------------------*/

#include "InputFile.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace edm {

  class FileCatalogItem;
  class IndexIntoFile;
  class InputFileCatalog;
  class ParameterSetDescription;
  class RootFile;

  class RootInputFileSequence {
  public:
    explicit RootInputFileSequence(ParameterSet const& pset, InputFileCatalog const& catalog);
    virtual ~RootInputFileSequence();

    RootInputFileSequence(RootInputFileSequence const&) = delete;             // Disallow copying and moving
    RootInputFileSequence& operator=(RootInputFileSequence const&) = delete;  // Disallow copying and moving

    bool containedInCurrentFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;
    bool readEvent(EventPrincipal& cache);
    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    bool readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal);
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    bool readRun_(RunPrincipal& runPrincipal);
    void fillProcessBlockHelper_();
    bool nextProcessBlock_(ProcessBlockPrincipal&);
    void readProcessBlock_(ProcessBlockPrincipal&);
    bool skipToItem(RunNumber_t run,
                    LuminosityBlockNumber_t lumi,
                    EventNumber_t event,
                    size_t fileNameHash = 0U,
                    bool currentFileFirst = true);
    std::shared_ptr<ProductRegistry const> fileProductRegistry() const;
    std::shared_ptr<BranchIDListHelper const> fileBranchIDListHelper() const;

    void closeFile();

  protected:
    typedef std::shared_ptr<RootFile> RootFileSharedPtr;
    void initFile(bool skipBadFiles) { initFile_(skipBadFiles); }
    void initTheFile(bool skipBadFiles,
                     bool deleteIndexIntoFile,
                     InputSource* input,
                     char const* inputTypeName,
                     InputType inputType);

    bool skipToItemInNewFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event);
    bool skipToItemInNewFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, size_t fileNameHash);

    bool atFirstFile() const { return fileIter_ == fileIterBegin_; }
    bool atLastFile() const { return fileIter_ + 1 == fileIterEnd_; }
    bool noMoreFiles() const { return fileIter_ == fileIterEnd_; }
    bool noFiles() const { return fileIterBegin_ == fileIterEnd_; }
    size_t sequenceNumberOfFile() const { return fileIter_ - fileIterBegin_; }
    size_t numberOfFiles() const { return fileIterEnd_ - fileIterBegin_; }

    void setAtFirstFile() { fileIter_ = fileIterBegin_; }
    void setAtFileSequenceNumber(size_t offset) { fileIter_ = fileIterBegin_ + offset; }
    void setNoMoreFiles() { fileIter_ = fileIterEnd_; }
    void setAtNextFile() { ++fileIter_; }
    void setAtPreviousFile() { --fileIter_; }

    std::vector<std::string> const& fileNames() const { return fileIter_->fileNames(); }

    std::string const& logicalFileName() const { return fileIter_->logicalFileName(); }
    std::string const& lfn() const { return lfn_; }
    std::vector<FileCatalogItem> const& fileCatalogItems() const;

    std::vector<std::shared_ptr<IndexIntoFile>> const& indexesIntoFiles() const { return indexesIntoFiles_; }
    void setIndexIntoFile(size_t index);
    size_t lfnHash() const { return lfnHash_; }
    bool usedFallback() const { return usedFallback_; }

    std::shared_ptr<RootFile const> rootFile() const { return get_underlying_safe(rootFile_); }
    std::shared_ptr<RootFile>& rootFile() { return get_underlying_safe(rootFile_); }

  private:
    InputFileCatalog const& catalog_;
    std::string lfn_;
    size_t lfnHash_;
    bool usedFallback_;
    edm::propagate_const<std::unique_ptr<std::unordered_multimap<size_t, size_t>>> findFileForSpecifiedID_;
    std::vector<FileCatalogItem>::const_iterator const fileIterBegin_;
    std::vector<FileCatalogItem>::const_iterator const fileIterEnd_;
    std::vector<FileCatalogItem>::const_iterator fileIter_;
    std::vector<FileCatalogItem>::const_iterator fileIterLastOpened_;
    edm::propagate_const<RootFileSharedPtr> rootFile_;
    std::vector<std::shared_ptr<IndexIntoFile>> indexesIntoFiles_;

  private:
    virtual RootFileSharedPtr makeRootFile(std::shared_ptr<InputFile> filePtr) = 0;
    virtual void initFile_(bool skipBadFiles) = 0;
    virtual void closeFile_() = 0;

  };  // class RootInputFileSequence
}  // namespace edm
#endif
