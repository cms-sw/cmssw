#ifndef IOPool_Input_InputFile_h
#define IOPool_Input_InputFile_h

/*----------------------------------------------------------------------

Holder for an input TFile.
----------------------------------------------------------------------*/
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "TFile.h"
#include "TTree.h"
#include "TTreeCache.h"

#include <map>
#include <string>
#include <vector>

class TObject;

namespace edm::rntuple_temp {
  class InputFile {
  public:
    explicit InputFile(char const* fileName, char const* msg, InputType inputType);
    ~InputFile();

    InputFile(InputFile const&) = delete;             // Disallow copying and moving
    InputFile& operator=(InputFile const&) = delete;  // Disallow copying and moving

    void Close();
    void inputFileOpened(std::string const& logicalFileName,
                         std::string const& inputType,
                         std::string const& moduleName,
                         std::string const& label,
                         std::string const& fid,
                         std::vector<std::string> const& branchNames);
    void eventReadFromFile() const;
    void reportInputRunNumber(unsigned int run) const;
    void reportInputLumiSection(unsigned int run, unsigned int lumi) const;
    static void reportSkippedFile(std::string const& fileName, std::string const& logicalFileName);
    static void reportFallbackAttempt(std::string const& pfn,
                                      std::string const& logicalFileName,
                                      std::string const& errorMessage);
    // reportReadBranches is a per job report, rather than per file report.
    // Nevertheless, it is defined here for convenience.
    static void reportReadBranches();
    static void reportReadBranch(InputType inputType, std::string const& branchname);

    TObject* Get(char const* name) { return file_->Get(name); }
    template<typename T>
    T* Get(char const* name) { return file_->Get<T>(name); }
    std::unique_ptr<TTreeCache> createCacheWithSize(TTree& iTree, unsigned int cacheSize) {
      iTree.SetCacheSize(static_cast<Long64_t>(cacheSize));
      std::unique_ptr<TTreeCache> newCache(dynamic_cast<TTreeCache*>(file_->GetCacheRead(&iTree)));
      file_->SetCacheRead(nullptr, &iTree, TFile::kDoNotDisconnect);
      return newCache;
    }

    class CacheGuard {
    public:
      CacheGuard(TFile* file, TObject* tree, TFileCacheRead* tfcr) : file_(file), tree_(tree) {
        file_->SetCacheRead(tfcr, tree_, TFile::kDoNotDisconnect);
      }
      CacheGuard() = delete;
      CacheGuard(CacheGuard const&) = delete;
      CacheGuard& operator=(CacheGuard const&) = delete;
      CacheGuard(CacheGuard&&) = delete;
      CacheGuard& operator=(CacheGuard&&) = delete;
      ~CacheGuard() { file_->SetCacheRead(nullptr, tree_, TFile::kDoNotDisconnect); }

    private:
      TFile* file_;
      TObject* tree_;
    };
    [[nodiscard]] CacheGuard setCacheReadTemporarily(TFileCacheRead* tfcr, TObject* iTree) {
      return CacheGuard(file_.get(), iTree, tfcr);
    }
    void clearCacheRead(TObject* iTree) { file_->SetCacheRead(nullptr, iTree, TFile::kDoNotDisconnect); }
    void logFileAction(char const* msg, char const* fileName) const;

  private:
    edm::propagate_const<std::unique_ptr<TFile>> file_;
    std::string fileName_;
    JobReport::Token reportToken_;
    InputType inputType_;
  };
}  // namespace edm::rntuple_temp
#endif
