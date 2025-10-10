#ifndef FWIO_RNTupleTempOutput_RootOutputRNTuple_h
#define FWIO_RNTupleTempOutput_RootOutputRNTuple_h

/*----------------------------------------------------------------------

RootOutputRNTuple.h // used by ROOT output modules

----------------------------------------------------------------------*/

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "ROOT/RNTuple.hxx"
#include "ROOT/RNTupleWriter.hxx"

class TFile;

namespace edm {

  class RootOutputRNTuple {
  public:
    struct Config {
      enum class CompressionAlgos { kLZMA, kZSTD, kZLIB, kLZ4 };

      std::vector<std::string> doNotSplitSubFields;
      CompressionAlgos compressionAlgo = CompressionAlgos::kZSTD;
      int compressionLevel = 4;
      unsigned long long approxZippedClusterSize;
      unsigned long long maxUnzippedClusterSize;
      unsigned long long initialUnzippedPageSize;
      unsigned long long maxUnzippedPageSize;
      unsigned long long pageBufferBudget;
      bool useBufferedWrite;
      bool useDirectIO;
    };

    RootOutputRNTuple(std::shared_ptr<TFile> filePtr,
                      BranchType const& branchType,
                      std::string const& processName = std::string());

    ~RootOutputRNTuple() {}

    RootOutputRNTuple(RootOutputRNTuple const&) = delete;             // Disallow copying and moving
    RootOutputRNTuple& operator=(RootOutputRNTuple const&) = delete;  // Disallow copying and moving

    template <typename T>
    void addAuxiliary(std::string const& branchName, T const** pAux, int bufSize) {
      assert(model_);
      auto field = std::make_unique<ROOT::RField<T>>(branchName);
      model_->AddField(std::move(field));
      auxBranches_.push_back(model_->GetToken(branchName));
      auxBranchPointers_.push_back(reinterpret_cast<void**>(const_cast<T**>(pAux)));
    }

    bool isValid() const;

    void addField(std::string const& branchName,
                  std::string const& className,
                  void const** pProd,
                  int splitLevel,
                  int basketSize,
                  bool produced);

    void fill();

    void finishInitialization(Config const& config);

    std::string const& name() const { return name_; }

    void setEntries() {}

    void close();

    void optimizeBaskets(ULong64_t size) {}

    void setAutoFlush(Long64_t size) {}

  private:
    // We use bare pointers for pointers to some ROOT entities.
    // Root owns them and uses bare pointers internally.
    // Therefore, using smart pointers here will do no good.
    edm::propagate_const<std::shared_ptr<TFile>> filePtr_;
    std::string name_;
    std::unique_ptr<ROOT::RNTupleModel> model_;
    edm::propagate_const<std::unique_ptr<ROOT::RNTupleWriter>> writer_;

    std::vector<ROOT::RFieldToken> producedBranches_;
    std::vector<void**> producedBranchPointers_;
    std::vector<ROOT::RFieldToken> auxBranches_;
    std::vector<void**> auxBranchPointers_;
  };
}  // namespace edm
#endif
