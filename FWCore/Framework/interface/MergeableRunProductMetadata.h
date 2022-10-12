#ifndef FWCore_Framework_MergeableRunProductMetadata_h
#define FWCore_Framework_MergeableRunProductMetadata_h

/** \class edm::MergeableRunProductMetadata

This class holds information used to decide how to merge together
mergeable run products when multiple run entries with the same
run number and ProcessHistoryID are read from input files
contiguously.

Most of the information here is associated with the current
run being processed. Most of it is cleared when a new run
is started. If multiple runs are being processed concurrently,
then there will be an object instantiated for each concurrent
run. The primary RunPrincipal for the current run owns the
object.

This class gets information from the input file from the
StoredMergeableRunProductMetadata object and IndexIntoFile object.
It uses that information to make the decision how to merge
run products read from data, and puts information into the
StoredMergeableRunProductMetadata written into the output file
to use in later processing steps.

If there are SubProcesses, they use the same object as the top
level process because they share the same input.

There is a TWIKI page on the Framework page of the Software
Guide which explains the details about how this works. There
are significant limitations related to what the Framework does
and does not do managing mergeable run products.

\author W. David Dagenhart, created 9 July, 2018

*/

#include "DataFormats/Provenance/interface/MergeableRunProductMetadataBase.h"
#include "FWCore/Framework/interface/MergeableRunProductProcesses.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include "oneapi/tbb/concurrent_vector.h"

#include <string>
#include <vector>

namespace edm {

  class IndexIntoFileItrHolder;
  class StoredMergeableRunProductMetadata;

  class MergeableRunProductMetadata : public MergeableRunProductMetadataBase {
  public:
    enum MergeDecision { MERGE, REPLACE, IGNORE };

    MergeableRunProductMetadata(MergeableRunProductProcesses const&);

    ~MergeableRunProductMetadata() override;

    // Called each time a new input file is opened
    void preReadFile();

    // Called each time a run entry is read from an input file. This
    // should be called before the run products themselves are read
    // because it sets the decision whether to merge, replace, or ignore
    // run products as they read.
    void readRun(long long inputRunEntry,
                 StoredMergeableRunProductMetadata const& inputStoredMergeableRunProductMetadata,
                 IndexIntoFileItrHolder const& inputIndexIntoFileItr);

    // Called to record which lumis were processed by the current run
    void writeLumi(LuminosityBlockNumber_t lumi);

    void preWriteRun();
    void postWriteRun();

    void addEntryToStoredMetadata(StoredMergeableRunProductMetadata&) const;

    MergeDecision getMergeDecision(std::string const& processThatCreatedProduct) const;

    // If Runs were split on lumi boundaries, but then the files were merged
    // in a way that made it impossible to properly merge run products, this
    // function will return true. The Framework usually does not know
    // enough to detect other cases where there is a merging problem.
    bool knownImproperlyMerged(std::string const& processThatCreatedProduct) const override;

    std::string const& getProcessName(unsigned int index) const {
      return mergeableRunProductProcesses_->getProcessName(index);
    }

    class MetadataForProcess {
    public:
      MetadataForProcess() = default;

      std::vector<LuminosityBlockNumber_t>& lumis() { return lumis_; }
      std::vector<LuminosityBlockNumber_t> const& lumis() const { return lumis_; }

      MergeDecision mergeDecision() const { return mergeDecision_; }
      void setMergeDecision(MergeDecision v) { mergeDecision_ = v; }

      bool valid() const { return valid_; }
      void setValid(bool v) { valid_ = v; }

      bool useIndexIntoFile() const { return useIndexIntoFile_; }
      void setUseIndexIntoFile(bool v) { useIndexIntoFile_ = v; }

      bool allLumisProcessed() const { return allLumisProcessed_; }
      void setAllLumisProcessed(bool v) { allLumisProcessed_ = v; }

      void reset();

    private:
      std::vector<LuminosityBlockNumber_t> lumis_;
      MergeDecision mergeDecision_ = MERGE;
      bool valid_ = true;
      bool useIndexIntoFile_ = false;
      bool allLumisProcessed_ = false;
    };

    MetadataForProcess const* metadataForOneProcess(std::string const& processName) const;

    // The next few functions are only intended to be used in tests.
    std::vector<MetadataForProcess> const& metadataForProcesses() const { return metadataForProcesses_; }

    std::vector<LuminosityBlockNumber_t> const& lumisFromIndexIntoFile() const { return lumisFromIndexIntoFile_; }

    bool gotLumisFromIndexIntoFile() const { return gotLumisFromIndexIntoFile_; }

    oneapi::tbb::concurrent_vector<LuminosityBlockNumber_t> const& lumisProcessed() const { return lumisProcessed_; }

  private:
    void mergeLumisFromIndexIntoFile();

    bool addProcess(StoredMergeableRunProductMetadata& storedMetadata,
                    MetadataForProcess const& metadataForProcess,
                    unsigned int storedProcessIndex,
                    unsigned long long beginProcess,
                    unsigned long long endProcess) const;

    MergeableRunProductProcesses const* mergeableRunProductProcesses_;

    // This vector has an entry for each process that has a
    // mergeable run product in the input. It has an exact one to
    // correspondence with mergeableRunProductProcesses_ and
    // is in the same order.
    std::vector<MetadataForProcess> metadataForProcesses_;

    std::vector<LuminosityBlockNumber_t> lumisFromIndexIntoFile_;
    bool gotLumisFromIndexIntoFile_ = false;

    oneapi::tbb::concurrent_vector<LuminosityBlockNumber_t> lumisProcessed_;
  };
}  // namespace edm
#endif
