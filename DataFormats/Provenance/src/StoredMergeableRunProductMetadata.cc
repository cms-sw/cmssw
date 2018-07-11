#include "DataFormats/Provenance/interface/StoredMergeableRunProductMetadata.h"

namespace edm {

  StoredMergeableRunProductMetadata::StoredMergeableRunProductMetadata() :
    allValidAndUseIndexIntoFile_(true) { }

  StoredMergeableRunProductMetadata::
  StoredMergeableRunProductMetadata(std::vector<std::string> const& processesWithMergeableRunProducts):
    processesWithMergeableRunProducts_(processesWithMergeableRunProducts),
    allValidAndUseIndexIntoFile_(true) { }

  StoredMergeableRunProductMetadata::SingleRunEntry::SingleRunEntry() :
    beginProcess_(0),
    endProcess_(0) { }

  StoredMergeableRunProductMetadata::SingleRunEntry::SingleRunEntry(unsigned long long iBeginProcess,
                                                                    unsigned long long iEndProcess) :
    beginProcess_(iBeginProcess),
    endProcess_(iEndProcess) { }

  StoredMergeableRunProductMetadata::SingleRunEntryAndProcess::SingleRunEntryAndProcess() :
    beginLumi_(0),
    endLumi_(0),
    process_(0),
    valid_(false),
    useIndexIntoFile_(false) { }

  StoredMergeableRunProductMetadata::SingleRunEntryAndProcess::
  SingleRunEntryAndProcess(unsigned long long iBeginLumi,
                           unsigned long long iEndLumi,
                           unsigned int iProcess,
                           bool iValid,
                           bool iUseIndexIntoFile)  :
    beginLumi_(iBeginLumi),
    endLumi_(iEndLumi),
    process_(iProcess),
    valid_(iValid),
    useIndexIntoFile_(iUseIndexIntoFile) { }

  void StoredMergeableRunProductMetadata::optimizeBeforeWrite() {
    if (allValidAndUseIndexIntoFile_) {
      processesWithMergeableRunProducts_.clear();
      singleRunEntries_.clear();
      singleRunEntryAndProcesses_.clear();
      lumis_.clear();
    }
  }

  bool StoredMergeableRunProductMetadata::
  getLumiContent(unsigned long long runEntry,
                 std::string const& process,
                 bool& valid,
                 std::vector<LuminosityBlockNumber_t>::const_iterator & lumisBegin,
                 std::vector<LuminosityBlockNumber_t>::const_iterator & lumisEnd) const {

    valid = true;
    if (allValidAndUseIndexIntoFile_) {
      return false;
    }

    SingleRunEntry const& singleRunEntry = singleRunEntries_.at(runEntry);
    for (unsigned long long j = singleRunEntry.beginProcess(); j < singleRunEntry.endProcess(); ++j) {
      SingleRunEntryAndProcess const& singleRunEntryAndProcess = singleRunEntryAndProcesses_.at(j);
      // This string comparison could be optimized away by storing an index mapping in
      // MergeableRunProductMetadata that gets recalculated each time a new input
      // file is opened
      if (processesWithMergeableRunProducts_.at(singleRunEntryAndProcess.process()) == process) {
        valid = singleRunEntryAndProcess.valid();
        if (singleRunEntryAndProcess.useIndexIntoFile()) {
          return false;
        } else {
          lumisBegin = lumis_.begin() + singleRunEntryAndProcess.beginLumi();
          lumisEnd = lumis_.begin() + singleRunEntryAndProcess.endLumi();
          return true;
        }
      }
    }
    return false;
  }
}
