
#include "FWCore/Framework/interface/MergeableRunProductMetadata.h"

#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/StoredMergeableRunProductMetadata.h"
#include "FWCore/Framework/interface/MergeableRunProductProcesses.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <memory>

namespace edm {

  MergeableRunProductMetadata::
  MergeableRunProductMetadata(MergeableRunProductProcesses const& mergeableRunProductProcesses) :
    mergeableRunProductProcesses_(&mergeableRunProductProcesses),
    metadataForProcesses_(mergeableRunProductProcesses.size()) {
  }

  MergeableRunProductMetadata::~MergeableRunProductMetadata() {
  }

  void MergeableRunProductMetadata::preReadFile() {
    mergeLumisFromIndexIntoFile();
  }

  void MergeableRunProductMetadata::readRun(
    long long inputRunEntry,
    StoredMergeableRunProductMetadata const& inputStoredMergeableRunProductMetadata,
    IndexIntoFileItrHolder const& inputIndexIntoFileItr) {

    unsigned int processIndex {0};
    for (auto & metadataForProcess : metadataForProcesses_) {

      bool valid = true;
      std::vector<LuminosityBlockNumber_t>::const_iterator lumisInRunBeingReadBegin;
      std::vector<LuminosityBlockNumber_t>::const_iterator lumisInRunBeingReadEnd;

      std::string const& processName = mergeableRunProductProcesses_->processesWithMergeableRunProducts()[processIndex];

      if (inputStoredMergeableRunProductMetadata.getLumiContent(inputRunEntry,
                                                                processName,
                                                                valid,
                                                                lumisInRunBeingReadBegin,
                                                                lumisInRunBeingReadEnd)) {

        // This is a reference to the container accumulating the luminosity
        // block numbers for the run entries read associated with the current
        // run being processed that correspond to the luminosity block content
        // for the mergeable run products created in the process.
        std::vector<LuminosityBlockNumber_t>& lumis = metadataForProcess.lumis();

        // In the following, iter1 refers to the lumis associated with run entries already read
        // and iter2 refers to the lumis associated with the current run entry being read.

        bool elementsIn2NotIn1 = false;
        bool elementsIn1NotIn2 = false;
        bool sharedElements = false;

        std::vector<LuminosityBlockNumber_t> temp;
        temp.reserve(lumis.size() + (lumisInRunBeingReadEnd - lumisInRunBeingReadBegin));
        std::vector<LuminosityBlockNumber_t>::const_iterator end1 = lumis.end();
        std::vector<LuminosityBlockNumber_t>::const_iterator end2 = lumisInRunBeingReadEnd;
        for (std::vector<LuminosityBlockNumber_t>::const_iterator iter1 = lumis.begin(),
             iter2 = lumisInRunBeingReadBegin;
             iter1 != end1 || iter2 != end2;) {
          if (iter1 == end1) {
            temp.push_back(*iter2);
            ++iter2;
            elementsIn2NotIn1 = true;
            continue;
          } else if (iter2 == end2) {
            temp.push_back(*iter1);
            ++iter1;
            elementsIn1NotIn2 = true;
            continue;
          } else if (*iter1 < *iter2) {
            temp.push_back(*iter1);
            ++iter1;
            elementsIn1NotIn2 = true;
          } else if (*iter1 > *iter2) {
            temp.push_back(*iter2);
            ++iter2;
            elementsIn2NotIn1 = true;
          } else {
            // they must be equal
            sharedElements = true;
            temp.push_back(*iter1);
            ++iter1;
            ++iter2;
          }
        }
        lumis.swap(temp);
        if (!sharedElements && elementsIn2NotIn1 && elementsIn1NotIn2) {
          metadataForProcess.setMergeDecision(MERGE);
          if (!valid) {
            metadataForProcess.setValid(false);
          }
        } else if (!elementsIn2NotIn1) {
          metadataForProcess.setMergeDecision(IGNORE);
        } else if (!elementsIn1NotIn2) {
          metadataForProcess.setMergeDecision(REPLACE);
          if (!valid) {
            metadataForProcess.setValid(false);
          }
        } else {
          // In this case there is no way to get the correct answer.
          // The result will always be invalid.
          metadataForProcess.setMergeDecision(MERGE);
          metadataForProcess.setValid(false);
        }

      } else {

        metadataForProcess.setMergeDecision(MERGE);
        if (!valid) {
          metadataForProcess.setValid(false);
        }
        metadataForProcess.setUseIndexIntoFile(true);
        if (!gotLumisFromIndexIntoFile_) {
          inputIndexIntoFileItr.getLumisInRun(lumisFromIndexIntoFile_);
          gotLumisFromIndexIntoFile_ = true;
        }
      }
      ++processIndex;
    } // end of loop over processes
  } // end of readRun function

  void MergeableRunProductMetadata::writeLumi(LuminosityBlockNumber_t lumi) {

    if (metadataForProcesses_.empty()) {
      return;
    }
    lumisProcessed_.push_back(lumi);
  }

  void MergeableRunProductMetadata::preWriteRun() {

    if (metadataForProcesses_.empty()) {
      return;
    }

    mergeLumisFromIndexIntoFile();

    // Sort the lumiProcessed vector and ignore the duplicate
    // entries

    // Not sure if this copy is necessary. I'm copying because
    // I am not sure the standard algorithms work on TBB containers.
    // I couldn't find anything saying they did when I searched ...
    std::vector<LuminosityBlockNumber_t> lumisProcessed;
    lumisProcessed.reserve(lumisProcessed_.size());
    for (auto const& lumi : lumisProcessed_) {
      lumisProcessed.push_back(lumi);
    }

    std::sort(lumisProcessed.begin(), lumisProcessed.end());
    auto uniqueEnd = std::unique(lumisProcessed.begin(), lumisProcessed.end());

    for (auto & metadataForProcess : metadataForProcesses_) {

      // Did we process all the lumis in this process that were processed
      // in the process that created the mergeable run products.
      metadataForProcess.setAllLumisProcessed(
        std::includes(lumisProcessed.begin(), uniqueEnd,
                      metadataForProcess.lumis().begin(),
                      metadataForProcess.lumis().end()));
    }
  }

  void MergeableRunProductMetadata::postWriteRun() {

    lumisProcessed_.clear();
    for (auto & metadataForProcess : metadataForProcesses_) {
      metadataForProcess.reset();
    }
  }

  void MergeableRunProductMetadata::
  addEntryToStoredMetadata(StoredMergeableRunProductMetadata& storedMetadata) const {

    if (metadataForProcesses_.empty()) {
      return;
    }

    std::vector<std::string> const& storedProcesses = storedMetadata.processesWithMergeableRunProducts();
    if (storedProcesses.empty()) {
      return;
    }

    unsigned long long beginProcess = storedMetadata.singleRunEntryAndProcesses().size();
    unsigned long long endProcess = beginProcess;


    std::vector<std::string> const& processesWithMergeableRunProducts =
      mergeableRunProductProcesses_->processesWithMergeableRunProducts();

    for (unsigned int storedProcessIndex = 0;
         storedProcessIndex < storedProcesses.size();
         ++storedProcessIndex) {

      // Look for a matching process. It is intentional that no process
      // is added when there is no match. storedProcesses only contains
      // processes which created mergeable run products selected by the
      // output module to be written out. processesWithMergeableRunProducts_
      // only has processes which created mergeable run products that were
      // read from the input data files. Note storedProcesses may be
      // missing processes because the output module dropped products.
      // The other vector may be missing processes because of SubProcesses.
      for (unsigned int transientProcessIndex = 0;
           transientProcessIndex < processesWithMergeableRunProducts.size();
           ++transientProcessIndex) {

        // This string comparison could be optimized away by storing an index mapping in
        // OutputModuleBase calculated once early in a job. (? Given how rare
        // mergeable run products are this optimization may not be worth doing)
        if (processesWithMergeableRunProducts[transientProcessIndex] ==
            storedProcesses[storedProcessIndex]) {

          if (addProcess(storedMetadata,
                         metadataForProcesses_.at(transientProcessIndex),
                         storedProcessIndex,
                         beginProcess,
                         endProcess)) {
            ++endProcess;
          }
          break;
        }
      }
    }
    storedMetadata.singleRunEntries().emplace_back(beginProcess, endProcess);
  }

  bool MergeableRunProductMetadata::
  addProcess(StoredMergeableRunProductMetadata& storedMetadata,
             MetadataForProcess const& metadataForProcess,
             unsigned int storedProcessIndex,
             unsigned long long beginProcess,
             unsigned long long endProcess) const {

    if (metadataForProcess.valid() &&
        metadataForProcess.allLumisProcessed()) {
      return false;
    }

    storedMetadata.allValidAndUseIndexIntoFile() = false;

    unsigned long long iBeginLumi = 0;
    unsigned long long iEndLumi = 0;

    // See if we need to store the set of lumi numbers corresponding
    // to this process and run entry. If they were all processed then
    // we can just get the lumi numbers out of IndexIntoFile and do
    // not need to store them here
    if (!metadataForProcess.allLumisProcessed()) {
      // If we need to store the numbers, then we can check to
      // make sure this does not duplicate the lumi numbers we
      // stored for another process. If we did then we can just
      // just reference same indices and avoid copying a duplicate
      // sequence of lumi numbers. It is sufficient to check the
      // size only. As you go back in the time sequence of processes
      // the only thing that can happen is more lumi numbers appear
      // at steps where a run was only partially processed.
      bool found = false;
      for (unsigned long long kProcess = beginProcess;
           kProcess < endProcess;
           ++kProcess) {
        StoredMergeableRunProductMetadata::SingleRunEntryAndProcess const& storedSingleRunEntryAndProcess =
          storedMetadata.singleRunEntryAndProcesses().at(kProcess);

        if (metadataForProcess.lumis().size() ==
            (storedSingleRunEntryAndProcess.endLumi() - storedSingleRunEntryAndProcess.beginLumi())) {

          iBeginLumi = storedSingleRunEntryAndProcess.beginLumi();
          iEndLumi = storedSingleRunEntryAndProcess.endLumi();
          found = true;
          break;
        }
      }
      if (!found) {
        std::vector<LuminosityBlockNumber_t>& storedLumis = storedMetadata.lumis();
        std::vector<LuminosityBlockNumber_t> const& metdataLumis = metadataForProcess.lumis();
        iBeginLumi = storedLumis.size();
        storedLumis.insert( storedLumis.end(), metdataLumis.begin(), metdataLumis.end() );
        iEndLumi = storedLumis.size();
      }
    }
    storedMetadata.singleRunEntryAndProcesses().emplace_back(iBeginLumi,
                                                             iEndLumi,
                                                             storedProcessIndex,
                                                             metadataForProcess.valid(),
                                                             metadataForProcess.allLumisProcessed());
    return true;
  }

  MergeableRunProductMetadata::MergeDecision
  MergeableRunProductMetadata::getMergeDecision(std::string const& processThatCreatedProduct) const {

    MetadataForProcess const* metadataForProcess = metadataForOneProcess(processThatCreatedProduct);
    if (metadataForProcess) {
      return metadataForProcess->mergeDecision();
    }
    throw Exception(errors::LogicError)
      << "MergeableRunProductMetadata::getMergeDecision could not find process.\n"
      << "It should not be possible for this error to occur.\n"
      << "Contact a Framework developer\n";
    return MERGE;
  }

  bool
  MergeableRunProductMetadata::knownImproperlyMerged(std::string const& processThatCreatedProduct) const {

    MetadataForProcess const* metadataForProcess = metadataForOneProcess(processThatCreatedProduct);
    if (metadataForProcess) {
      return !metadataForProcess->valid();
    }
    return false;
  }

  void MergeableRunProductMetadata::MetadataForProcess::reset() {
    lumis_.clear();
    mergeDecision_ = MERGE;
    useIndexIntoFile_ = false;
    valid_ = true;
    allLumisProcessed_ = false;
  }

  MergeableRunProductMetadata::MetadataForProcess const*
  MergeableRunProductMetadata::metadataForOneProcess(std::string const& processName) const {
    unsigned int processIndex = 0;
    for (auto const& metadataForProcess : metadataForProcesses_) {
      // This string comparison could be optimized away by storing an index in
      // BranchDescription as a transient calculated once early in a job.
      if (getProcessName(processIndex) == processName) {
        return &metadataForProcess;
      }
      ++processIndex;
    }
    return nullptr;
  }

  void MergeableRunProductMetadata::mergeLumisFromIndexIntoFile() {

    for (auto & metadataForProcess : metadataForProcesses_) {
      if (metadataForProcess.useIndexIntoFile()) {
        metadataForProcess.setUseIndexIntoFile(false);

        std::vector<LuminosityBlockNumber_t> temp;
        temp.reserve(metadataForProcess.lumis().size() + lumisFromIndexIntoFile_.size());
        std::vector<LuminosityBlockNumber_t>::const_iterator end1 = metadataForProcess.lumis().end();
        std::vector<LuminosityBlockNumber_t>::const_iterator end2 = lumisFromIndexIntoFile_.end();
        for (std::vector<LuminosityBlockNumber_t>::const_iterator iter1 = metadataForProcess.lumis().begin(),
             iter2 = lumisFromIndexIntoFile_.begin();
             iter1 != end1 || iter2 != end2;) {
          if (iter1 == end1) {
            temp.push_back(*iter2);
            ++iter2;
            continue;
          } else if (iter2 == end2) {
            temp.push_back(*iter1);
            ++iter1;
            continue;
          } else if (*iter1 < *iter2) {
            temp.push_back(*iter1);
            ++iter1;
          } else if (*iter1 > *iter2) {
            temp.push_back(*iter2);
            ++iter2;
          } else {
            // they must be equal
            temp.push_back(*iter1);
            ++iter1;
            ++iter2;
          }
        }
        metadataForProcess.lumis().swap(temp);
      }
    }
    lumisFromIndexIntoFile_.clear();
    gotLumisFromIndexIntoFile_ = false;
  }
}
