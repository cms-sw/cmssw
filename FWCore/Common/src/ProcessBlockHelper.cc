#include "FWCore/Common/interface/ProcessBlockHelper.h"

#include "DataFormats/Provenance/interface/EventToProcessBlockIndexes.h"
#include "DataFormats/Provenance/interface/StoredProcessBlockHelper.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace edm {

  ProcessBlockHelperBase const* ProcessBlockHelper::topProcessBlockHelper() const { return this; }

  std::vector<std::string> const& ProcessBlockHelper::topProcessesWithProcessBlockProducts() const {
    return processesWithProcessBlockProducts();
  }

  unsigned int ProcessBlockHelper::nProcessesInFirstFile() const { return nProcessesInFirstFile_; }

  std::vector<std::vector<unsigned int>> const& ProcessBlockHelper::processBlockCacheIndices() const {
    return processBlockCacheIndices_;
  }

  std::vector<std::vector<unsigned int>> const& ProcessBlockHelper::nEntries() const { return nEntries_; }

  std::vector<unsigned int> const& ProcessBlockHelper::cacheIndexVectorsPerFile() const {
    return cacheIndexVectorsPerFile_;
  }

  std::vector<unsigned int> const& ProcessBlockHelper::cacheEntriesPerFile() const { return cacheEntriesPerFile_; }

  unsigned int ProcessBlockHelper::processBlockIndex(
      std::string const& processName, EventToProcessBlockIndexes const& eventToProcessBlockIndexes) const {
    for (unsigned int iProcess = 0; iProcess < nProcessesInFirstFile_; ++iProcess) {
      if (processName == processesWithProcessBlockProducts()[iProcess]) {
        return processBlockCacheIndices_[eventToProcessBlockIndexes.index()][iProcess];
      }
    }
    return invalidCacheIndex();
  }

  unsigned int ProcessBlockHelper::outerOffset() const { return outerOffset_; }

  // Modifies the process names and cache indices in the StoredProcessBlockHelper.
  // Part of the dropOnInput. Also part of reordering that forces ProcessBlocks
  // to be read in the same order for all input files.
  bool ProcessBlockHelper::firstFileDropProcessesAndReorderStored(
      StoredProcessBlockHelper& storedProcessBlockHelper,
      std::set<std::string> const& processesToKeep,
      std::vector<unsigned int> const& nEntries,
      std::vector<unsigned int>& finalIndexToStoredIndex) const {
    std::vector<std::string> const& storedProcesses = storedProcessBlockHelper.processesWithProcessBlockProducts();

    bool isModified = false;
    unsigned int finalSize = 0;
    for (auto const& process : storedProcesses) {
      if (processesToKeep.find(process) == processesToKeep.end()) {
        isModified = true;
      } else {
        ++finalSize;
      }
    }
    if (!isModified) {
      return isModified;
    }

    std::vector<std::string> finalProcesses;
    finalProcesses.reserve(finalSize);
    finalIndexToStoredIndex.reserve(finalSize);
    for (unsigned int iStored = 0; iStored < storedProcesses.size(); ++iStored) {
      if (processesToKeep.find(storedProcesses[iStored]) != processesToKeep.end()) {
        finalProcesses.emplace_back(storedProcesses[iStored]);
        finalIndexToStoredIndex.emplace_back(iStored);
      }
    }
    dropProcessesAndReorderStoredImpl(storedProcessBlockHelper, finalProcesses, nEntries, finalIndexToStoredIndex);
    return isModified;
  }

  // Modifies the process names and cache indices in the StoredProcessBlockHelper.
  // Part of the dropOnInput. Also part of reordering that forces ProcessBlocks
  // to be read in the same order for all input files.
  bool ProcessBlockHelper::dropProcessesAndReorderStored(StoredProcessBlockHelper& storedProcessBlockHelper,
                                                         std::set<std::string> const& processesToKeep,
                                                         std::vector<unsigned int> const& nEntries,
                                                         std::vector<unsigned int>& finalIndexToStoredIndex,
                                                         std::vector<std::string> const& firstFileFinalProcesses) const {
    std::vector<std::string> const& storedProcesses = storedProcessBlockHelper.processesWithProcessBlockProducts();

    std::vector<std::string> finalProcesses;
    // Always will allocate enough space (sometimes too much)
    finalProcesses.reserve(storedProcesses.size());
    finalIndexToStoredIndex.reserve(storedProcesses.size());

    // The outer loop here establishes the order
    // Only allows processes that got into finalProcesses for the first file
    for (unsigned int iFirst = 0; iFirst < firstFileFinalProcesses.size(); ++iFirst) {
      // Only includes processes also in storedProcesses
      for (unsigned int iStored = 0; iStored < storedProcesses.size(); ++iStored) {
        std::string const& storedProcess = storedProcesses[iStored];
        if (firstFileFinalProcesses[iFirst] == storedProcess) {
          // Also requires processes have kept ProcessBlock products
          if (processesToKeep.find(storedProcess) != processesToKeep.end()) {
            finalProcesses.emplace_back(storedProcess);
            finalIndexToStoredIndex.emplace_back(iStored);
            break;
          }
        }
      }
    }

    bool isModified = true;
    if (storedProcesses == finalProcesses) {
      isModified = false;
      return isModified;
    }

    dropProcessesAndReorderStoredImpl(storedProcessBlockHelper, finalProcesses, nEntries, finalIndexToStoredIndex);
    return isModified;
  }

  void ProcessBlockHelper::initializeFromPrimaryInput(StoredProcessBlockHelper const& storedProcessBlockHelper) {
    if (!initializedFromInput_) {
      initializedFromInput_ = true;

      assert(processesWithProcessBlockProducts().empty());
      setProcessesWithProcessBlockProducts(storedProcessBlockHelper.processesWithProcessBlockProducts());
      nProcessesInFirstFile_ = processesWithProcessBlockProducts().size();
    }
  }

  void ProcessBlockHelper::fillFromPrimaryInput(StoredProcessBlockHelper const& storedProcessBlockHelper,
                                                std::vector<unsigned int> const& nEntries) {
    // I've written this so it will continue to work even if we someday relax
    // the strict merging requirement in the ProductRegistry (there
    // is a little extra complexity that may be unnecessary...).

    std::vector<std::string> const& storedProcesses = storedProcessBlockHelper.processesWithProcessBlockProducts();
    std::vector<unsigned int> const& storedCacheIndices = storedProcessBlockHelper.processBlockCacheIndices();

    outerOffset_ = processBlockCacheIndices_.size();

    if (nProcessesInFirstFile_ == 0) {
      // There were no ProcessBlock products in the first file. Nothing to do.
      return;
    } else if (!storedProcesses.empty()) {
      // Subsequent input file with ProcessBlock products
      fillFromPrimaryInputWhenNotEmpty(storedProcesses, storedCacheIndices, nEntries);
    } else if (storedProcesses.empty()) {
      // Subsequent input file without ProcessBlock products
      processBlockCacheIndices_.emplace_back(nProcessesInFirstFile_, invalidCacheIndex());
      cacheIndexVectorsPerFile_.push_back(1);
      std::vector<unsigned int> newNEntries(nProcessesInFirstFile_, 0);
      fillEntriesFromPrimaryInput(std::move(newNEntries));
    }
  }

  void ProcessBlockHelper::clearAfterOutputFilesClose() {
    processBlockCacheIndices_.clear();
    nEntries_.clear();
    cacheIndexVectorsPerFile_.clear();
    cacheEntriesPerFile_.clear();
    outerOffset_ = 0;
    cacheIndexOffset_ = 0;
  }

  // Modifies the process names and cache indices in the StoredProcessBlockHelper.
  // Part of the dropOnInput. Also part of reordering that forces ProcessBlocks
  // to be read in the same order for all input files.
  void ProcessBlockHelper::dropProcessesAndReorderStoredImpl(
      StoredProcessBlockHelper& storedProcessBlockHelper,
      std::vector<std::string>& finalProcesses,
      std::vector<unsigned int> const& nEntries,
      std::vector<unsigned int> const& finalIndexToStoredIndex) const {
    std::vector<std::string> const& storedProcesses = storedProcessBlockHelper.processesWithProcessBlockProducts();
    std::vector<unsigned int> const& storedCacheIndices = storedProcessBlockHelper.processBlockCacheIndices();

    assert(!storedProcesses.empty());
    std::vector<unsigned int> newCacheIndices;
    if (!finalProcesses.empty()) {
      // ProcessBlocks are read in the order of storedProcesses and within
      // each process in entry order in the TTree.  This establishes the cache
      // order that the values in storedCacheIndices refer to. The "offset" refers
      // to cache index value of the first ProcessBlock in a TTree.
      std::vector<unsigned int> oldOffsets;
      oldOffsets.reserve(storedProcesses.size());
      unsigned int offset = 0;
      for (auto const& entries : nEntries) {
        oldOffsets.emplace_back(offset);
        offset += entries;
      }

      unsigned int nFinalProcesses = finalProcesses.size();
      std::vector<unsigned int> newOffsets;
      newOffsets.reserve(nFinalProcesses);
      offset = 0;
      for (unsigned int iNew = 0; iNew < nFinalProcesses; ++iNew) {
        newOffsets.emplace_back(offset);
        offset += nEntries[finalIndexToStoredIndex[iNew]];
      }

      unsigned int nOuterLoop = storedCacheIndices.size() / storedProcesses.size();
      assert(nOuterLoop * storedProcesses.size() == storedCacheIndices.size());
      newCacheIndices.reserve(nOuterLoop * nFinalProcesses);
      unsigned int storedOuterOffset = 0;

      for (unsigned int k = 0; k < nOuterLoop; ++k) {
        for (unsigned int j = 0; j < nFinalProcesses; ++j) {
          unsigned int storedIndex = finalIndexToStoredIndex[j];
          unsigned int oldCacheIndex = storedCacheIndices[storedOuterOffset + storedIndex];
          unsigned int newCacheIndex = invalidCacheIndex();
          if (oldCacheIndex != invalidCacheIndex()) {
            newCacheIndex = oldCacheIndex - oldOffsets[storedIndex] + newOffsets[j];
          }
          newCacheIndices.emplace_back(newCacheIndex);
        }
        storedOuterOffset += storedProcesses.size();
      }
    }
    storedProcessBlockHelper.setProcessBlockCacheIndices(std::move(newCacheIndices));
    storedProcessBlockHelper.setProcessesWithProcessBlockProducts(std::move(finalProcesses));
  }

  void ProcessBlockHelper::fillFromPrimaryInputWhenNotEmpty(std::vector<std::string> const& storedProcesses,
                                                            std::vector<unsigned int> const& storedCacheIndices,
                                                            std::vector<unsigned int> const& nEntries) {
    assert(nProcessesInFirstFile_ <= processesWithProcessBlockProducts().size());

    // Calculate a translation from an index into the process names from the first file
    // to an index into the process names from the current file. Note the value will
    // be left invalid if the process name is not found in the current file.
    std::vector<unsigned int> firstFileToStored(nProcessesInFirstFile_, invalidProcessIndex());
    std::vector<unsigned int> newNEntries(nProcessesInFirstFile_, 0);
    for (unsigned int j = 0; j < nProcessesInFirstFile_; ++j) {
      for (unsigned int k = 0; k < storedProcesses.size(); ++k) {
        if (processesWithProcessBlockProducts()[j] == storedProcesses[k]) {
          firstFileToStored[j] = k;
          newNEntries[j] = nEntries[k];
          break;
        }
      }
    }

    // Append the cache indices from the current input file to the
    // cache indices container from the previous files.
    unsigned int nCacheIndexVectors = storedCacheIndices.size() / storedProcesses.size();
    assert(storedProcesses.size() * nCacheIndexVectors == storedCacheIndices.size());
    processBlockCacheIndices_.resize(processBlockCacheIndices_.size() + nCacheIndexVectors);
    unsigned int storedIndex = 0;
    for (unsigned int k = 0; k < nCacheIndexVectors; ++k) {
      processBlockCacheIndices_[outerOffset_ + k].reserve(nProcessesInFirstFile_);
      for (unsigned int j = 0; j < nProcessesInFirstFile_; ++j) {
        unsigned int iStored = firstFileToStored[j];
        if (iStored == invalidProcessIndex()) {
          processBlockCacheIndices_[outerOffset_ + k].push_back(invalidCacheIndex());
        } else {
          unsigned int oldCacheIndex = storedCacheIndices[storedIndex];
          ++storedIndex;
          unsigned int newCacheIndex = invalidCacheIndex();
          if (oldCacheIndex != invalidCacheIndex()) {
            newCacheIndex = oldCacheIndex + cacheIndexOffset_;
          }
          processBlockCacheIndices_[outerOffset_ + k].push_back(newCacheIndex);
        }
      }
    }
    cacheIndexVectorsPerFile_.push_back(nCacheIndexVectors);
    fillEntriesFromPrimaryInput(std::move(newNEntries));
  }

  void ProcessBlockHelper::fillEntriesFromPrimaryInput(std::vector<unsigned int> nEntries) {
    unsigned int entriesThisFile = 0;
    for (auto const& entries : nEntries) {
      entriesThisFile += entries;
    }
    nEntries_.push_back(std::move(nEntries));
    cacheEntriesPerFile_.push_back(entriesThisFile);
    cacheIndexOffset_ += entriesThisFile;
  }

}  // namespace edm
