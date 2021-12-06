#include "FWCore/Common/interface/OutputProcessBlockHelper.h"

#include "DataFormats/Provenance/interface/StoredProcessBlockHelper.h"
#include "FWCore/Common/interface/ProcessBlockHelperBase.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace edm {

  void OutputProcessBlockHelper::updateAfterProductSelection(
      std::set<std::string> const& processesWithKeptProcessBlockProducts,
      ProcessBlockHelperBase const& processBlockHelper) {
    processBlockHelper_ = &processBlockHelper;

    // Copy the list of processes with ProcessBlock products from the EventProcessor or SubProcess,
    // except remove any processes where the output module calling this has dropped all ProcessBlock
    // products. We want to maintain the same order and only remove elements. Fill a vector that can
    // translate from one process list to the other.
    assert(processesWithProcessBlockProducts_.empty());
    unsigned int iProcess = 0;
    for (auto const& process : processBlockHelper.processesWithProcessBlockProducts()) {
      if (processesWithKeptProcessBlockProducts.find(process) != processesWithKeptProcessBlockProducts.end()) {
        processesWithProcessBlockProducts_.emplace_back(process);
        translateFromStoredIndex_.emplace_back(iProcess);
      }
      ++iProcess;
    }

    for (auto const& addedProcess : processBlockHelper.addedProcesses()) {
      // count new processes producing ProcessBlock products that are
      // kept by the OutputModule. There can be at most 1 except
      // in the case of SubProcesses.
      if (std::find(processesWithProcessBlockProducts_.begin(),
                    processesWithProcessBlockProducts_.end(),
                    addedProcess) != processesWithProcessBlockProducts_.end()) {
        ++nAddedProcesses_;
      }
    }

    // Determine if any ProcessBlock product from the input file is kept by the output module.
    // Do this by looking for a process name on both the list of processes with ProcessBlock
    // products kept by the output module and process names from the input with ProcessBlock
    // products.
    productsFromInputKept_ =
        std::find_first_of(processesWithProcessBlockProducts_.begin(),
                           processesWithProcessBlockProducts_.end(),
                           processBlockHelper.topProcessesWithProcessBlockProducts().begin(),
                           processBlockHelper.topProcessesWithProcessBlockProducts().begin() +
                               processBlockHelper.nProcessesInFirstFile()) != processesWithProcessBlockProducts_.end();
  }

  void OutputProcessBlockHelper::fillCacheIndices(StoredProcessBlockHelper& storedProcessBlockHelper) const {
    // The stored cache indices are the ones we want to fill.
    // This will get written to the output file.
    // Note for output the vector of vectors is flattened into a single vector
    std::vector<unsigned int> storedCacheIndices;

    // Number of processes in StoredProcessBlockHelper.
    unsigned int nStoredProcesses = storedProcessBlockHelper.processesWithProcessBlockProducts().size();

    if (!productsFromInputKept_) {
      // This is really simple if we are not keeping any ProcessBlock products
      // from the input file. Deal with that special case first.
      // Except for the special case of a SubProcess, nStoredProcesses will be 1.
      assert(nAddedProcesses_ == nStoredProcesses);
      storedCacheIndices.reserve(nStoredProcesses);
      for (unsigned int i = 0; i < nStoredProcesses; ++i) {
        storedCacheIndices.push_back(i);
      }
      storedProcessBlockHelper.setProcessBlockCacheIndices(std::move(storedCacheIndices));
      return;
    }

    // Cache indices of the main ProcessBlockHelper we use as input. This
    // ProcessBlockHelper is owned by the EventProcessor.
    std::vector<std::vector<unsigned int>> const& cacheIndices = processBlockHelper_->processBlockCacheIndices();

    // We need to convert the cache indices in the ProcessBlockHelper to have different values when
    // put in the StoredProcessBlockHelper. The values are not the same because the ProcessBlocks are
    // not read in the same order in this process as compared to the next process which will read
    // the output file that is being written (the read order is the same as the order the cache
    // objects are placed in the cache vectors). In this process, they are ordered first by input file,
    // second by process and last by TTree entry. In the next process, this output file will be read
    // as a single input file. The ProcessBlocks are read in process order (this will be a subset
    // of the process list in ProcessBlockHelper, maybe smaller or maybe the same), and finally in
    // order of entry in the TTree. This conversion is done by subtracting and adding some
    // offsets and a lot of the following code involves calculating these offsets to do the conversion.

    // We will need the info in these to calculate the offsets
    std::vector<unsigned int> const& cacheIndexVectorsPerFile = processBlockHelper_->cacheIndexVectorsPerFile();
    std::vector<unsigned int> const& cacheEntriesPerFile = processBlockHelper_->cacheEntriesPerFile();
    std::vector<std::vector<unsigned int>> const& nEntries = processBlockHelper_->nEntries();

    assert(!cacheIndices.empty());
    // Count processes in the input file with saved ProcessBlock products in the output
    unsigned int nInputProcesses = 0;
    for (unsigned int iStoredProcess = 0; iStoredProcess < nStoredProcesses; ++iStoredProcess) {
      // The existing cache indices in processBlockHelper include only indices
      // corresponding to the processes in the input files. If there are more, then
      // they correspond to current process (and there only will be more if some
      // of the newly produced ProcessBlock products are going to be kept).
      // There will be at most 1 added process except in the case of SubProcesses.
      if (translateFromStoredIndex_[iStoredProcess] < cacheIndices[0].size()) {
        ++nInputProcesses;
      }
    }

    // The following are the 4 offsets. The first two are defined relative to the
    // cache sequence in this process. The second two offsets are defined relative
    // to the cache sequence when the output file we are writing is read.

    // 1. Total number of cache entries in all input files prior to the current input file
    unsigned int fileOffset = 0;

    // 2. For each process, the total number of cache entries in processes in the current
    // input file and before the process
    std::vector<unsigned int> processOffset(nInputProcesses, 0);

    // 3. For each input process with ProcessBlock products stored by this
    // output module, the total number of cache entries in earlier input processes
    // that have ProcessBlock products stored by this output module.
    // Summed over all input files and including only processes in StoredProcessBlockHelper.
    // Plus an extra element at the end that includes all entries in all such processes.
    assert(!nEntries.empty());
    std::vector<unsigned int> storedProcessOffset(nInputProcesses + 1, 0);

    // 4. For each process with ProcessBlock products stored by this output module,
    // the total number of cache entries in that process in all input files before
    // the current input file.
    std::vector<unsigned int> storedFileInProcessOffset(nInputProcesses, 0);

    setStoredProcessOffset(nInputProcesses, nEntries, storedProcessOffset);

    storedCacheIndices.reserve(cacheIndices.size() * nStoredProcesses);

    unsigned int iFile = 0;
    unsigned int innerVectorsCurrentFile = 0;

    // In ProcessBlockHelper, there is a vector which contains vectors
    // of cache indices. Iterate over the inner vectors.
    for (auto const& innerVectorOfCacheIndices : cacheIndices) {
      // The inner vectors are associated with input files. Several contiguous
      // inner vectors can be associated with the same input file. Check to
      // see if we have crossed an input file boundary and set the file
      // index, iFile, at the next file associated with at least
      // one inner vector if necessary.
      while (innerVectorsCurrentFile == cacheIndexVectorsPerFile[iFile]) {
        // Sum cache entries for all files before the current file in
        // ProcessBlockHelper
        fileOffset += cacheEntriesPerFile[iFile];
        ++iFile;
        innerVectorsCurrentFile = 0;
      }
      if (innerVectorsCurrentFile == 0) {
        // Call these when the input file changes
        setProcessOffset(iFile, nInputProcesses, nEntries, processOffset);
        setStoredFileInProcessOffset(iFile, nInputProcesses, nEntries, storedFileInProcessOffset);
      }
      ++innerVectorsCurrentFile;

      assert(nInputProcesses + nAddedProcesses_ == nStoredProcesses);

      // Really fill the cache indices that will be stored in the output file in this loop
      for (unsigned int iStoredProcess = 0; iStoredProcess < nStoredProcesses; ++iStoredProcess) {
        unsigned int storedCacheIndex = ProcessBlockHelperBase::invalidCacheIndex();
        if (iStoredProcess < nInputProcesses) {
          unsigned int cacheIndex = innerVectorOfCacheIndices[translateFromStoredIndex_[iStoredProcess]];
          if (cacheIndex != ProcessBlockHelperBase::invalidCacheIndex()) {
            // The offsets count in the cache sequence to the first entry in
            // the current input file and process
            unsigned int inputOffset = fileOffset + processOffset[iStoredProcess];
            unsigned int storedOffset = storedProcessOffset[iStoredProcess] + storedFileInProcessOffset[iStoredProcess];
            storedCacheIndex = cacheIndex - inputOffset + storedOffset;
          }
        } else {
          // This corresponds to the current process if it has newly produced
          // ProcessBlock products (plus possibly SubProcesses).
          storedCacheIndex = storedProcessOffset[nInputProcesses] + iStoredProcess - nInputProcesses;
        }
        storedCacheIndices.push_back(storedCacheIndex);
      }
    }
    storedProcessBlockHelper.setProcessBlockCacheIndices(std::move(storedCacheIndices));
  }

  void OutputProcessBlockHelper::setStoredProcessOffset(unsigned int nInputProcesses,
                                                        std::vector<std::vector<unsigned int>> const& nEntries,
                                                        std::vector<unsigned int>& storedProcessOffset) const {
    unsigned int iStored = 0;
    for (auto& offset : storedProcessOffset) {
      offset = 0;
      // loop over earlier processes
      for (unsigned int jStored = 0; jStored < iStored; ++jStored) {
        unsigned int indexInEventProcessor = translateFromStoredIndex_[jStored];
        // loop over input files
        for (auto const& entries : nEntries) {
          assert(indexInEventProcessor < entries.size());
          offset += entries[indexInEventProcessor];
        }
      }
      ++iStored;
    }
  }

  void OutputProcessBlockHelper::setProcessOffset(unsigned int iFile,
                                                  unsigned int nInputProcesses,
                                                  std::vector<std::vector<unsigned int>> const& nEntries,
                                                  std::vector<unsigned int>& processOffset) const {
    unsigned int iStored = 0;
    for (auto& offset : processOffset) {
      offset = 0;
      unsigned int iProcess = translateFromStoredIndex_[iStored];
      for (unsigned int jProcess = 0; jProcess < iProcess; ++jProcess) {
        offset += nEntries[iFile][jProcess];
      }
      ++iStored;
    }
  }

  void OutputProcessBlockHelper::setStoredFileInProcessOffset(
      unsigned int iFile,
      unsigned int nInputProcesses,
      std::vector<std::vector<unsigned int>> const& nEntries,
      std::vector<unsigned int>& storedFileInProcessOffset) const {
    unsigned int iStored = 0;
    for (auto& offset : storedFileInProcessOffset) {
      offset = 0;
      unsigned int indexInEventProcessor = translateFromStoredIndex_[iStored];
      // loop over input files before current input file
      for (unsigned int jFile = 0; jFile < iFile; ++jFile) {
        assert(indexInEventProcessor < nEntries[jFile].size());
        offset += nEntries[jFile][indexInEventProcessor];
      }
      ++iStored;
    }
  }

}  // namespace edm
