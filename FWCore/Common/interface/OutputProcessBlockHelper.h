#ifndef FWCore_Common_OutputProcessBlockHelper_h
#define FWCore_Common_OutputProcessBlockHelper_h

/** \class edm::OutputProcessBlockHelper

\author W. David Dagenhart, created 28 December, 2020

*/

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

#include <set>
#include <string>
#include <vector>

namespace edm {

  class ProcessBlockHelperBase;

  class OutputProcessBlockHelper {
  public:
    std::vector<std::string> const& processesWithProcessBlockProducts() const {
      return processesWithProcessBlockProducts_;
    }

    void updateAfterProductSelection(std::set<std::string> const& processesWithKeptProcessBlockProducts,
                                     ProcessBlockHelperBase const&);

    void fillCacheIndices(StoredProcessBlockHelper&) const;

    bool productsFromInputKept() const { return productsFromInputKept_; }

    ProcessBlockHelperBase const* processBlockHelper() const { return processBlockHelper_; }

  private:
    // The next two functions are intended to be used only for testing purposes
    friend class TestOneOutput;
    std::vector<unsigned int> const& translateFromStoredIndex() const { return translateFromStoredIndex_; }
    unsigned int nAddedProcesses() const { return nAddedProcesses_; }

    void setStoredProcessOffset(unsigned int nInputProcesses,
                                std::vector<std::vector<unsigned int>> const& nEntries,
                                std::vector<unsigned int>& storedProcessOffset) const;

    void setProcessOffset(unsigned int iFile,
                          unsigned int nInputProcesses,
                          std::vector<std::vector<unsigned int>> const& nEntries,
                          std::vector<unsigned int>& processOffset) const;

    void setStoredFileInProcessOffset(unsigned int iFile,
                                      unsigned int nInputProcesses,
                                      std::vector<std::vector<unsigned int>> const& nEntries,
                                      std::vector<unsigned int>& storedFileInProcessOffset) const;

    // Includes processes with at least one ProcessBlock branch present
    // in the output file
    std::vector<std::string> processesWithProcessBlockProducts_;

    // This will have the value of 0 or 1, except for the SubProcess case.
    // This is incremented to 1 if the current process produces new
    // ProcessBlock products and they are kept by the OutputModule.
    unsigned int nAddedProcesses_ = 0;

    // Translate from the vector of process names in this class to
    // the one in the ProcessBlockHelper
    std::vector<unsigned int> translateFromStoredIndex_;

    // Points to the main ProcessBlockHelper owned by the EventProcessor
    // or SubProcess
    ProcessBlockHelperBase const* processBlockHelper_ = nullptr;

    bool productsFromInputKept_ = false;
  };
}  // namespace edm
#endif
