#ifndef DataFormats_Provenance_StoredMergeableRunProductMetadata_h
#define DataFormats_Provenance_StoredMergeableRunProductMetadata_h

/** \class edm::StoredMergeableRunProductMetadata

This class holds information used to decide how to merge together
run products when multiple run entries with the same run number
and ProcessHistoryID are read from input files contiguously. This
class is persistent and stores the information that needs to be
remembered from one process to the next. Most of the work related
to this decision is performed by the class MergeableRunProductMetadata.
The main purpose of this class is to hold the information that
needs to be persistently stored. PoolSource and PoolOutputModule
interface with this class to read and write it.

Note that the information is not stored for each product.
The information is stored for each run entry in Run TTree
in the input file and also for each process in which at least
one mergeable run product was selected to be written to the
output file. It is not necessary to save information
for each product individually, it will be the same for every
product created in the same process and in the same run entry.

The main piece of information stored is the list of luminosity
block numbers processed when the product was created. Often,
this list can be obtained from the IndexIntoFile and we do not
need to duplicate this information here and so as an optimization
we don't. There are also cases where we can detect that the merging
has created invalid run products where part of the content
has probably been double counted. We save a value to record
this problem.

To improve performance, the data structure has been flattened
into 4 vectors instead of containing a vector containing vectors
containing vectors.

When the user of this class fails to find a run entry with a
particular process, the assumption should be made that the lumi
numbers are in IndexIntoFile and valid.

Another optimization is that if in all cases the lumi numbers
can be obtained from IndexIntoFile and are valid, then all
the vectors are cleared and a boolean value is set to indicate
this.

\author W. David Dagenhart, created 23 May, 2018

*/

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include <string>
#include <vector>

namespace edm {

  class StoredMergeableRunProductMetadata {
  public:

    // This constructor exists for ROOT I/O
    StoredMergeableRunProductMetadata();

    // This constructor is used when creating a new object
    // each time an output file is opened.
    StoredMergeableRunProductMetadata(std::vector<std::string> const& processesWithMergeableRunProducts);

    std::vector<std::string> const& processesWithMergeableRunProducts() const {
      return processesWithMergeableRunProducts_;
    }

    class SingleRunEntry {
    public:

      SingleRunEntry();
      SingleRunEntry(unsigned long long iBeginProcess, unsigned long long iEndProcess);

      unsigned long long beginProcess() const { return beginProcess_; }
      unsigned long long endProcess() const { return endProcess_; }

    private:

      // indexes into singleRunEntryAndProcesses_ for a single run entry
      unsigned long long beginProcess_;
      unsigned long long endProcess_;
    };

    class SingleRunEntryAndProcess {
    public:

      SingleRunEntryAndProcess();
      SingleRunEntryAndProcess(unsigned long long iBeginLumi,
                               unsigned long long iEndLumi,
                               unsigned int iProcess,
                               bool iValid,
                               bool iUseIndexIntoFile);


      unsigned long long beginLumi() const { return beginLumi_; }
      unsigned long long endLumi() const { return endLumi_; }

      unsigned int process() const { return process_; }

      bool valid() const { return valid_; }
      bool useIndexIntoFile() const { return useIndexIntoFile_; }

    private:

      // indexes into lumis_ for products created in one process and
      // written into a single run entry.
      unsigned long long beginLumi_;
      unsigned long long endLumi_;

      // index into processesWithMergeableRunProducts_
      unsigned int process_;

      // If false this indicates the way files were split and merged
      // has created run products that are invalid and probably
      // double count some of their content.
      bool valid_;

      // If true the lumi numbers can be obtained from IndexIntoFile
      // and are not stored in the vector named lumis_
      bool useIndexIntoFile_;
    };

    // These four functions are called by MergeableRunProductMetadata which
    // fills the vectors.
    std::vector<SingleRunEntry>& singleRunEntries() { return singleRunEntries_; }
    std::vector<SingleRunEntryAndProcess>& singleRunEntryAndProcesses() { return singleRunEntryAndProcesses_; }
    std::vector<LuminosityBlockNumber_t>& lumis() { return lumis_; }
    bool& allValidAndUseIndexIntoFile() { return allValidAndUseIndexIntoFile_; }

    // Called by RootOutputFile immediately before writing the object
    // when an output file is closed.
    void optimizeBeforeWrite();

    bool getLumiContent(unsigned long long runEntry,
                        std::string const& process,
                        bool& valid,
                        std::vector<LuminosityBlockNumber_t>::const_iterator & lumisBegin,
                        std::vector<LuminosityBlockNumber_t>::const_iterator & lumisEnd) const;

  private:

    std::vector<std::string> processesWithMergeableRunProducts_;
    std::vector<SingleRunEntry> singleRunEntries_;  // index is the run entry
    std::vector<SingleRunEntryAndProcess> singleRunEntryAndProcesses_;
    std::vector<LuminosityBlockNumber_t> lumis_;
    bool allValidAndUseIndexIntoFile_;
  };
}
#endif
