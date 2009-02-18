#ifndef DataFormats_Provenance_DuplicateChecker_h
#define DataFormats_Provenance_DuplicateChecker_h


/*----------------------------------------------------------------------

IOPool/Input/src/DuplicateChecker.h

Used by PoolSource to detect events with
the same run, lumi, and event number.  It is configurable
whether it checks for duplicates within the scope
of each single input file or all input files or
not at all.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"

#include "boost/shared_ptr.hpp"

#include <vector>
#include <set>
#include <string>


namespace edm {

  class ParameterSet;

  class DuplicateChecker {
  public:

    DuplicateChecker(ParameterSet const& pset);

    void inputFileOpened(
      bool realData,
      FileIndex const& fileIndex,
      std::vector<boost::shared_ptr<FileIndex> > const& fileIndexes,
      std::vector<boost::shared_ptr<FileIndex> >::size_type currentFileIndex);

    void inputFileClosed();

    bool fastCloningOK() const;

    bool isDuplicateAndCheckActive(EventID const& eventID,
                                   LuminosityBlockNumber_t const& lumi,
                                   std::string const& fileName);

  private:

    enum DuplicateCheckMode { noDuplicateCheck, checkEachFile, checkEachRealDataFile, checkAllFilesOpened };

    DuplicateCheckMode duplicateCheckMode_;

    enum DataType { isRealData, isSimulation, unknown };

    DataType dataType_;

    // If checking the entire input for duplicates, then this holds
    // events from previous files that duplicate events in the
    // the current file.  Plus it holds events that have been already
    // processed in the current file.  It is not used if there are
    // no duplicates or duplicate checking has been disabled.
    std::set<FileIndex::Element> relevantPreviousEvents_;

    bool itIsKnownTheFileHasNoDuplicates_;
  };
}
#endif
