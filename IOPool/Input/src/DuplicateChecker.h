#ifndef DataFormats_Provenance_DuplicateChecker_h
#define DataFormats_Provenance_DuplicateChecker_h


/*----------------------------------------------------------------------

IOPool/Input/src/DuplicateChecker.h

Used by PoolSource to detect events with
the same process history, run, lumi, and event number.
It is configurable whether it checks for duplicates
within the scope of each single input file or all input
files or does not check for duplicates at all.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"

#include "boost/shared_ptr.hpp"

#include <vector>
#include <set>
#include <string>


namespace edm {

  class ParameterSet;
  class ParameterSetDescription;

  class DuplicateChecker {
  public:

    DuplicateChecker(ParameterSet const& pset);

    void disable();

    void inputFileOpened(
      bool realData,
      IndexIntoFile const& indexIntoFile,
      std::vector<boost::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
      std::vector<boost::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile);

    void inputFileClosed();

    bool noDuplicatesInFile() const { return itIsKnownTheFileHasNoDuplicates_; }

    bool checkDisabled() const {
      return duplicateCheckMode_ == noDuplicateCheck ||
	(duplicateCheckMode_ == checkEachRealDataFile && dataType_ == isSimulation) ||
        disabled_;
    }

    bool isDuplicateAndCheckActive(int index,
                                   RunNumber_t run,
                                   LuminosityBlockNumber_t lumi,
                                   EventNumber_t event,
                                   std::string const& fileName);

    bool checkingAllFiles() const {return checkAllFilesOpened == duplicateCheckMode_;} 

    static void fillDescription(ParameterSetDescription & desc);

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
    std::set<IndexIntoFile::IndexRunLumiEventKey> relevantPreviousEvents_;

    bool itIsKnownTheFileHasNoDuplicates_;

    bool disabled_;
  };
}
#endif
