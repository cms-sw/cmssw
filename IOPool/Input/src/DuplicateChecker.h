#ifndef DataFormats_Provenance_DuplicateChecker_h
#define DataFormats_Provenance_DuplicateChecker_h


/*----------------------------------------------------------------------

IOPool/Input/src/DuplicateChecker.h

Used by PoolSource to detect events with
the same run and event number.  It is configurable
whether it checks for duplicates within the scope
of each single input file or all input files or
not at all.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

#include <set>
#include <string>

namespace edm {

  class ParameterSet;
  class FileIndex;

  class DuplicateChecker {
  public:

    DuplicateChecker(ParameterSet const& pset);

    void init(bool realData,
              FileIndex const& fileIndex);

    void inputFileClosed();

    void rewind();

    bool isDuplicateAndCheckActive(EventID const& eventID,
                                   LuminosityBlockNumber_t const& lumi,
                                   std::string const& fileName);

  private:

    enum DuplicateCheckMode { noDuplicateCheck, checkEachFile, checkEachRealDataFile, checkAllFilesOpened };

    DuplicateCheckMode duplicateCheckMode_;

    enum DataType { isRealData, isSimulation, unknown };

    DataType dataType_;

    std::set<EventID> eventIDs_;

    bool itIsKnownTheFileHasNoDuplicates_;
  };
}
#endif
