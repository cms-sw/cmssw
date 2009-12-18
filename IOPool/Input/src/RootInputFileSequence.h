#ifndef IOPool_Input_RootInputFileSequence_h
#define IOPool_Input_RootInputFileSequence_h

/*----------------------------------------------------------------------

RootInputFileSequence: This is an InputSource

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>

#include "Inputfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GroupSelectorRules.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"

namespace CLHEP {
  class RandFlat;
}

namespace edm {

  class PoolSource;
  class RootFile;
  class FileCatalogItem;
  class InputFileCatalog;
  class FileIndex;
  class DuplicateChecker;

  class RootInputFileSequence : private boost::noncopyable {
  public:
    explicit RootInputFileSequence(ParameterSet const& pset, PoolSource const& input, InputFileCatalog const& catalog, PrincipalCache& cache, bool primarySequence);
    virtual ~RootInputFileSequence();

    typedef VectorInputSource::EventPrincipalVector EventPrincipalVector;
    typedef boost::shared_ptr<RootFile> RootFileSharedPtr;
    typedef input::EntryNumber EntryNumber;
    EventPrincipal* readEvent(EventPrincipal& cache, boost::shared_ptr<LuminosityBlockPrincipal> lb = boost::shared_ptr<LuminosityBlockPrincipal>());
    boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_(boost::shared_ptr<LuminosityBlockPrincipal> lbCache);
    boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    boost::shared_ptr<RunPrincipal> readRun_(boost::shared_ptr<RunPrincipal> rpCache);
    boost::shared_ptr<FileBlock> readFile_(PrincipalCache& cache);
    void closeFile_();
    void endJob();
    InputSource::ItemType getNextItemType();
    boost::shared_ptr<LuminosityBlockPrincipal> readIt(LuminosityBlockID const& id);
    boost::shared_ptr<RunPrincipal> readIt(RunID const& run);
    bool skipEvents(int offset, PrincipalCache& cache);
    bool skipToItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact, bool record);
    void rewind_();
    void reset(PrincipalCache& cache);
    void readMany(int number, EventPrincipalVector& result);
    void readMany(int number, EventPrincipalVector& result, EventID const& id, unsigned int fileSeqNumber);
    void readManyRandom(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    void readManySequential(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    void readManySpecified(std::vector<EventID> const& events, EventPrincipalVector& result); 
    void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches);
    boost::shared_ptr<ProductRegistry const> fileProductRegistry() const;
  private:
    void initFile(bool skipBadFiles);
    bool nextFile(PrincipalCache& cache);
    bool previousFile(PrincipalCache& cache);
    void rewindFile();
    void setSkipInfo();
    std::vector<FileCatalogItem> const& fileCatalogItems() const;

    boost::shared_ptr<ProductRegistry const> productRegistry() const;
    ProcessConfiguration const& processConfiguration() const;
    ProductRegistry & productRegistryUpdate() const;
    int remainingEvents() const;
    int remainingLuminosityBlocks() const;
    bool const primary() const;
    void logFileAction(const char* msg, std::string const& file);

    PoolSource const& input_;
    InputFileCatalog const& catalog_;
    bool firstFile_;
    std::vector<FileCatalogItem>::const_iterator fileIterBegin_;
    std::vector<FileCatalogItem>::const_iterator fileIterEnd_;
    std::vector<FileCatalogItem>::const_iterator fileIter_;
    RootFileSharedPtr rootFile_;
    BranchDescription::MatchMode parametersMustMatch_;
    BranchDescription::MatchMode branchesMustMatch_;

    boost::scoped_ptr<CLHEP::RandFlat> flatDistribution_;
    std::vector<boost::shared_ptr<FileIndex> > fileIndexes_;

    boost::scoped_ptr<EventSkipperByID> eventSkipperByID_;
    int eventsRemainingInFile_;
    RunNumber_t currentRun_;
    LuminosityBlockNumber_t currentLumi_;
    RunNumber_t skippedToRun_;
    LuminosityBlockNumber_t skippedToLumi_;
    EventNumber_t skippedToEvent_;
    FileIndex::EntryNumber_t skippedToEntry_;
    int numberOfEventsToSkip_;
    bool noEventSort_;
    bool skipBadFiles_;
    unsigned int treeCacheSize_;
    int const treeMaxVirtualSize_;
    RunNumber_t setRun_;
    GroupSelectorRules groupSelectorRules_;
    bool primarySequence_;
    boost::shared_ptr<DuplicateChecker> duplicateChecker_;
    bool dropDescendants_;
  }; // class RootInputFileSequence
}
#endif
