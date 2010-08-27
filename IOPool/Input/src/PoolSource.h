#ifndef IOPool_Input_PoolSource_h
#define IOPool_Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>

#include "Inputfwd.h"

#include "IOPool/Common/interface/RootServiceChecker.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/ProcessingController.h"

#include "boost/scoped_ptr.hpp"
#include "boost/array.hpp"

namespace edm {

  class RootInputFileSequence;
  class FileCatalogItem;
  class ConfigurationDescriptions;

  class PoolSource : public VectorInputSource {
  public:
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolSource();
    using InputSource::productRegistryUpdate;

    static void fillDescriptions(ConfigurationDescriptions & descriptions);

  private:
    typedef boost::shared_ptr<RootFile> RootFileSharedPtr;
    typedef input::EntryNumber EntryNumber;
    virtual EventPrincipal* readEvent_();
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_(boost::shared_ptr<LuminosityBlockPrincipal> lbCache);
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual boost::shared_ptr<RunPrincipal> readRun_(boost::shared_ptr<RunPrincipal> rpCache);
    virtual boost::shared_ptr<FileBlock> readFile_();
    virtual void closeFile_();
    virtual void endJob();
    virtual ItemType getNextItemType();
    virtual EventPrincipal* readIt(EventID const& id);
    virtual void skip(int offset);
    virtual bool goToEvent_(EventID const& eventID);
    virtual void rewind_();
    virtual void readMany_(int number, EventPrincipalVector& result);
    virtual void readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    virtual void readManySequential_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    virtual void readManySpecified_(std::vector<EventID> const& events, EventPrincipalVector& result);
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches);
    virtual void preForkReleaseResources();
    virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren, unsigned int iNumberOfSequentialEvents);
    virtual bool randomAccess_() const;
    virtual ProcessingController::ForwardState forwardState_() const;
    virtual ProcessingController::ReverseState reverseState_() const;

    RootServiceChecker rootServiceChecker_;
    boost::scoped_ptr<RootInputFileSequence> primaryFileSequence_;
    boost::scoped_ptr<RootInputFileSequence> secondaryFileSequence_;
    boost::scoped_ptr<EventPrincipal> secondaryEventPrincipal_;
    boost::array<std::vector<BranchID>, NumBranchTypes>  branchIDsToReplace_;

    //used when process has been forked
    unsigned int numberOfEventsBeforeBigSkip_;
    unsigned int numberOfEventsInBigSkip_;
    unsigned int numberOfSequentialEvents_;
    unsigned int forkedChildIndex_;

  }; // class PoolSource
  typedef PoolSource PoolRASource;
}
#endif
