#ifndef IOPool_Input_PoolSource_h
#define IOPool_Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "IOPool/Common/interface/RootServiceChecker.h"

#include "boost/array.hpp"
#include "boost/scoped_ptr.hpp"

#include <memory>
#include <string>
#include <vector>

namespace edm {

  class ConfigurationDescriptions;
  class FileCatalogItem;
  class RootInputFileSequence;

  class PoolSource : public VectorInputSource {
  public:
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolSource();
    using InputSource::productRegistryUpdate;

    static void fillDescriptions(ConfigurationDescriptions & descriptions);

  private:
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
    virtual EventPrincipal* readOneRandom();
    virtual EventPrincipal* readOneSequential();
    virtual EventPrincipal* readOneSpecified(EventID const& id);
    virtual void readMany_(int number, EventPrincipalVector& result);
    virtual void readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    virtual void readManySequential_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    virtual void readManySpecified_(std::vector<EventID> const& events, EventPrincipalVector& result);
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches);
    virtual void preForkReleaseResources();
    virtual void postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource>);
    virtual bool randomAccess_() const;
    virtual ProcessingController::ForwardState forwardState_() const;
    virtual ProcessingController::ReverseState reverseState_() const;

    RootServiceChecker rootServiceChecker_;
    boost::scoped_ptr<RootInputFileSequence> primaryFileSequence_;
    boost::scoped_ptr<RootInputFileSequence> secondaryFileSequence_;
    boost::shared_ptr<RunPrincipal> secondaryRunPrincipal_;
    boost::shared_ptr<LuminosityBlockPrincipal> secondaryLumiPrincipal_;
    boost::scoped_ptr<EventPrincipal> secondaryEventPrincipal_;
    boost::array<std::vector<BranchID>, NumBranchTypes>  branchIDsToReplace_;

    //used when process has been forked
    boost::shared_ptr<edm::multicore::MessageReceiverForSource> receiver_;
    unsigned int numberOfEventsBeforeBigSkip_;

  }; // class PoolSource
  typedef PoolSource PoolRASource;
}
#endif
