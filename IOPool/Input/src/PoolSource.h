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

#include "boost/shared_ptr.hpp"

#include <array>
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
    virtual EventPrincipal* readEvent_(EventPrincipal& eventPrincipal);
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_(boost::shared_ptr<LuminosityBlockPrincipal> lbCache);
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual boost::shared_ptr<RunPrincipal> readRun_(boost::shared_ptr<RunPrincipal> rpCache);
    virtual boost::shared_ptr<FileBlock> readFile_();
    virtual void closeFile_();
    virtual void endJob();
    virtual ItemType getNextItemType();
    virtual EventPrincipal* readIt(EventID const& id, EventPrincipal& eventPrincipal);
    virtual void skip(int offset);
    virtual bool goToEvent_(EventID const& eventID);
    virtual void rewind_();
    virtual EventPrincipal* readOneRandom();
    virtual EventPrincipal* readOneRandomWithID(LuminosityBlockID const& lumiID);
    virtual EventPrincipal* readOneSequential();
    virtual EventPrincipal* readOneSequentialWithID(LuminosityBlockID const& lumiID);
    virtual EventPrincipal* readOneSpecified(EventID const& id);
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches);
    virtual void preForkReleaseResources();
    virtual void postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource>);
    virtual bool randomAccess_() const;
    virtual ProcessingController::ForwardState forwardState_() const;
    virtual ProcessingController::ReverseState reverseState_() const;

    RootServiceChecker rootServiceChecker_;
    std::unique_ptr<RootInputFileSequence> primaryFileSequence_;
    std::unique_ptr<RootInputFileSequence> secondaryFileSequence_;
    boost::shared_ptr<RunPrincipal> secondaryRunPrincipal_;
    boost::shared_ptr<LuminosityBlockPrincipal> secondaryLumiPrincipal_;
    std::unique_ptr<EventPrincipal> secondaryEventPrincipal_;
    std::array<std::vector<BranchID>, NumBranchTypes>  branchIDsToReplace_;

    //used when process has been forked
    boost::shared_ptr<edm::multicore::MessageReceiverForSource> receiver_;
    unsigned int numberOfEventsBeforeBigSkip_;

  }; // class PoolSource
  typedef PoolSource PoolRASource;
}
#endif
