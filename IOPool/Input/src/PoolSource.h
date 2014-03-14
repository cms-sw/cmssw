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

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  class ConfigurationDescriptions;
  class FileCatalogItem;
  class RootInputFileSequence;

  class PoolSource : public VectorInputSource {
  public:
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolSource();
    using InputSource::processHistoryRegistryUpdate;
    using InputSource::productRegistryUpdate;

    static void fillDescriptions(ConfigurationDescriptions & descriptions);

  private:
    virtual void readEvent_(EventPrincipal& eventPrincipal);
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal);
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual void readRun_(RunPrincipal& runPrincipal);
    virtual std::unique_ptr<FileBlock> readFile_();
    virtual void closeFile_();
    virtual void endJob();
    virtual ItemType getNextItemType();
    virtual bool readIt(EventID const& id, EventPrincipal& eventPrincipal, StreamContext& streamContext) override;
    virtual void skip(int offset);
    virtual bool goToEvent_(EventID const& eventID);
    virtual void rewind_();
    virtual void readOneRandom(EventPrincipal& cache, CLHEP::HepRandomEngine*) override;
    virtual bool readOneRandomWithID(EventPrincipal& cache, LuminosityBlockID const& lumiID, CLHEP::HepRandomEngine*) override;
    virtual bool readOneSequential(EventPrincipal& cache);
    virtual bool readOneSequentialWithID(EventPrincipal& cache, LuminosityBlockID const& lumiID);
    virtual void readOneSpecified(EventPrincipal& cache, EventID const& id);
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches);
    virtual void preForkReleaseResources();
    virtual bool randomAccess_() const;
    virtual ProcessingController::ForwardState forwardState_() const;
    virtual ProcessingController::ReverseState reverseState_() const;

    RootServiceChecker rootServiceChecker_;
    std::unique_ptr<RootInputFileSequence> primaryFileSequence_;
    std::unique_ptr<RootInputFileSequence> secondaryFileSequence_;
    boost::shared_ptr<RunPrincipal> secondaryRunPrincipal_;
    boost::shared_ptr<LuminosityBlockPrincipal> secondaryLumiPrincipal_;
    std::vector<std::unique_ptr<EventPrincipal>> secondaryEventPrincipals_;
    std::array<std::vector<BranchID>, NumBranchTypes>  branchIDsToReplace_;
  }; // class PoolSource
  typedef PoolSource PoolRASource;
}
#endif
