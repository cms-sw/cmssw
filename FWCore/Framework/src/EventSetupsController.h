#ifndef FWCore_Framework_EventSetupsController_h
#define FWCore_Framework_EventSetupsController_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupsController
//
/** \class edm::eventsetup::EventSetupsController

 Description: Manages a group of EventSetups which can share components

 Usage:
    <usage>

*/
//
// Original Authors:  Chris Jones, David Dagenhart
//          Created:  Wed Jan 12 14:30:42 CST 2011
//

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Framework/src/EventSetupRecordIOVQueue.h"
#include "FWCore/Framework/src/NumberOfConcurrentIOVs.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <map>
#include <memory>
#include <vector>

namespace edm {

  class ActivityRegistry;
  class EventSetupImpl;
  class EventSetupRecordIntervalFinder;
  class ParameterSet;
  class IOVSyncValue;
  class WaitingTaskHolder;
  class WaitingTaskList;

  namespace eventsetup {

    class DataProxyProvider;
    class EventSetupProvider;

    class ESProducerInfo {
    public:
      ESProducerInfo(ParameterSet const* ps, std::shared_ptr<DataProxyProvider> const& pr)
          : pset_(ps), provider_(pr), subProcessIndexes_() {}

      ParameterSet const* pset() const { return pset_; }
      std::shared_ptr<DataProxyProvider> const& provider() { return get_underlying(provider_); }
      DataProxyProvider const* providerGet() const { return provider_.get(); }
      std::vector<unsigned>& subProcessIndexes() { return subProcessIndexes_; }
      std::vector<unsigned> const& subProcessIndexes() const { return subProcessIndexes_; }

    private:
      ParameterSet const* pset_;
      propagate_const<std::shared_ptr<DataProxyProvider>> provider_;
      std::vector<unsigned> subProcessIndexes_;
    };

    class ESSourceInfo {
    public:
      ESSourceInfo(ParameterSet const* ps, std::shared_ptr<EventSetupRecordIntervalFinder> const& fi)
          : pset_(ps), finder_(fi), subProcessIndexes_() {}

      ParameterSet const* pset() const { return pset_; }
      std::shared_ptr<EventSetupRecordIntervalFinder> const& finder() { return get_underlying(finder_); }
      EventSetupRecordIntervalFinder const* finderGet() const { return finder_.get(); }
      std::vector<unsigned>& subProcessIndexes() { return subProcessIndexes_; }
      std::vector<unsigned> const& subProcessIndexes() const { return subProcessIndexes_; }

    private:
      ParameterSet const* pset_;
      propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>> finder_;
      std::vector<unsigned> subProcessIndexes_;
    };

    class EventSetupsController {
    public:
      EventSetupsController();

      EventSetupsController(EventSetupsController const&) = delete;
      EventSetupsController const& operator=(EventSetupsController const&) = delete;

      std::shared_ptr<EventSetupProvider> makeProvider(ParameterSet&,
                                                       ActivityRegistry*,
                                                       ParameterSet const* eventSetupPset = nullptr);

      void setMaxConcurrentIOVs(unsigned int nStreams, unsigned int nConcurrentLumis);

      // Pass in an IOVSyncValue to let the EventSetup system know which run and lumi
      // need to be processed and prepare IOVs for it (also could be a time or only a run).
      // Pass in a WaitingTaskHolder that allows the EventSetup to communicate when all
      // the IOVs are ready to process this IOVSyncValue. Note this preparation is often
      // done in asynchronous tasks and the function might return before all the preparation
      // is complete.
      // Pass in endIOVWaitingTasks, additions to this WaitingTaskList allow the lumi to notify
      // the EventSetup when the lumi is done and no longer needs its EventSetup IOVs.
      // Pass in a vector of EventSetupImpl that gets filled and is used to give clients
      // of EventSetup access to the EventSetup system such that for each record the IOV
      // associated with this IOVSyncValue will be used. The first element of the vector
      // is for the top level process and each additional element corresponds to a SubProcess.
      void eventSetupForInstance(IOVSyncValue const&,
                                 WaitingTaskHolder const& taskToStartAfterIOVInit,
                                 WaitingTaskList& endIOVWaitingTasks,
                                 std::vector<std::shared_ptr<const EventSetupImpl>>&);

      // Version to use when IOVs are not allowed to run concurrently
      void eventSetupForInstance(IOVSyncValue const&);

      bool doWeNeedToWaitForIOVsToFinish(IOVSyncValue const&) const;

      void forceCacheClear();

      std::shared_ptr<DataProxyProvider> getESProducerAndRegisterProcess(ParameterSet const& pset,
                                                                         unsigned subProcessIndex);
      void putESProducer(ParameterSet const& pset,
                         std::shared_ptr<DataProxyProvider> const& component,
                         unsigned subProcessIndex);

      std::shared_ptr<EventSetupRecordIntervalFinder> getESSourceAndRegisterProcess(ParameterSet const& pset,
                                                                                    unsigned subProcessIndex);
      void putESSource(ParameterSet const& pset,
                       std::shared_ptr<EventSetupRecordIntervalFinder> const& component,
                       unsigned subProcessIndex);

      void finishConfiguration();
      void clearComponents();

      unsigned indexOfNextProcess() const { return providers_.size(); }

      void lookForMatches(ParameterSetID const& psetID,
                          unsigned subProcessIndex,
                          unsigned precedingProcessIndex,
                          bool& firstProcessWithThisPSet,
                          bool& precedingHasMatchingPSet) const;

      bool isFirstMatch(ParameterSetID const& psetID, unsigned subProcessIndex, unsigned precedingProcessIndex) const;

      bool isLastMatch(ParameterSetID const& psetID, unsigned subProcessIndex, unsigned precedingProcessIndex) const;

      bool isMatchingESSource(ParameterSetID const& psetID,
                              unsigned subProcessIndex,
                              unsigned precedingProcessIndex) const;

      bool isMatchingESProducer(ParameterSetID const& psetID,
                                unsigned subProcessIndex,
                                unsigned precedingProcessIndex) const;

      ParameterSet const* getESProducerPSet(ParameterSetID const& psetID, unsigned subProcessIndex) const;

      std::vector<propagate_const<std::shared_ptr<EventSetupProvider>>> const& providers() const { return providers_; }

      std::multimap<ParameterSetID, ESProducerInfo> const& esproducers() const { return esproducers_; }

      std::multimap<ParameterSetID, ESSourceInfo> const& essources() const { return essources_; }

      bool hasNonconcurrentFinder() const { return hasNonconcurrentFinder_; }
      bool mustFinishConfiguration() const { return mustFinishConfiguration_; }

    private:
      void checkESProducerSharing();
      void initializeEventSetupRecordIOVQueues();

      // ---------- member data --------------------------------
      std::vector<propagate_const<std::shared_ptr<EventSetupProvider>>> providers_;
      NumberOfConcurrentIOVs numberOfConcurrentIOVs_;

      // This data member is intentionally declared after providers_
      // It is important that this is destroyed first.
      std::vector<propagate_const<std::unique_ptr<EventSetupRecordIOVQueue>>> eventSetupRecordIOVQueues_;

      // The following two multimaps have one entry for each unique
      // ParameterSet. The ESProducerInfo or ESSourceInfo object
      // contains a list of the processes that use that ParameterSet
      // (0 for the top level process, then the SubProcesses are
      // identified by counting their execution order starting at 1).
      // There can be multiple entries for a single ParameterSetID because
      // of a difference in untracked parameters. These are only
      // used during initialization.  The Info objects also contain
      // a pointer to the full validated ParameterSet and a shared_ptr
      // to the component.
      std::multimap<ParameterSetID, ESProducerInfo> esproducers_;
      std::multimap<ParameterSetID, ESSourceInfo> essources_;

      bool hasNonconcurrentFinder_ = false;
      bool mustFinishConfiguration_ = true;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
