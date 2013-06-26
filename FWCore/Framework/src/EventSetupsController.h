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

#include <boost/shared_ptr.hpp>

#include <map>
#include <vector>

namespace edm {

   class EventSetupRecordIntervalFinder;
   class ParameterSet;
   class IOVSyncValue;
   
   namespace eventsetup {

      class DataProxyProvider;
      class EventSetupProvider;

      class ESProducerInfo {
      public:
         ESProducerInfo(ParameterSet const* ps,
                        boost::shared_ptr<DataProxyProvider> const& pr) : 
            pset_(ps), provider_(pr), subProcessIndexes_() { }

         ParameterSet const* pset() const { return pset_; }
         boost::shared_ptr<DataProxyProvider> const& provider() const { return provider_; }
         std::vector<unsigned>& subProcessIndexes() { return subProcessIndexes_; }
         std::vector<unsigned> const& subProcessIndexes() const { return subProcessIndexes_; }

      private:
         ParameterSet const* pset_;
         boost::shared_ptr<DataProxyProvider> provider_;
         std::vector<unsigned> subProcessIndexes_;
      };

      class ESSourceInfo {
      public:
         ESSourceInfo(ParameterSet const* ps,
                      boost::shared_ptr<EventSetupRecordIntervalFinder> const& fi) :
            pset_(ps), finder_(fi), subProcessIndexes_() { }

         ParameterSet const* pset() const { return pset_; }
         boost::shared_ptr<EventSetupRecordIntervalFinder> const& finder() const { return finder_; }
         std::vector<unsigned>& subProcessIndexes() { return subProcessIndexes_; }
         std::vector<unsigned> const& subProcessIndexes() const { return subProcessIndexes_; }

      private:
         ParameterSet const* pset_;
         boost::shared_ptr<EventSetupRecordIntervalFinder> finder_;
         std::vector<unsigned> subProcessIndexes_;
      };

      class EventSetupsController {
         
      public:
         EventSetupsController();

         boost::shared_ptr<EventSetupProvider> makeProvider(ParameterSet&);

         void eventSetupForInstance(IOVSyncValue const& syncValue);

         void forceCacheClear() const;

         boost::shared_ptr<DataProxyProvider> getESProducerAndRegisterProcess(ParameterSet const& pset, unsigned subProcessIndex);
         void putESProducer(ParameterSet const& pset, boost::shared_ptr<DataProxyProvider> const& component, unsigned subProcessIndex);

         boost::shared_ptr<EventSetupRecordIntervalFinder> getESSourceAndRegisterProcess(ParameterSet const& pset, unsigned subProcessIndex);
         void putESSource(ParameterSet const& pset, boost::shared_ptr<EventSetupRecordIntervalFinder> const& component, unsigned subProcessIndex);

         void clearComponents();

         unsigned indexOfNextProcess() const { return providers_.size(); }

         void lookForMatches(ParameterSetID const& psetID,
                             unsigned subProcessIndex,
                             unsigned precedingProcessIndex,
                             bool& firstProcessWithThisPSet,
                             bool& precedingHasMatchingPSet) const;

         bool isFirstMatch(ParameterSetID const& psetID,
                           unsigned subProcessIndex,
                           unsigned precedingProcessIndex) const;

         bool isLastMatch(ParameterSetID const& psetID,
                          unsigned subProcessIndex,
                          unsigned precedingProcessIndex) const;

         bool isMatchingESSource(ParameterSetID const& psetID,
                                 unsigned subProcessIndex,
                                 unsigned precedingProcessIndex) const;

         bool isMatchingESProducer(ParameterSetID const& psetID,
                                   unsigned subProcessIndex,
                                   unsigned precedingProcessIndex) const;

         ParameterSet const* getESProducerPSet(ParameterSetID const& psetID,
                                               unsigned subProcessIndex) const;

         std::vector<boost::shared_ptr<EventSetupProvider> > const& providers() const { return providers_; }

         std::multimap<ParameterSetID, ESProducerInfo> const& esproducers() const { return esproducers_; }

         std::multimap<ParameterSetID, ESSourceInfo> const& essources() const { return essources_; }

         bool mustFinishConfiguration() const { return mustFinishConfiguration_; }

      private:
         EventSetupsController(EventSetupsController const&); // stop default
         
         EventSetupsController const& operator=(EventSetupsController const&); // stop default

         void checkESProducerSharing();
         
         // ---------- member data --------------------------------
         std::vector<boost::shared_ptr<EventSetupProvider> > providers_;

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

         bool mustFinishConfiguration_;
      };
   }
}
#endif
