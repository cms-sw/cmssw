// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupsController
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones, W. David Dagenhart
//         Created:  Wed Jan 12 14:30:44 CST 2011
// $Id: EventSetupsController.cc,v 1.3 2012/04/16 15:43:49 wdd Exp $
//

#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetupProviderMaker.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/ParameterSetIDHolder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <iostream>

namespace edm {
  namespace eventsetup {

    EventSetupsController::EventSetupsController() : mustFinishConfiguration_(true) {
    }

    boost::shared_ptr<EventSetupProvider>
    EventSetupsController::makeProvider(ParameterSet& iPSet) {

      // Makes an EventSetupProvider
      // Also parses the prefer information from ParameterSets and puts
      // it in a map that is stored in the EventSetupProvider
      boost::shared_ptr<EventSetupProvider> returnValue(makeEventSetupProvider(iPSet, providers_.size()) );

      // Construct the ESProducers and ESSources
      // shared_ptrs to them are temporarily stored in this
      // EventSetupsController and in the EventSetupProvider
      fillEventSetupProvider(*this, *returnValue, iPSet);
   
      providers_.push_back(returnValue);
      return returnValue;
    }

    void
    EventSetupsController::eventSetupForInstance(IOVSyncValue const& syncValue) {

      if (mustFinishConfiguration_) {
        std::for_each(providers_.begin(), providers_.end(), [](boost::shared_ptr<EventSetupProvider> const& esp) {
          esp->finishConfiguration();
        });
        // When the ESSources and ESProducers were constructed a first pass was
        // done which attempts to get component sharing between SubProcesses
        // correct, but in this pass only the configuration of the components
        // being shared are compared. This is not good enough for ESProducers.
        // In the following function, all the other components that contribute 
        // to the same record and also the records that record depends on are
        // also checked. The component sharing is appropriately fixed as necessary.
        checkESProducerSharing();
        clearComponents();
        mustFinishConfiguration_ = false;
      }

      std::for_each(providers_.begin(), providers_.end(), [&syncValue](boost::shared_ptr<EventSetupProvider> const& esp) {
        esp->eventSetupForInstance(syncValue);
      });
    }

    void
    EventSetupsController::forceCacheClear() const {
      std::for_each(providers_.begin(), providers_.end(), [](boost::shared_ptr<EventSetupProvider> const& esp) {
        esp->forceCacheClear();
      });
    }

    boost::shared_ptr<DataProxyProvider>
    EventSetupsController::getESProducerAndRegisterProcess(ParameterSet const& pset, unsigned subProcessIndex) {
      // Try to find a DataProxyProvider with a matching ParameterSet
      auto elements = esproducers_.equal_range(pset.id());
      for (auto it = elements.first; it != elements.second; ++it) {
        // Untracked parameters must also match, do complete comparison if IDs match
        if (isTransientEqual(pset, *it->second.pset())) {
          // Register processes with an exact match
          it->second.subProcessIndexes().push_back(subProcessIndex);
          // Return the DataProxyProvider
          return it->second.provider();
        }
      }
      // Could not find it
      return boost::shared_ptr<DataProxyProvider>();
    }

    void
    EventSetupsController::putESProducer(ParameterSet const& pset, boost::shared_ptr<DataProxyProvider> const& component, unsigned subProcessIndex) {
      auto newElement = esproducers_.insert(std::pair<ParameterSetID, ESProducerInfo>(pset.id(), 
                                                                                      ESProducerInfo(&pset, component)));
      // Register processes with an exact match
      newElement->second.subProcessIndexes().push_back(subProcessIndex);
    }

    boost::shared_ptr<EventSetupRecordIntervalFinder>
    EventSetupsController::getESSourceAndRegisterProcess(ParameterSet const& pset, unsigned subProcessIndex) {
      // Try to find a EventSetupRecordIntervalFinder with a matching ParameterSet
      auto elements = essources_.equal_range(pset.id());
      for (auto it = elements.first; it != elements.second; ++it) {
        // Untracked parameters must also match, do complete comparison if IDs match
        if (isTransientEqual(pset, *it->second.pset())) {
          // Register processes with an exact match
          it->second.subProcessIndexes().push_back(subProcessIndex);
          // Return the EventSetupRecordIntervalFinder
          return it->second.finder();
        }
      }
      // Could not find it
      return boost::shared_ptr<EventSetupRecordIntervalFinder>();
    }

    void
    EventSetupsController::putESSource(ParameterSet const& pset, boost::shared_ptr<EventSetupRecordIntervalFinder> const& component, unsigned subProcessIndex) {
      auto newElement = essources_.insert(std::pair<ParameterSetID, ESSourceInfo>(pset.id(), 
                                                                                  ESSourceInfo(&pset, component)));
      // Register processes with an exact match
      newElement->second.subProcessIndexes().push_back(subProcessIndex);
    }

    void
    EventSetupsController::clearComponents() {
      esproducers_.clear();
      essources_.clear();
    }

    void
    EventSetupsController::lookForMatches(ParameterSetID const& psetID,
                                          unsigned subProcessIndex,
                                          unsigned precedingProcessIndex,
                                          bool& firstProcessWithThisPSet,
                                          bool& precedingHasMatchingPSet) const {

      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {

        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        if (iFound == subProcessIndexes.begin()) {
          firstProcessWithThisPSet = true;
          precedingHasMatchingPSet = false;      
        } else {
          auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
          if (iFoundPreceding == iFound) {
            firstProcessWithThisPSet = false;
            precedingHasMatchingPSet = false;
          } else {
            firstProcessWithThisPSet = false;
            precedingHasMatchingPSet = true;    
          }
        }
        return;
      }
      throw edm::Exception(edm::errors::LogicError)
        << "EventSetupsController::lookForMatches\n"
        << "Subprocess index not found. This should never happen\n"
        << "Please report this to a Framework Developer\n";   
    }

    bool
    EventSetupsController::isFirstMatch(ParameterSetID const& psetID,
                                        unsigned subProcessIndex,
                                        unsigned precedingProcessIndex) const {

      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {

        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
        if (iFoundPreceding == iFound) {
          break;
        } else {
          return iFoundPreceding == subProcessIndexes.begin();
        }
      }
      throw edm::Exception(edm::errors::LogicError)
        << "EventSetupsController::isFirstMatch\n"
        << "Subprocess index not found. This should never happen\n"
        << "Please report this to a Framework Developer\n";
      return false;
    }

    bool
    EventSetupsController::isLastMatch(ParameterSetID const& psetID,
                                       unsigned subProcessIndex,
                                       unsigned precedingProcessIndex) const {

      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {

        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
        if (iFoundPreceding == iFound) {
          break;
        } else {
          return (++iFoundPreceding) == iFound;
        }
      }
      throw edm::Exception(edm::errors::LogicError)
        << "EventSetupsController::isLastMatch\n"
        << "Subprocess index not found. This should never happen\n"
        << "Please report this to a Framework Developer\n";
      return false;
    }

    bool
    EventSetupsController::isMatchingESSource(ParameterSetID const& psetID,
                                              unsigned subProcessIndex,
                                              unsigned precedingProcessIndex) const {
      auto elements = essources_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {

        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
        if (iFoundPreceding == iFound) {
          return false;
        } else {
          return true;    
        }
      }
      throw edm::Exception(edm::errors::LogicError)
        << "EventSetupsController::lookForMatchingESSource\n"
        << "Subprocess index not found. This should never happen\n"
        << "Please report this to a Framework Developer\n";
      return false;
    }

    bool
    EventSetupsController::isMatchingESProducer(ParameterSetID const& psetID,
                                                unsigned subProcessIndex,
                                                unsigned precedingProcessIndex) const {
      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {

        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
        if (iFoundPreceding == iFound) {
          return false;
        } else {
          return true;    
        }
      }
      throw edm::Exception(edm::errors::LogicError)
        << "EventSetupsController::lookForMatchingESSource\n"
        << "Subprocess index not found. This should never happen\n"
        << "Please report this to a Framework Developer\n";
      return false;
    }

    ParameterSet const*
    EventSetupsController::getESProducerPSet(ParameterSetID const& psetID,
                                             unsigned subProcessIndex) const {
   
      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {

        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }
        return it->second.pset();
      }
      throw edm::Exception(edm::errors::LogicError)
        << "EventSetupsController::getESProducerPSet\n"
        << "Subprocess index not found. This should never happen\n"
        << "Please report this to a Framework Developer\n";
      return 0;
    }

    void
    EventSetupsController::checkESProducerSharing() {

      // Loop over SubProcesses, skip the top level process.
      auto esProvider = providers_.begin();
      auto esProviderEnd = providers_.end();
      if (esProvider != esProviderEnd) ++esProvider;
      for ( ; esProvider != esProviderEnd; ++esProvider) {

        // An element is added to this set for each ESProducer
        // when we have determined which preceding process
        // this process can share that ESProducer with or
        // we have determined that it cannot be shared with
        // any preceding process.
        // Note the earliest possible preceding process
        // will be the one selected if there is more than one.
        std::set<ParameterSetIDHolder> sharingCheckDone;

        // This will hold an entry for DataProxy's that are
        // referenced by an EventSetupRecord in this SubProcess.
        // But only for DataProxy's that are associated with
        // an ESProducer (not the ones associated with ESSource's
        // or EDLooper's)
        std::map<EventSetupRecordKey, std::vector<ComponentDescription const*> > referencedESProducers;

        // For each EventSetupProvider from a SubProcess, loop over the
        // EventSetupProviders from the preceding processes (the first
        // preceding process will be the top level process and the others
        // SubProcess's)
        for (auto precedingESProvider = providers_.begin();
             precedingESProvider != esProvider;
             ++precedingESProvider) {

          (*esProvider)->checkESProducerSharing(**precedingESProvider, sharingCheckDone, referencedESProducers, *this);
        }

        (*esProvider)->resetRecordToProxyPointers();
      }
      esProvider = providers_.begin();
      for ( ; esProvider != esProviderEnd; ++esProvider) {
        (*esProvider)->clearInitializationData();
      }
    }
  }
}
