// -*- C++ -*-
//
// Package:     FWCore/Common
// Class  :     EventBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Aug 27 11:20:06 CDT 2009
//

// system include files
#include <vector>
#include "tbb/concurrent_unordered_map.h"

// user include files
#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

namespace {
   struct key_hash {
      std::size_t operator()(edm::ParameterSetID const& iKey) const{
         return iKey.smallHash();
      }
   };
   typedef tbb::concurrent_unordered_map<edm::ParameterSetID, edm::TriggerNames, key_hash> TriggerNamesMap;
   [[cms::thread_safe]] static TriggerNamesMap triggerNamesMap;
}

namespace edm
{

   EventBase::EventBase()
   {
   }

   EventBase::~EventBase()
   {
   }

   TriggerNames const*
   EventBase::triggerNames_(edm::TriggerResults const& triggerResults) {

      // If TriggerNames was already created and cached here in the map,
      // then look it up and return that one
      TriggerNamesMap::const_iterator iter =
         triggerNamesMap.find(triggerResults.parameterSetID());
      if (iter != triggerNamesMap.end()) {
         return &iter->second;
      }

      // Look for the parameter set containing the trigger names in the parameter
      // set registry using the ID from TriggerResults as the key used to find it.
      edm::pset::Registry* psetRegistry = edm::pset::Registry::instance();
      edm::ParameterSet const* pset=0;
      if (0!=(pset=psetRegistry->getMapped(triggerResults.parameterSetID()))) {

	 if (pset->existsAs<std::vector<std::string> >("@trigger_paths", true)) {
            TriggerNames triggerNames(*pset);

            // This should never happen
            if (triggerNames.size() != triggerResults.size()) {
               throw cms::Exception("LogicError")
                  << "edm::EventBase::triggerNames_ Encountered vector\n"
                     "of trigger names and a TriggerResults object with\n"
                     "different sizes.  This should be impossible.\n"
                     "Please send information to reproduce this problem to\n"
                     "the edm developers.\n";
            }

            std::pair<TriggerNamesMap::iterator, bool> ret =
               triggerNamesMap.insert(std::pair<edm::ParameterSetID, edm::TriggerNames>(triggerResults.parameterSetID(), triggerNames));
            return &(ret.first->second);
         }
      }
      // For backward compatibility to very old data
      if (triggerResults.getTriggerNames().size() > 0U) {
	 edm::ParameterSet fakePset;
         fakePset.addParameter<std::vector<std::string> >("@trigger_paths", triggerResults.getTriggerNames());
         fakePset.registerIt();
         TriggerNames triggerNames(fakePset);

         // This should never happen
         if (triggerNames.size() != triggerResults.size()) {
            throw cms::Exception("LogicError")
               << "edm::EventBase::triggerNames_ Encountered vector\n"
                  "of trigger names and a TriggerResults object with\n"
                  "different sizes.  This should be impossible.\n"
                  "Please send information to reproduce this problem to\n"
                  "the edm developers (2).\n";
         }

         std::pair<TriggerNamesMap::iterator, bool> ret =
            triggerNamesMap.insert(std::pair<edm::ParameterSetID, edm::TriggerNames>(fakePset.id(), triggerNames));
         return &(ret.first->second);
      }
      return 0;
   }
}
