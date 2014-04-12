// -*- C++ -*-
//
// Package:     FWCore/Common
// Class  :     LuminosityBlockBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Eric Vaandering
//         Created:  Tue Jan 12 15:31:00 CDT 2010
//

// system include files
#include <vector>
#include <map>

// user include files
#include "FWCore/Common/interface/LuminosityBlockBase.h"
//#include "FWCore/Common/interface/TriggerNames.h"
//#include "DataFormats/Provenance/interface/ParameterSetID.h"
//#include "DataFormats/Common/interface/TriggerResults.h"
//#include "FWCore/Utilities/interface/Exception.h"
//#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/ParameterSet/interface/Registry.h"

namespace edm
{
//    typedef std::map<edm::ParameterSetID, edm::TriggerNames> TriggerNamesMap;
//    static TriggerNamesMap triggerNamesMap;
//    static TriggerNamesMap::const_iterator previousTriggerName;

   LuminosityBlockBase::LuminosityBlockBase()
   {
   }

   LuminosityBlockBase::~LuminosityBlockBase()
   {
   }

/*   TriggerNames const*
   EventBase::triggerNames_(edm::TriggerResults const& triggerResults) {

      // If the current and previous requests are for the same TriggerNames
      // then just return it.
      if (!triggerNamesMap.empty() &&
          previousTriggerName->first == triggerResults.parameterSetID()) {
         return &previousTriggerName->second;
      }

      // If TriggerNames was already created and cached here in the map,
      // then look it up and return that one
      TriggerNamesMap::const_iterator iter =
         triggerNamesMap.find(triggerResults.parameterSetID());
      if (iter != triggerNamesMap.end()) {
         previousTriggerName = iter;
         return &iter->second;
      }

      // Look for the parameter set containing the trigger names in the parameter
      // set registry using the ID from TriggerResults as the key used to find it.
      edm::ParameterSet pset;
      edm::pset::Registry* psetRegistry = edm::pset::Registry::instance();
      if (psetRegistry->getMapped(triggerResults.parameterSetID(),
                                  pset)) {

         if (pset.existsAs<std::vector<std::string> >("@trigger_paths", true)) {
            TriggerNames triggerNames(pset);

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
            previousTriggerName = ret.first;
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
         previousTriggerName = ret.first;
         return &(ret.first->second);
      }
      return 0;
   }*/
}
