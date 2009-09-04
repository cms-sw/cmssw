// -*- C++ -*-
//
// Package:     FWLite
// Class  :     EventBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Charles Plager
//         Created:  
// $Id: 
//

// system include files
#include <vector>
#include <map>
#include <iostream>
#include "Reflex/Type.h"

// user include files
#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/TriggerNames.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"

static const edm::ProductID s_id;
static const edm::BranchDescription s_branch;
static const edm::Provenance s_prov(s_branch,s_id);

namespace {
   //This is used by the shared_ptr required to be passed to BasicHandle to keep the shared_ptr from doing the delete
   struct null_deleter
   {
      void operator()(void const *) const
      {
      }
   };
}
   
namespace fwlite
{
   typedef std::map<edm::ParameterSetID, fwlite::TriggerNames> TriggerNamesMap;
   static TriggerNamesMap triggerNamesMap;
   static TriggerNamesMap::const_iterator previousTriggerName;

   EventBase::EventBase()
   {
   }

   EventBase::~EventBase()
   {
   }
   
   TriggerNames const*
   EventBase::triggerNames_(edm::TriggerResults const& triggerResults) {

      if (!triggerNamesMap.empty() &&
          previousTriggerName->first == triggerResults.parameterSetID()) {
         return &previousTriggerName->second;
      }

      // If TriggerNames was already created and cached here, then return that one
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
                  << "fwlite::EventBase::triggerNames_ Encountered vector\n"
                     "of trigger names and a TriggerResults object with\n"
                     "different sizes.  This should be impossible.\n"
                     "Please send information to reproduce this problem to\n"
                     "the fwlite developers.\n";
            }

            std::pair<TriggerNamesMap::iterator, bool> ret =
               triggerNamesMap.insert(std::pair<edm::ParameterSetID, fwlite::TriggerNames>(triggerResults.parameterSetID(), triggerNames));
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
               << "fwlite::EventBase::triggerNames_ Encountered vector\n"
                  "of trigger names and a TriggerResults object with\n"
                  "different sizes.  This should be impossible.\n"
                  "Please send information to reproduce this problem to\n"
                  "the fwlite developers (2).\n";
         }

         std::pair<TriggerNamesMap::iterator, bool> ret =
            triggerNamesMap.insert(std::pair<edm::ParameterSetID, fwlite::TriggerNames>(fakePset.id(), triggerNames));
         previousTriggerName = ret.first;
         return &(ret.first->second);
      }
      return 0;
   }

   edm::BasicHandle 
   EventBase::getByLabelImpl(const std::type_info& iWrapperInfo, const std::type_info& /*iProductInfo*/, const edm::InputTag& iTag) const 
   {
      edm::EDProduct* prod=0;
      void* prodPtr = &prod;
      getByLabel(iWrapperInfo, 
                 iTag.label().c_str(), 
                 iTag.instance().empty()?static_cast<const char*>(0):iTag.instance().c_str(),
                 iTag.process().empty()?static_cast<const char*> (0):iTag.process().c_str(),
                 prodPtr);
      if(0==prod) {
         edm::TypeID productType(iWrapperInfo);
         boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound));
         *whyFailed
         << "getByLabel: Found zero products matching all criteria\n"
         << "Looking for type: " << productType << "\n"
         << "Looking for module label: " << iTag.label() << "\n"
         << "Looking for productInstanceName: " << iTag.instance() << "\n"
         << (iTag.process().empty() ? "" : "Looking for process: ") << iTag.process() << "\n";
         
         edm::BasicHandle failed(whyFailed);
         return failed;
      }
      edm::BasicHandle value(boost::shared_ptr<edm::EDProduct>(prod,null_deleter()),&s_prov);
      return value;
   }
}
