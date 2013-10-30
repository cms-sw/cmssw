// -*- C++ -*-
//
// Implementation:
//
// Original Author:  Jim Kowalkowski
//

#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"


namespace edm {
  namespace service {

    TriggerNamesService::TriggerNamesService(const ParameterSet& pset) {

      trigger_pset_ = 
	pset.getParameterSet("@trigger_paths");

      trignames_ = trigger_pset_.getParameter<Strings>("@trigger_paths");
      end_names_ = pset.getParameter<Strings>("@end_paths");

      ParameterSet defopts;
      ParameterSet const& opts = 
	pset.getUntrackedParameterSet("options", defopts);
      wantSummary_ =
	opts.getUntrackedParameter("wantSummary", false);

      process_name_ = pset.getParameter<std::string>("@process_name");

      loadPosMap(trigpos_,trignames_);
      loadPosMap(end_pos_,end_names_);

      const unsigned int n(trignames_.size());
      for(unsigned int i = 0; i != n; ++i) {
        modulenames_.push_back(pset.getParameter<Strings>(trignames_[i]));
      }
      for(unsigned int i = 0; i != end_names_.size(); ++i) {
        end_modulenames_.push_back(pset.getParameter<Strings>(end_names_[i]));
      }
    }

    bool
    TriggerNamesService::getTrigPaths(TriggerResults const& triggerResults,
                                      Strings& trigPaths,
                                      bool& fromPSetRegistry) {

      // Get the parameter set containing the trigger names from the parameter set registry
      // using the ID from TriggerResults as the key used to find it.
      ParameterSet const* pset = nullptr;
      pset::Registry const* psetRegistry = pset::Registry::instance();
      if (nullptr != (pset = psetRegistry->getMapped(triggerResults.parameterSetID()))) {

        // Check to make sure the parameter set contains
        // a Strings parameter named "trigger_paths".
        // We do not want to throw an exception if it is not there
        // for reasons of backward compatibility
        Strings const& psetNames = pset->getParameterNamesForType<Strings>();
        std::string name("@trigger_paths");
	if (search_all(psetNames, name)) {
          // It is there, get it
          trigPaths = pset->getParameter<Strings>("@trigger_paths");

          // This should never happen
          if (trigPaths.size() != triggerResults.size()) {
            throw edm::Exception(edm::errors::Unknown)
              << "TriggerNamesService::getTrigPaths, Trigger names vector and\n"
                 "TriggerResults are different sizes.  This should be impossible,\n"
                 "please send information to reproduce this problem to\n"
                 "the edm developers.\n";
          }

          fromPSetRegistry = true;
          return true;
        }
      }

      fromPSetRegistry = false;

      // In older versions of the code the the trigger names were stored
      // inside of the TriggerResults object.  This will provide backward
      // compatibility.
      if (triggerResults.size() == triggerResults.getTriggerNames().size()) {
        trigPaths = triggerResults.getTriggerNames();
        return true;
      }

      return false;
    }

    bool
    TriggerNamesService::getTrigPaths(TriggerResults const& triggerResults,
                                      Strings& trigPaths) {
      bool dummy;
      return getTrigPaths(triggerResults, trigPaths, dummy);
    }
  }
}
