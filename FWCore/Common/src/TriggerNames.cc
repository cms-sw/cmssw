
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  TriggerNames::TriggerNames() { }

  TriggerNames::TriggerNames(edm::ParameterSet const& pset) {

    triggerNames_ = pset.getParameter<Strings>("@trigger_paths");

    unsigned int index = 0;
    for (Strings::const_iterator iName = triggerNames_.begin(),
         iEnd = triggerNames_.end();
         iName != iEnd;
         ++iName, ++index) {
      indexMap_[*iName] = index;
    }
    psetID_ = pset.id();
  }

  TriggerNames::Strings const&
  TriggerNames::triggerNames() const { return triggerNames_; }

  std::string const&
  TriggerNames::triggerName(unsigned int index) const {
    return triggerNames_.at(index);
  }

  unsigned int
  TriggerNames::triggerIndex(const std::string& name) const {
    IndexMap::const_iterator const pos = indexMap_.find(name);
    if (pos == indexMap_.end()) return indexMap_.size();
    return pos->second;
  }

  TriggerNames::Strings::size_type
  TriggerNames::size() const { return triggerNames_.size(); }

  ParameterSetID const&
  TriggerNames::parameterSetID() const { return psetID_; }

  TriggerNames::TriggerNames(TriggerResults const& triggerResults)
  {
    init(triggerResults);
  }

  bool
  TriggerNames::init(TriggerResults const& triggerResults) {

    if ( parameterSetID() != ParameterSetID() &&
         parameterSetID() == triggerResults.parameterSetID()) {
      return false;
    }

    if (triggerResults.parameterSetID() != ParameterSetID()) {

      // Get the parameter set containing the trigger names from the parameter
      // set registry using the ID from TriggerResults as the key used to find it.
      ParameterSet pset;
      pset::Registry* psetRegistry = pset::Registry::instance();
      if (psetRegistry->getMapped(triggerResults.parameterSetID(),
                                  pset)) {

        // Check to make sure the parameter set contains
        // a Strings parameter named "@trigger_paths".
        // We do not want to throw an exception if it is not there
        // for reasons of backward compatibility
        if (pset.existsAs<std::vector<std::string> >("@trigger_paths", true)) {

          // Set it to an invalid state temporarily until we are done
          // resetting the names just in case an exception occurs
          psetID_ = ParameterSetID();

          triggerNames_ = pset.getParameter<Strings>("@trigger_paths");
          unsigned int index = 0;
          for (Strings::const_iterator iName = triggerNames_.begin(),
               iEnd = triggerNames_.end();
               iName != iEnd;
               ++iName, ++index) {
               indexMap_[*iName] = index;
          }
          psetID_ = triggerResults.parameterSetID();

          // This should never happen
          if (triggerNames_.size() != triggerResults.size()) {
            throw edm::Exception(edm::errors::Unknown)
              << "TriggerNames::init, Trigger names vector and TriggerResults\n"
                 "are different sizes.  This should be impossible,\n"
                 "please send information to reproduce this problem to\n"
                 "the edm developers.\n";
          }
          return true;
	}
      }
    }

    // In very old versions of the code the the trigger names were stored
    // inside of the TriggerResults object.  This will provide backward
    // compatibility for data written with those versions.
    if (triggerResults.size() == triggerResults.getTriggerNames().size()) {

      // Set it to an invalid state temporarily until we are done
      // resetting the names just in case an exception occurs
      psetID_ = ParameterSetID();

      triggerNames_ = triggerResults.getTriggerNames();

      unsigned int index = 0;
      for (Strings::const_iterator iName = triggerNames_.begin(),
           iEnd = triggerNames_.end();
           iName != iEnd;
           ++iName, ++index) {
        indexMap_[*iName] = index;
      }
      psetID_ = triggerResults.parameterSetID();
      return true;
    }

    throw edm::Exception(edm::errors::Unknown)
      << "TriggerNames::init cannot find the trigger names for\n"
         "a TriggerResults object.  This should be impossible,\n"
         "please send information to reproduce this problem to\n"
         "the edm developers.\n"; 
    return true;
  }
}
