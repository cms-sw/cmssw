// -*- C++ -*-
//
// Package:     FWLite
// Class  :     TriggerNames
// 
// Implementation:
//
// Original Author:  W. David Dagenhart
//         Created:  28 August 2009
//

#include "DataFormats/FWLite/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace fwlite {

   TriggerNames::TriggerNames(edm::ParameterSet const& pset) {
      triggerNames_ = pset.getParameter<Strings>("@trigger_paths");

      unsigned int index = 0;
      for (Strings::const_iterator iName = triggerNames_.begin(),
           iEnd = triggerNames_.end();
           iName != iEnd;
           ++iName, ++index) {
         indexMap_[*iName] = index;
      }
   }

   TriggerNames::~TriggerNames() { }

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
}
