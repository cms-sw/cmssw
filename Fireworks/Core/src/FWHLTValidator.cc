// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHLTValidator
//

// system include files
#include <algorithm>
#include <cstring>
#include <regex>

// user include files
#include "Fireworks/Core/interface/FWHLTValidator.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/EventBase.h"

void FWHLTValidator::fillOptions(const char* iBegin,
                                 const char* iEnd,
                                 std::vector<std::pair<std::shared_ptr<std::string>, std::string> >& oOptions) const {
  oOptions.clear();
  std::string part(iBegin, iEnd);
  part = std::regex_replace(part, std::regex(".*?(\\&\\&|\\|\\||\\s)+"), "");

  if (m_triggerNames.empty()) {
    edm::Handle<edm::TriggerResults> hTriggerResults;
    edm::TriggerNames const* triggerNames(nullptr);
    try {
      edm::InputTag tag("TriggerResults", "", m_process.c_str());
      const edm::EventBase* event = FWGUIManager::getGUIManager()->getCurrentEvent();
      event->getByLabel(tag, hTriggerResults);
      triggerNames = &event->triggerNames(*hTriggerResults);
    } catch (...) {
      fwLog(fwlog::kWarning) << " no trigger results with process name " << m_process << "  is available" << std::endl;
      return;
    }
    for (unsigned int i = 0; i < triggerNames->size(); ++i)
      m_triggerNames.push_back(triggerNames->triggerName(i));
    std::sort(m_triggerNames.begin(), m_triggerNames.end());
  }

  //only use add items which begin with the part of the member we are trying to match
  unsigned int part_size = part.size();
  for (std::vector<std::string>::const_iterator trigger = m_triggerNames.begin(); trigger != m_triggerNames.end();
       ++trigger)
    if (part == trigger->substr(0, part_size)) {
      oOptions.push_back(std::make_pair(std::make_shared<std::string>(*trigger),
                                        trigger->substr(part_size, trigger->size() - part_size)));
    }
}

//
// static member functions
//
