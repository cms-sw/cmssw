#include "FWCore/Services/interface/UpdaterService.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>

UpdaterService::UpdaterService(edm::ParameterSet const&, edm::ActivityRegistry& r) :
  theEventId(0) {
  r.watchPreProcessEvent(this, &UpdaterService::init);
  theInit();
}

UpdaterService::~UpdaterService() {
}

void UpdaterService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("UpdaterService", desc);
}

void UpdaterService::init(edm::EventID const& eId, edm::Timestamp const&) {
  theInit();
  theEventId = &eId;
}

void UpdaterService::theInit() {
  theCounts.clear();
}

bool UpdaterService::checkOnce(std::string tag) {
  bool answer = true;

  std::map<std::string, unsigned int>::iterator i = theCounts.find(tag);
  if(i != theCounts.end()) {
    ++i->second;
    answer = false;
  } else{
    theCounts[tag] = 1;
    answer = true;
  }

  if(theEventId) {
    LogDebug("UpdaterService")
      << "checking ONCE on tag: " << tag << "on run: " << theEventId->run()
      << " event: " << theEventId->event() << ((answer) ? " -> true" : " -> false");
  }
  return answer;
}

bool UpdaterService::check(std::string /*tag*/, std::string /*label*/) {
  return true;
}
