#include "FWCore/Framework/interface/EventForOutput.h"
#include "EventStringOutputFields.h"

void EventStringOutputFields::registerToken(const edm::EDGetToken &token) { m_tokens.push_back(token); }

void EventStringOutputFields::createFields(RNTupleModel &model) {
  m_evstrings = RNTupleFieldPtr<std::vector<std::string>>("EventStrings", "", model);
}

void EventStringOutputFields::fill(const edm::EventForOutput &iEvent) {
  std::vector<std::string> evstrings;
  if (m_lastLumi != iEvent.id().luminosityBlock()) {
    edm::Handle<std::string> handle;
    for (const auto &t : m_tokens) {
      iEvent.getByToken(t, handle);
      const std::string &evstr = *handle;
      if (!evstr.empty()) {
        evstrings.push_back(evstr);
      }
    }
    m_lastLumi = iEvent.id().luminosityBlock();
  }
  m_evstrings.fill(evstrings);
}
