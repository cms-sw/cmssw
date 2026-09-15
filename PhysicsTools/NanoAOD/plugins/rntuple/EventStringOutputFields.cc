#include "FWCore/Framework/interface/EventForOutput.h"
#include "EventStringOutputFields.h"

void EventStringOutputFields::registerToken(const edm::EDGetToken &token) { m_tokens.push_back(token); }

void EventStringOutputFields::createFields(ROOT::RNTupleModel &model) {
  // With nothing kept there is no field at all, as in the TTree module. A token whose string is
  // always empty still gets one -- the ordinary-MC case -- because RNTuple has to commit to the
  // schema before seeing any event, where the TTree module can wait and then create no branch.
  if (m_tokens.empty()) {
    return;
  }
  m_evstrings = RNTupleFieldPtr<std::vector<std::string>>("EventStrings", "", model);
}

void EventStringOutputFields::bind(ROOT::REntry &entry) const {
  if (m_tokens.empty()) {
    return;
  }
  m_evstrings.bind(entry);
}

void EventStringOutputFields::fill(const edm::EventForOutput &iEvent) {
  if (m_tokens.empty()) {
    return;
  }
  // Read on every event rather than caching at lumi boundaries: a value has to be written for every
  // entry anyway, and the producer makes no promise that genModel is constant within a lumi.
  m_buffer.clear();
  edm::Handle<std::string> handle;
  for (const auto &token : m_tokens) {
    iEvent.getByToken(token, handle);
    if (!handle->empty()) {
      m_buffer.push_back(*handle);
    }
  }
  m_evstrings.fill(m_buffer);
}
