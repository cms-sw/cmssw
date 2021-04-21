#ifndef PhysicsTools_NanoAOD_EventStringOutputFields_h
#define PhysicsTools_NanoAOD_EventStringOutputFields_h

#include <string>
#include <vector>
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <ROOT/RNTupleModel.hxx>
using ROOT::Experimental::RNTupleModel;

#include "RNTupleFieldPtr.h"

class EventStringOutputFields {
private:
  std::vector<edm::EDGetToken> m_tokens;
  RNTupleFieldPtr<std::vector<std::string>> m_evstrings;
  long m_lastLumi = -1;
public:
  EventStringOutputFields() = default;

  void register_token(const edm::EDGetToken &token) {
    m_tokens.push_back(token);
  }

  void createFields(RNTupleModel &model) {
    m_evstrings = RNTupleFieldPtr<std::vector<std::string>>("EventStrings", model);
  }

  void fill(const edm::EventForOutput &iEvent) {
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
};

#endif
