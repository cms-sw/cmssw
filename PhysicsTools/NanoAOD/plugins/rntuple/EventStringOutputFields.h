#ifndef PhysicsTools_NanoAOD_EventStringOutputFields_h
#define PhysicsTools_NanoAOD_EventStringOutputFields_h

#include <string>
#include <vector>
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <ROOT/RNTupleModel.hxx>

#include "RNTupleFieldPtr.h"

namespace edm {
  class EventForOutput;
}

class EventStringOutputFields {
public:
  EventStringOutputFields() = default;

  void registerToken(const edm::EDGetToken &token);
  void createFields(ROOT::RNTupleModel &model);
  void bind(ROOT::REntry &entry) const;
  void fill(const edm::EventForOutput &iEvent);

private:
  std::vector<edm::EDGetToken> m_tokens;
  RNTupleFieldPtr<std::vector<std::string>> m_evstrings;
  // Reused across events, as in the other vector-valued fields of this module.
  std::vector<std::string> m_buffer;
};

#endif
