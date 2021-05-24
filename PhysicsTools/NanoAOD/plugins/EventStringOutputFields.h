#ifndef PhysicsTools_NanoAOD_EventStringOutputFields_h
#define PhysicsTools_NanoAOD_EventStringOutputFields_h

#include <string>
#include <vector>
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

  void registerToken(const edm::EDGetToken &token);
  void createFields(RNTupleModel &model);
  void fill(const edm::EventForOutput &iEvent);
};

#endif
