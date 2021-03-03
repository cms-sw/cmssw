#ifndef PhysicsTools_NanoAOD_TriggerOutputFields_h
#define PhysicsTools_NanoAOD_TriggerOutputFields_h

#include "RNTupleFieldPtr.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
  class EventForOutput;
  class TriggerResults;
}

class TriggerOutputFields {
public:
  TriggerOutputFields() = default;
  explicit TriggerOutputFields(const std::string& processName, const edm::EDGetToken &token)
      : m_token(token), m_lastRun(-1), m_processName(processName) {}
  void createFields(const edm::EventForOutput& event, RNTupleModel& model);
  void fill(const edm::EventForOutput& event);

private:
  static std::vector<std::string> getTriggerNames(const edm::TriggerResults& triggerResults);
  void makeUniqueFieldName(/*const*/ RNTupleModel& model, std::string& name);

  edm::EDGetToken m_token;
  long m_lastRun;
  std::string m_processName;
  std::vector<RNTupleFieldPtr<bool>> m_triggerFields;
  std::vector<int> m_triggerFieldIndices;
};

#endif
