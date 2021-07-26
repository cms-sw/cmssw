#ifndef PhysicsTools_NanoAOD_TriggerOutputFields_h
#define PhysicsTools_NanoAOD_TriggerOutputFields_h

#include "RNTupleFieldPtr.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
  class EventForOutput;
  class TriggerResults;
}  // namespace edm

class TriggerFieldPtr {
public:
  TriggerFieldPtr() = default;
  TriggerFieldPtr(std::string name, int index, std::string fieldName, std::string fieldDesc, RNTupleModel& model);
  void fill(const edm::TriggerResults& triggers);
  const std::string& getTriggerName() const { return m_triggerName; }
  void setIndex(int newIndex) { m_triggerIndex = newIndex; }

private:
  RNTupleFieldPtr<bool> m_field;
  // The trigger results name extracted from the TriggerResults with version suffixes trimmed
  std::string m_triggerName;
  int m_triggerIndex = -1;
};

class TriggerOutputFields {
public:
  TriggerOutputFields() = default;
  explicit TriggerOutputFields(const std::string& processName, const edm::EDGetToken& token)
      : m_token(token), m_lastRun(-1), m_processName(processName) {}
  void createFields(const edm::EventForOutput& event, RNTupleModel& model);
  void fill(const edm::EventForOutput& event);

private:
  static std::vector<std::string> getTriggerNames(const edm::TriggerResults& triggerResults);
  // Update trigger field information on run boundaries
  void updateTriggerFields(const edm::TriggerResults& triggerResults);
  void makeUniqueFieldName(/*const*/ RNTupleModel& model, std::string& name);

  edm::EDGetToken m_token;
  long m_lastRun;
  std::string m_processName;
  std::vector<TriggerFieldPtr> m_triggerFields;
};

#endif
