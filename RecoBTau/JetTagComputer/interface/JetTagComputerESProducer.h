#ifndef RecoBTau_JetTagComputerESProducer_h
#define RecoBTau_JetTagComputerESProducer_h

#include <string>
#include <memory>
#include <type_traits>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

template <typename ConcreteJetTagComputer>
class JetTagComputerESProducer : public edm::ESProducer {
private:
  // check that the template parameter inherits from JetTagComputer
  static_assert(std::is_convertible_v<ConcreteJetTagComputer*, JetTagComputer*>);

public:
  JetTagComputerESProducer(const edm::ParameterSet& pset) : m_pset(pset) {
    setWhatProduced(this, m_pset.getParameter<std::string>("@module_label"));
  }

  ~JetTagComputerESProducer() override {}

  std::unique_ptr<JetTagComputer> produce(const JetTagComputerRecord& record) {
    std::unique_ptr<JetTagComputer> jetTagComputer = std::make_unique<ConcreteJetTagComputer>(m_pset);
    jetTagComputer->initialize(record);
    jetTagComputer->setupDone();
    return jetTagComputer;
  }

private:
  edm::ParameterSet m_pset;
};

#endif  // RecoBTau_JetTagComputerESProducer_h
