#ifndef RecoBTau_JetTagComputerESProducer_h
#define RecoBTau_JetTagComputerESProducer_h

#include <string>
#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

template <typename ConcreteJetTagComputer>
class JetTagComputerESProducer: public edm::ESProducer {
public:
  JetTagComputerESProducer(const edm::ParameterSet & pset) : m_pset(pset) {
    setWhatProduced(this, m_pset.retrieve("ComponentName").getString());

    m_jetTagComputer = boost::shared_ptr<JetTagComputer>(
    new ConcreteJetTagComputer( pset.getParameter<edm::ParameterSet>("JetTagComputerPSet" ) ));
  }
  
  virtual ~JetTagComputerESProducer() {
  }

  boost::shared_ptr<JetTagComputer> produce(const JetTagComputerRecord & record) {
    return m_jetTagComputer;
  }

private:
  boost::shared_ptr<JetTagComputer> m_jetTagComputer;
  edm::ParameterSet m_pset;
};

#endif // RecoBTag_SoftLepton_JetTagComputerESProducer_h
