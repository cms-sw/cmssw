#ifndef RecoBTau_JetTagComputerESProducer_h
#define RecoBTau_JetTagComputerESProducer_h

#include <string>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

template <typename ConcreteJetTagComputer>
class JetTagComputerESProducer: public edm::ESProducer {
private:
  // check that the template parameter inherits from JetTagComputer
  BOOST_STATIC_ASSERT((boost::is_convertible<ConcreteJetTagComputer*,JetTagComputer*>::value));
  
public:
  JetTagComputerESProducer(const edm::ParameterSet & pset) : m_pset(pset) {
    setWhatProduced(this, m_pset.getParameter<std::string>("@module_label") );

    m_jetTagComputer = boost::shared_ptr<JetTagComputer>( new ConcreteJetTagComputer( m_pset ) );
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

#endif // RecoBTau_JetTagComputerESProducer_h
