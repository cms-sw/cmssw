#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "L1TriggerConfig/GctConfigProducers/interface/L1GctJfParamsConfigurer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GctJfParamsConfigurer::L1GctJfParamsConfigurer(const edm::ParameterSet& iConfig) :
  m_CenJetSeed(iConfig.getParameter<unsigned>("JetFinderCentralJetSeed")),
  m_FwdJetSeed(iConfig.getParameter<unsigned>("JetFinderForwardJetSeed")),
  m_TauJetSeed(iConfig.getParameter<unsigned>("JetFinderCentralJetSeed")), // no separate tau jet seed yet
  m_EtaBoundry(7) // not programmable!
{
                                
}


L1GctJfParamsConfigurer::~L1GctJfParamsConfigurer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
    

// ------------ methods called to produce the data  ------------
L1GctJfParamsConfigurer::JfParamsReturnType
L1GctJfParamsConfigurer::produceJfParams()
{
   boost::shared_ptr<L1GctJetFinderParams> pL1GctJetFinderParams =
     boost::shared_ptr<L1GctJetFinderParams> (new L1GctJetFinderParams(m_CenJetSeed,
                                                                       m_FwdJetSeed,
                                                                       m_TauJetSeed,
                                                                       m_EtaBoundry));

   return pL1GctJetFinderParams ;
}
