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
  m_rgnEtLsb(iConfig.getParameter<double>("RctRegionEtLSB")),
  m_htLsb(iConfig.getParameter<double>("GctHtLSB")),
  m_CenJetSeed(iConfig.getParameter<double>("JetFinderCentralJetSeed")),
  m_FwdJetSeed(iConfig.getParameter<double>("JetFinderForwardJetSeed")),
  m_TauJetSeed(iConfig.getParameter<double>("JetFinderCentralJetSeed")), // no separate tau jet seed yet
  m_tauIsoThresh(iConfig.getParameter<double>("TauIsoEtThreshold")),
  m_htJetThresh(iConfig.getParameter<double>("HtJetEtThreshold")),
  m_mhtJetThresh(iConfig.getParameter<double>("MHtJetEtThreshold")),
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
     boost::shared_ptr<L1GctJetFinderParams> (new L1GctJetFinderParams(m_rgnEtLsb,
								       m_htLsb,
								       m_CenJetSeed,
                                                                       m_FwdJetSeed,
                                                                       m_TauJetSeed,
								       m_tauIsoThresh,
								       m_htJetThresh,
								       m_mhtJetThresh,
                                                                       m_EtaBoundry));

   return pL1GctJetFinderParams ;
}
