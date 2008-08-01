#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "L1TriggerConfig/GctConfigProducers/interface/L1GctConfigProducers.h"

#include "L1TriggerConfig/GctConfigProducers/interface/L1GctCalibFunConfigurer.h"
#include "L1TriggerConfig/GctConfigProducers/interface/L1GctJctSetupConfigurer.h"
#include "L1TriggerConfig/GctConfigProducers/interface/L1GctJfParamsConfigurer.h"

#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterNegativeEtaRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterPositiveEtaRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GctConfigProducers::L1GctConfigProducers(const edm::ParameterSet& iConfig) :
  m_CalibFunConf(new L1GctCalibFunConfigurer(iConfig)),
  m_JctSetupConfNegativeEta(new L1GctJctSetupConfigurer(iConfig.getParameter<edm::ParameterSet>("jetCounterSetup").getParameter< std::vector<edm::ParameterSet> >("jetCountersNegativeWheel"))),
  m_JctSetupConfPositiveEta(new L1GctJctSetupConfigurer(iConfig.getParameter<edm::ParameterSet>("jetCounterSetup").getParameter< std::vector<edm::ParameterSet> >("jetCountersPositiveWheel"))),
  m_JfParamsConf(new L1GctJfParamsConfigurer(iConfig))
{
   //the following lines are needed to tell the framework what
   // data is being produced
   setWhatProduced(this,&L1GctConfigProducers::produceCalibFun);
   setWhatProduced(this,&L1GctConfigProducers::produceJfParams);
   setWhatProduced(this,&L1GctConfigProducers::produceJCNegEta);
   setWhatProduced(this,&L1GctConfigProducers::produceJCPosEta);

   //now do what ever other initialization is needed

}


L1GctConfigProducers::~L1GctConfigProducers()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  if (m_CalibFunConf != 0) { delete m_CalibFunConf; }
  if (m_JctSetupConfNegativeEta != 0) { delete m_JctSetupConfNegativeEta; }
  if (m_JctSetupConfPositiveEta != 0) { delete m_JctSetupConfPositiveEta; }
  if (m_JfParamsConf != 0) { delete m_JfParamsConf; }

}

// The producer methods are handled by the "Configurer" objects

L1GctConfigProducers::
CalibFunReturnType L1GctConfigProducers::produceCalibFun(const L1GctJetCalibFunRcd& aRcd)
        {
	  const L1CaloGeometryRecord& geomRcd =
	    aRcd.getRecord< L1CaloGeometryRecord >() ;
	  edm::ESHandle< L1CaloGeometry > geom ;
	  geomRcd.get( geom ) ;
	  return m_CalibFunConf->produceCalibFun( geom.product() ); }

L1GctConfigProducers::
JCtSetupReturnType L1GctConfigProducers::produceJCNegEta(const L1GctJetCounterNegativeEtaRcd&)
        { return m_JctSetupConfNegativeEta->produceJctSetup(); }

L1GctConfigProducers::
JCtSetupReturnType L1GctConfigProducers::produceJCPosEta(const L1GctJetCounterPositiveEtaRcd&)
        { return m_JctSetupConfPositiveEta->produceJctSetup(); }

L1GctConfigProducers::
JfParamsReturnType L1GctConfigProducers::produceJfParams(const L1GctJetFinderParamsRcd&)
        { return m_JfParamsConf->produceJfParams(); }




//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1GctConfigProducers);
