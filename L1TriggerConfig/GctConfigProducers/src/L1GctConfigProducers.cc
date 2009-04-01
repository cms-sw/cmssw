#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "L1TriggerConfig/GctConfigProducers/interface/L1GctConfigProducers.h"

#include "L1TriggerConfig/GctConfigProducers/interface/L1GctJfParamsConfigurer.h"

#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"

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
  m_JfParamsConf(new L1GctJfParamsConfigurer(iConfig))
{
   //the following lines are needed to tell the framework what
   // data is being produced
   setWhatProduced(this,&L1GctConfigProducers::produceJfParams);
   setWhatProduced(this,&L1GctConfigProducers::produceChanMask);

   //now do what ever other initialization is needed

}


L1GctConfigProducers::~L1GctConfigProducers()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  if (m_JfParamsConf != 0) { delete m_JfParamsConf; }

}

// The producer methods are handled by the "Configurer" objects

L1GctConfigProducers::
JfParamsReturnType L1GctConfigProducers::produceJfParams(const L1GctJetFinderParamsRcd& aRcd)
        {
	  const L1CaloGeometryRecord& geomRcd = aRcd.getRecord< L1CaloGeometryRecord >() ;
	  edm::ESHandle< L1CaloGeometry > geom ;
	  geomRcd.get( geom ) ;
	  return m_JfParamsConf->produceJfParams( geom.product() ); }

L1GctConfigProducers::
ChanMaskReturnType L1GctConfigProducers::produceChanMask(const L1GctChannelMaskRcd&) {
  return boost::shared_ptr<L1GctChannelMask>(new L1GctChannelMask);
}




//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1GctConfigProducers);
