// -*- C++ -*-
//
// Package:    RCTConfigProducers
// Class:      RCTConfigProducers
// 
/**\class RCTConfigProducers RCTConfigProducers.h L1TriggerConfig/RCTConfigProducers/src/RCTConfigProducers.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sridhara Dasu
//         Created:  Mon Jul 16 23:48:35 CEST 2007
// $Id: RCTConfigProducers.cc,v 1.7 2008/05/13 20:54:50 jleonard Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"

//
// class declaration
//

class RCTConfigProducers : public edm::ESProducer {
public:
  RCTConfigProducers(const edm::ParameterSet&);
  ~RCTConfigProducers();
  
  //typedef boost::shared_ptr<L1RCTParameters> ReturnType;
  //typedef edm::ESProducts< boost::shared_ptr<L1RCTParameters>, boost::shared_ptr<L1RCTChannelMask> > ReturnType;
  
  //ReturnType produce(const L1RCTParametersRcd&);
  boost::shared_ptr<L1RCTParameters> produceL1RCTParameters(const L1RCTParametersRcd&);
  boost::shared_ptr<L1RCTChannelMask> produceL1RCTChannelMask(const L1RCTChannelMaskRcd&);

private:
  // ----------member data ---------------------------
  L1RCTParameters *rctParameters;
  L1RCTChannelMask *rctChannelMask;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RCTConfigProducers::RCTConfigProducers(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   //setWhatProduced(this);
  setWhatProduced(this, &RCTConfigProducers::produceL1RCTParameters);
  setWhatProduced(this, &RCTConfigProducers::produceL1RCTChannelMask);

   //now do what ever other initialization is needed
   rctParameters = 
     new L1RCTParameters(iConfig.getParameter<double>("eGammaLSB"),
			 iConfig.getParameter<double>("jetMETLSB"),
			 iConfig.getParameter<double>("eMinForFGCut"),
			 iConfig.getParameter<double>("eMaxForFGCut"),
			 iConfig.getParameter<double>("hOeCut"),
			 iConfig.getParameter<double>("eMinForHoECut"),
			 iConfig.getParameter<double>("eMaxForHoECut"),
			 iConfig.getParameter<double>("hMinForHoECut"),
			 iConfig.getParameter<double>("eActivityCut"),
			 iConfig.getParameter<double>("hActivityCut"),
			 iConfig.getParameter<unsigned>("eicIsolationThreshold"),
			 iConfig.getParameter<unsigned>("jscQuietThresholdBarrel"),
			 iConfig.getParameter<unsigned>("jscQuietThresholdEndcap"),
			 iConfig.getParameter<bool>("noiseVetoHB"),
			 iConfig.getParameter<bool>("noiseVetoHEplus"),
			 iConfig.getParameter<bool>("noiseVetoHEminus"),
			 iConfig.getParameter<std::vector< double > >("eGammaECalScaleFactors"),
                         iConfig.getParameter<std::vector< double > >("eGammaHCalScaleFactors"),
                         iConfig.getParameter<std::vector< double > >("jetMETECalScaleFactors"),
                         iConfig.getParameter<std::vector< double > >("jetMETHCalScaleFactors")
			 );

   // value of true if channel is masked, false if not masked
   for (int i = 0; i < 18; i++)
     {
       for (int j = 0; j < 2; j++)
	 {
	   for (int k = 0; k < 28; k++)
	     {
	       rctChannelMask->ecalMask[i][j][k] = false;
	       rctChannelMask->hcalMask[i][j][k] = false;
	     }
	   for (int k = 0; k < 4; k++)
	     {
	       rctChannelMask->hfMask[i][j][k] = false;
	     }
	 }
     }
}


RCTConfigProducers::~RCTConfigProducers()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
//RCTConfigProducers::ReturnType
boost::shared_ptr<L1RCTParameters>
RCTConfigProducers::produceL1RCTParameters(const L1RCTParametersRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1RCTParameters> pL1RCTParameters =
     (boost::shared_ptr<L1RCTParameters>) rctParameters;
   return pL1RCTParameters ;
   //return products( pL1RCTParameters, pL1RCTChannelMask );
}

boost::shared_ptr<L1RCTChannelMask>
RCTConfigProducers::produceL1RCTChannelMask(const L1RCTChannelMaskRcd& iRecord)
{
  using namespace edm::es;
   boost::shared_ptr<L1RCTChannelMask> pL1RCTChannelMask =
     (boost::shared_ptr<L1RCTChannelMask>) rctChannelMask;
   return pL1RCTChannelMask ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RCTConfigProducers);
