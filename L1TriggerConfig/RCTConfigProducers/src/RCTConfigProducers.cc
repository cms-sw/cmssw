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
// $Id: RCTConfigProducers.cc,v 1.12 2010/05/07 14:41:48 bachtis Exp $
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
#include "CondFormats/DataRecord/interface/L1RCTNoisyChannelMaskRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTNoisyChannelMask.h"

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
  boost::shared_ptr<L1RCTNoisyChannelMask> produceL1RCTNoisyChannelMask(const L1RCTNoisyChannelMaskRcd&);

private:
  // ----------member data ---------------------------
  L1RCTParameters rctParameters;
  L1RCTChannelMask rctChannelMask;
  L1RCTNoisyChannelMask rctNoisyChannelMask;
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
  setWhatProduced(this, &RCTConfigProducers::produceL1RCTNoisyChannelMask);

   //now do what ever other initialization is needed
   rctParameters = 
     L1RCTParameters(iConfig.getParameter<double>("eGammaLSB"),
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
			 iConfig.getParameter<bool>("useCorrectionsLindsey"),
			 iConfig.getParameter<std::vector< double > >("eGammaECalScaleFactors"),
                         iConfig.getParameter<std::vector< double > >("eGammaHCalScaleFactors"),
                         iConfig.getParameter<std::vector< double > >("jetMETECalScaleFactors"),
                         iConfig.getParameter<std::vector< double > >("jetMETHCalScaleFactors"),
			 iConfig.getParameter<std::vector< double > >("ecal_calib_Lindsey"),
                         iConfig.getParameter<std::vector< double > >("hcal_calib_Lindsey"),
                         iConfig.getParameter<std::vector< double > >("hcal_high_calib_Lindsey"),
                         iConfig.getParameter<std::vector< double > >("cross_terms_Lindsey"),
			 iConfig.getParameter<std::vector< double > >("HoverE_low_Lindsey"),
			 iConfig.getParameter<std::vector< double > >("HoverE_high_Lindsey")
			 );



   // value of true if channel is masked, false if not masked
   for (int i = 0; i < 18; i++)
     {
       for (int j = 0; j < 2; j++)
	 {
	   for (int k = 0; k < 28; k++)
	     {
	       rctChannelMask.ecalMask[i][j][k] = false;
	       rctChannelMask.hcalMask[i][j][k] = false;
	     }
	   for (int k = 0; k < 4; k++)
	     {
	       rctChannelMask.hfMask[i][j][k] = false;
	     }
	 }
     }



   //Now the hot tower mask
   
   //first the thresholds
   
   rctNoisyChannelMask.ecalThreshold = 0.0;
   rctNoisyChannelMask.hcalThreshold = 0.0;
   rctNoisyChannelMask.hfThreshold = 0.0;

   for (int i = 0; i < 18; i++)
     {
       for (int j = 0; j < 2; j++)
	 {
	   for (int k = 0; k < 28; k++)
	     {
	       rctNoisyChannelMask.ecalMask[i][j][k] = false;
	       rctNoisyChannelMask.hcalMask[i][j][k] = false;
	     }
	   for (int k = 0; k < 4; k++)
	     {
	       rctNoisyChannelMask.hfMask[i][j][k] = false;
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
     boost::shared_ptr<L1RCTParameters>( new L1RCTParameters( rctParameters ) ) ;
   return pL1RCTParameters ;
   //return products( pL1RCTParameters, pL1RCTChannelMask );
}

boost::shared_ptr<L1RCTChannelMask>
RCTConfigProducers::produceL1RCTChannelMask(const L1RCTChannelMaskRcd& iRecord)
{
  using namespace edm::es;
   boost::shared_ptr<L1RCTChannelMask> pL1RCTChannelMask =
     boost::shared_ptr<L1RCTChannelMask>( new L1RCTChannelMask( rctChannelMask ) ) ;
   return pL1RCTChannelMask ;
}

boost::shared_ptr<L1RCTNoisyChannelMask>
RCTConfigProducers::produceL1RCTNoisyChannelMask(const L1RCTNoisyChannelMaskRcd& iRecord)
{
  using namespace edm::es;
   boost::shared_ptr<L1RCTNoisyChannelMask> pL1RCTChannelMask =
     boost::shared_ptr<L1RCTNoisyChannelMask>( new L1RCTNoisyChannelMask( rctNoisyChannelMask ) ) ;
   return pL1RCTChannelMask ;
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RCTConfigProducers);
