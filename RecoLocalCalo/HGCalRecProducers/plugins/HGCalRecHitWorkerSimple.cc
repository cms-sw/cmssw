#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitWorkerSimple.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

HGCalRecHitWorkerSimple::HGCalRecHitWorkerSimple(const edm::ParameterSet&ps) :
        HGCalRecHitWorkerBaseClass(ps)
{
        rechitMaker_ = new HGCalRecHitSimpleAlgo();
	// HGCee constants 
	HGCEEmipInKeV_ =  ps.getParameter<double>("HGCEEmipInKeV");
	HGCEElsbInMIP_ =  ps.getParameter<double>("HGCEElsbInMIP");
	HGCEEmip2noise_ = ps.getParameter<double>("HGCEEmip2noise");
	hgceeADCtoGeV_ = HGCEEmipInKeV_ * HGCEElsbInMIP_/1000000; 
	// HGChef constants
	HGCHEFmipInKeV_ =  ps.getParameter<double>("HGCHEFmipInKeV");
	HGCHEFlsbInMIP_ =  ps.getParameter<double>("HGCHEFlsbInMIP");
	HGCHEFmip2noise_ = ps.getParameter<double>("HGCHEFmip2noise");
	hgchefADCtoGeV_ = HGCHEFmipInKeV_ * HGCHEFlsbInMIP_/1000000;
	// HGCheb constants
	HGCHEBmipInKeV_ =  ps.getParameter<double>("HGCHEBmipInKeV");
	HGCHEBlsbInMIP_ =  ps.getParameter<double>("HGCHEBlsbInMIP");
	HGCHEBmip2noise_ = ps.getParameter<double>("HGCHEBmip2noise");
	hgchebADCtoGeV_ = HGCHEBmipInKeV_ * HGCHEBlsbInMIP_/1000000;
	//        v_chstatus_ = ps.getParameter<std::vector<int> >("ChannelStatusToBeExcluded");
	//	v_DB_reco_flags_ = ps.getParameter<std::vector<int> >("flagsMapDBReco");
	//        killDeadChannels_ = ps.getParameter<bool>("killDeadChannels");
	// uncomment at more advanced simulation or data

}



void HGCalRecHitWorkerSimple::set(const edm::EventSetup& es)
{
  //        es.get<HGCalIntercalibConstantsRcd>().get(ical);
  //        es.get<HGCalTimeCalibConstantsRcd>().get(itime);
  //        es.get<HGCalTimeOffsetConstantRcd>().get(offtime);
  //        es.get<HGCalADCToGeVConstantRcd>().get(agc);
  //        es.get<HGCalChannelStatusRcd>().get(chStatus);
	// uncomment at more advanced simulation or data
}


bool
HGCalRecHitWorkerSimple::run( const edm::Event & evt,
                const HGCUncalibratedRecHit& uncalibRH,
                HGCRecHitCollection & result )
{
        DetId detid=uncalibRH.id();

        uint32_t recoFlag = 0;


	//	float offsetTime = 0; // the global time phase

        if ( detid.subdetId() == HGCEE ) {
	  rechitMaker_->setADCToGeVConstant(float(hgceeADCtoGeV_) );
	  //		offsetTime = offtime->getEEValue();
        } else if ( detid.subdetId() == HGCHEF ) {
	  rechitMaker_->setADCToGeVConstant(float(hgchefADCtoGeV_) );
	  //		offsetTime = offtime->getHEFValue();
	}else {
	  rechitMaker_->setADCToGeVConstant(float(hgchebADCtoGeV_) );
	  //	offsetTime = offtime->getHEBValue();
        }
	 
        // make the rechit and put in the output collection
	if (recoFlag == 0) {
          HGCRecHit myrechit( rechitMaker_->makeRecHit(uncalibRH, /*recoflags_*/ 0) );	

	  result.push_back(myrechit);
	}

        return true;
}

HGCalRecHitWorkerSimple::~HGCalRecHitWorkerSimple(){

  delete rechitMaker_;
}


#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( HGCalRecHitWorkerFactory, HGCalRecHitWorkerSimple, "HGCalRecHitWorkerSimple" );
