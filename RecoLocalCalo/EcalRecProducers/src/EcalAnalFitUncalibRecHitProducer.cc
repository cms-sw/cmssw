/** \class EcalAnalFitUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes with the analytical fit method
 *
  *  $Id: EcalAnalFitUncalibRecHitProducer.cc,v 1.14 2007/08/06 14:47:45 innocent Exp $
  *  $Date: 2007/08/06 14:47:45 $
  *  $Revision: 1.14 $
  *  \author Shahram Rahatlou, University of Rome & INFN, Sept 2005
  *
  */
#include "RecoLocalCalo/EcalRecProducers/interface/EcalAnalFitUncalibRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/Common/interface/Handle.h"

#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
//#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//#include "CLHEP/Matrix/Matrix.h"
//#include "CLHEP/Matrix/SymMatrix.h"
#include <vector>

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

EcalAnalFitUncalibRecHitProducer::EcalAnalFitUncalibRecHitProducer(const edm::ParameterSet& ps) {

   EBdigiCollection_ = ps.getParameter<edm::InputTag>("EBdigiCollection");
   EEdigiCollection_ = ps.getParameter<edm::InputTag>("EEdigiCollection");
   EBhitCollection_  = ps.getParameter<std::string>("EBhitCollection");
   EEhitCollection_  = ps.getParameter<std::string>("EEhitCollection");
   //   nMaxPrintout_   = ps.getUntrackedParameter<int>("nMaxPrintout",10);
   produces< EBUncalibratedRecHitCollection >(EBhitCollection_);
   produces< EEUncalibratedRecHitCollection >(EEhitCollection_);
   //   nEvt_ = 0; // reset local event counter
}

EcalAnalFitUncalibRecHitProducer::~EcalAnalFitUncalibRecHitProducer() {
}

void
EcalAnalFitUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

   using namespace edm;

   //   nEvt_++;

   Handle< EBDigiCollection > pEBDigis;
   Handle< EEDigiCollection > pEEDigis;

   const EBDigiCollection* EBdigis =0;
   const EEDigiCollection* EEdigis =0;
   
   try {
     evt.getByLabel( EBdigiCollection_, pEBDigis);
     //evt.getByLabel( digiProducer_, pEBDigis);
     EBdigis = pEBDigis.product(); // get a ptr to the produc
     edm::LogInfo("EcalUncalibRecHitInfo") << "EcalAnalFitUncalibRecHitProducer: total # EBdigis: " << EBdigis->size() ;
   } catch (...) {
     //     edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EBdigiCollection_.c_str() ;
   }

   try {
     evt.getByLabel( EEdigiCollection_, pEEDigis);
     //evt.getByLabel( digiProducer_, pEEDigis);
     EEdigis = pEEDigis.product(); // get a ptr to the product
     edm::LogInfo("EcalUncalibRecHitInfo") << "EcalAnalFitUncalibRecHitProducer: total # EEdigis: " << EEdigis->size() ;
   } catch (...) {
     //     edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EEdigiCollection_.c_str() ;
   }

   // Gain Ratios
   LogDebug("EcalUncalibRecHitDebug") << "fetching gainRatios....";
   edm::ESHandle<EcalGainRatios> pRatio;
   es.get<EcalGainRatiosRcd>().get(pRatio);
   //const EcalGainRatios::EcalGainRatioMap& gainMap = pRatio.product()->getMap(); // map of gain ratios
   const EcalGainRatioMap & gainMap = pRatio.product()->getMap(); // map of gain ratios


   // fetch the pedestals from the cond DB via EventSetup
   LogDebug("EcalUncalibRecHitDebug") << "fetching pedestals....";
   edm::ESHandle<EcalPedestals> pedHandle;
   es.get<EcalPedestalsRcd>().get( pedHandle );
   //const EcalPedestalsMap & pedMap = pedHandle.product()->m_pedestals; // map of pedestals
   const EcalPedestalsMap & pedMap = pedHandle.product()->getMap(); // map of pedestals
   LogDebug("EcalUncalibRecHitDebug") << "done." ;

//    if(!counterExceeded()) 
//      {


//      }


   // collection of reco'ed ampltudes to put in the event

   std::auto_ptr< EBUncalibratedRecHitCollection > EBuncalibRechits( new EBUncalibratedRecHitCollection );
   std::auto_ptr< EEUncalibratedRecHitCollection > EEuncalibRechits( new EEUncalibratedRecHitCollection );

   EcalPedestalsMapIterator pedIter; // pedestal iterator
   EcalPedestals::Item aped; // pedestal object for a single xtal

   EcalGainRatioMap::const_iterator gainIter; // gain iterator
   EcalMGPAGainRatio aGain; // gain object for a single xtal

   if (EBdigis)
     {
       for(EBDigiCollection::const_iterator itdg = EBdigis->begin(); itdg != EBdigis->end(); ++itdg) {

	 // find pedestals for this channel
	 LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << EBDetId(itdg->id()) ;
	 pedIter = pedMap.find(itdg->id());
	 if( pedIter != pedMap.end() ) {
	   aped = (*pedIter);
	 } else {
	   edm::LogError("EcalUncalibRecHitError") << "error!! could not find pedestals for channel: " << EBDetId(itdg->id()) 
						   << "\n  no uncalib rechit will be made for this digi!"
	     ;
	   continue;
	 }
	 double pedVec[3];
	 pedVec[0]=aped.mean_x12;pedVec[1]=aped.mean_x6;pedVec[2]=aped.mean_x1;


	 // find gain ratios
	 LogDebug("EcalUncalibRecHitDebug") << "looking up gainRatios for crystal: " << EBDetId(itdg->id()) ;
	 gainIter = gainMap.find(itdg->id());
	 if( gainIter != gainMap.end() ) {
	   aGain = (*gainIter);
	 } else {
	   edm::LogError("EcalUncalibRecHitError") << "error!! could not find gain ratios for channel: " << EBDetId(itdg->id()) 
						   << "\n  no uncalib rechit will be made for this digi!"
	     ;
	   continue;
	 }
	 double gainRatios[3];
	 gainRatios[0]=1.;gainRatios[1]=aGain.gain12Over6();gainRatios[2]=aGain.gain6Over1()*aGain.gain12Over6();

	 EcalUncalibratedRecHit aHit =
	   EBalgo_.makeRecHit(*itdg, pedVec, gainRatios, 0 ,0);

	 EBuncalibRechits->push_back( aHit );
	 
	 if(aHit.amplitude()>0.) {
	   LogDebug("EcalUncalibRecHitInfo") << "EcalAnalFitUncalibRecHitProducer: processed EBDataFrame with id: "
					     << EBDetId(itdg->id()) << "\n"
					     << "uncalib rechit amplitude: " << aHit.amplitude()
	     ;
	 }
	 
       }
     }

   if (EEdigis)
     {
       for(EEDigiCollection::const_iterator itdg = EEdigis->begin(); itdg != EEdigis->end(); ++itdg) {

	 // find pedestals for this channel
	 LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << EEDetId(itdg->id()) ;
	 pedIter = pedMap.find(itdg->id());
	 if( pedIter != pedMap.end() ) {
	   aped = (*pedIter);
	 } else {
	   edm::LogError("EcalUncalibRecHitError") << "error!! could not find pedestals for channel: " << EEDetId(itdg->id())
						   << "\n  no uncalib rechit will be made for this digi!"
	     ;
	   continue;
	 }
	 double pedVec[3];
	 pedVec[0]=aped.mean_x12;pedVec[1]=aped.mean_x6;pedVec[2]=aped.mean_x1;

	 // find gain ratios
	 LogDebug("EcalUncalibRecHitDebug") << "looking up gainRatios for crystal: " << EEDetId(itdg->id()) ;
	 gainIter = gainMap.find(itdg->id());
	 if( gainIter != gainMap.end() ) {
	   aGain = (*gainIter);
	 } else {
	   edm::LogError("EcalUncalibRecHitError") << "error!! could not find gain ratios for channel: " << EEDetId(itdg->id())
						   << "\n  no uncalib rechit will be made for this digi!"
	     ;
	   continue;
	 }
	 double gainRatios[3];
	 gainRatios[0]=1.;gainRatios[1]=aGain.gain12Over6();gainRatios[2]=aGain.gain6Over1()*aGain.gain12Over6();
	 
	 EcalUncalibratedRecHit aHit =
	   EEalgo_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
	 EEuncalibRechits->push_back( aHit );
	 
	 if(aHit.amplitude()>0.) {
	   LogDebug("EcalUncalibRecHitInfo") << "EcalAnalFitUncalibRecHitProducer: processed EEDataFrame with id: "
					     << EEDetId(itdg->id()) << "\n"
					     << "uncalib rechit amplitude: " << aHit.amplitude()
	     ;
	 }
	 
       }
     }

   // put the collection of recunstructed hits in the event
   evt.put( EBuncalibRechits, EBhitCollection_ );
   evt.put( EEuncalibRechits, EEhitCollection_ );
}

