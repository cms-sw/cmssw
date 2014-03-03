/** \class EcalDetailedTimeRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id: EcalDetailedTimeRecHitProducer.cc,v 1.7 2012/04/13 18:01:05 yangyong Exp $
 *  $Date: 2012/04/13 18:01:05 $
 *  $Revision: 1.7 $
 *  \author Federico Ferri, University of Milano Bicocca and INFN
 *
 **/

#include "RecoLocalCalo/EcalRecProducers/plugins/EcalDetailedTimeRecHitProducer.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <cmath>
#include <vector>

//#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
//#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTimeDigi.h"



EcalDetailedTimeRecHitProducer::EcalDetailedTimeRecHitProducer(const edm::ParameterSet& ps) {

   EBRecHitCollection_ = ps.getParameter<edm::InputTag>("EBRecHitCollection");
   EERecHitCollection_ = ps.getParameter<edm::InputTag>("EERecHitCollection");

   EBDetailedTimeRecHitCollection_        = ps.getParameter<std::string>("EBDetailedTimeRecHitCollection");
   EEDetailedTimeRecHitCollection_        = ps.getParameter<std::string>("EEDetailedTimeRecHitCollection");

   produces< EBRecHitCollection >(EBDetailedTimeRecHitCollection_);
   produces< EERecHitCollection >(EEDetailedTimeRecHitCollection_);
}

EcalDetailedTimeRecHitProducer::~EcalDetailedTimeRecHitProducer() {

}

void EcalDetailedTimeRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
        using namespace edm;
        Handle< EBRecHitCollection > pEBRecHits;
        Handle< EERecHitCollection > pEERecHits;

        const EBRecHitCollection*  EBRecHits = 0;
        const EERecHitCollection*  EERecHits = 0; 

	//        if ( EBRecHitCollection_.label() != "" && EBRecHitCollection_.instance() != "" ) {
        if ( EBRecHitCollection_.label() != "" ) {
                evt.getByLabel( EBRecHitCollection_, pEBRecHits);
                if ( pEBRecHits.isValid() ) {
                        EBRecHits = pEBRecHits.product(); // get a ptr to the product
#ifdef DEBUG
                        LogDebug("EcalRecHitDebug") << "total # EB rechits to be re-calibrated: " << EBRecHits->size();
#endif
                } else {
                        edm::LogError("EcalRecHitError") << "Error! can't get the product " << EBRecHitCollection_.label() ;
                }
        }

	//        if ( EERecHitCollection_.label() != "" && EERecHitCollection_.instance() != "" ) {
        if ( EERecHitCollection_.label() != ""  ) {
                evt.getByLabel( EERecHitCollection_, pEERecHits);
                if ( pEERecHits.isValid() ) {
                        EERecHits = pEERecHits.product(); // get a ptr to the product
#ifdef DEBUG
                        LogDebug("EcalRecHitDebug") << "total # EE uncalibrated rechits to be re-calibrated: " << EERecHits->size();
#endif
                } else {
                        edm::LogError("EcalRecHitError") << "Error! can't get the product " << EERecHitCollection_.label() ;
                }
        }

        // collection of rechits to put in the event
        std::auto_ptr< EBRecHitCollection > EBDetailedTimeRecHits( new EBRecHitCollection );
        std::auto_ptr< EERecHitCollection > EEDetailedTimeRecHits( new EERecHitCollection );


        if (EBRecHits) {
                // loop over uncalibrated rechits to make calibrated ones
                for(EBRecHitCollection::const_iterator it  = EBRecHits->begin(); it != EBRecHits->end(); ++it) {
		  EcalRecHit aHit( (*it) );
		  EBDetailedTimeRecHits->push_back( aHit );
                }
        }

        if (EERecHits)
        {
                // loop over uncalibrated rechits to make calibrated ones
                for(EERecHitCollection::const_iterator it  = EERecHits->begin();
                                it != EERecHits->end(); ++it) {
			
		  EcalRecHit aHit( *it );
		  EEDetailedTimeRecHits->push_back( aHit );
                }
        }
        // put the collection of recunstructed hits in the event   
        LogInfo("EcalDetailedTimeRecHitInfo") << "total # EB re-calibrated rechits: " << EBDetailedTimeRecHits->size();
        LogInfo("EcalDetailedTimeRecHitInfo") << "total # EE re-calibrated rechits: " << EEDetailedTimeRecHits->size();

        evt.put( EBDetailedTimeRecHits, EBDetailedTimeRecHitCollection_ );
        evt.put( EEDetailedTimeRecHits, EEDetailedTimeRecHitCollection_ );
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( EcalDetailedTimeRecHitProducer );
