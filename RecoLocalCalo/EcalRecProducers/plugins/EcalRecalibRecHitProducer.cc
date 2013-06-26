/** \class EcalRecalibRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id: EcalRecalibRecHitProducer.cc,v 1.7 2012/04/13 18:01:05 yangyong Exp $
 *  $Date: 2012/04/13 18:01:05 $
 *  $Revision: 1.7 $
 *  \author Federico Ferri, University of Milano Bicocca and INFN
 *
 **/

#include "RecoLocalCalo/EcalRecProducers/plugins/EcalRecalibRecHitProducer.h"
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



EcalRecalibRecHitProducer::EcalRecalibRecHitProducer(const edm::ParameterSet& ps) {

   EBRecHitCollection_ = ps.getParameter<edm::InputTag>("EBRecHitCollection");
   EERecHitCollection_ = ps.getParameter<edm::InputTag>("EERecHitCollection");
   EBRecalibRecHitCollection_        = ps.getParameter<std::string>("EBRecalibRecHitCollection");
   EERecalibRecHitCollection_        = ps.getParameter<std::string>("EERecalibRecHitCollection");
   doEnergyScale_             = ps.getParameter<bool>("doEnergyScale");
   doIntercalib_              = ps.getParameter<bool>("doIntercalib");
   doLaserCorrections_        = ps.getParameter<bool>("doLaserCorrections");

   doEnergyScaleInverse_             = ps.getParameter<bool>("doEnergyScaleInverse");
   doIntercalibInverse_ = ps.getParameter<bool>("doIntercalibInverse");
   doLaserCorrectionsInverse_        = ps.getParameter<bool>("doLaserCorrectionsInverse");

   EBalgo_ = new EcalRecHitSimpleAlgo();
   EEalgo_ = new EcalRecHitSimpleAlgo();

   produces< EBRecHitCollection >(EBRecalibRecHitCollection_);
   produces< EERecHitCollection >(EERecalibRecHitCollection_);
}

EcalRecalibRecHitProducer::~EcalRecalibRecHitProducer() {

  if (EBalgo_) delete EBalgo_;
  if (EEalgo_) delete EEalgo_;

}

void EcalRecalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es)
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
        std::auto_ptr< EBRecHitCollection > EBRecalibRecHits( new EBRecHitCollection );
        std::auto_ptr< EERecHitCollection > EERecalibRecHits( new EERecHitCollection );

        // now fetch all conditions we need to make rechits
        // ADC to GeV constant
        edm::ESHandle<EcalADCToGeVConstant> pAgc;
        const EcalADCToGeVConstant *agc = 0;
        float agc_eb = 1.;
        float agc_ee = 1.;
        if (doEnergyScale_) {
                es.get<EcalADCToGeVConstantRcd>().get(pAgc);
                agc = pAgc.product();
                // use this value in the algorithm
                agc_eb = float(agc->getEBValue());
                agc_ee = float(agc->getEEValue());
        }
        // Intercalib constants
        edm::ESHandle<EcalIntercalibConstants> pIcal;
        const EcalIntercalibConstants *ical = 0;
        if (doIntercalib_) {
                es.get<EcalIntercalibConstantsRcd>().get(pIcal);
                ical = pIcal.product();
        }
        // Laser corrections
        edm::ESHandle<EcalLaserDbService> pLaser;
        es.get<EcalLaserDbRecord>().get( pLaser );

	
	if(doEnergyScaleInverse_){
	  agc_eb = 1.0/agc_eb;
	  agc_ee = 1.0/agc_ee;
	}


        if (EBRecHits) {
                // loop over uncalibrated rechits to make calibrated ones
                for(EBRecHitCollection::const_iterator it  = EBRecHits->begin(); it != EBRecHits->end(); ++it) {

                        EcalIntercalibConstant icalconst = 1.;
                        if (doIntercalib_) {
                                // find intercalib constant for this xtal
                                const EcalIntercalibConstantMap &icalMap = ical->getMap();
                                EcalIntercalibConstantMap::const_iterator icalit = icalMap.find(it->id());
                                if( icalit!=icalMap.end() ){
                                        icalconst = (*icalit);
                                } else {
                                        edm::LogError("EcalRecHitError") << "No intercalib const found for xtal " << EBDetId(it->id()) << "! something wrong with EcalIntercalibConstants in your DB? "
                                                ;
                                }
                        }
                        // get laser coefficient
                        float lasercalib = 1;
                        if (doLaserCorrections_) {
                                lasercalib = pLaser->getLaserCorrection( EBDetId(it->id()), evt.time() );
                        }

                        // make the rechit and put in the output collection
                        // must implement op= for EcalRecHit
			
			if(doIntercalibInverse_){
			  icalconst = 1.0/icalconst;
			}
			if (doLaserCorrectionsInverse_){
			  lasercalib = 1.0/lasercalib;
			}

                        EcalRecHit aHit( (*it).id(), (*it).energy() * agc_eb * icalconst * lasercalib, (*it).time() );
                        EBRecalibRecHits->push_back( aHit );
                }
        }

        if (EERecHits)
        {
                // loop over uncalibrated rechits to make calibrated ones
                for(EERecHitCollection::const_iterator it  = EERecHits->begin();
                                it != EERecHits->end(); ++it) {

                        // find intercalib constant for this xtal
                        EcalIntercalibConstant icalconst = 1.;
                        if (doIntercalib_) {
                                const EcalIntercalibConstantMap &icalMap = ical->getMap();
                                EcalIntercalibConstantMap::const_iterator icalit=icalMap.find(it->id());
                                if( icalit!=icalMap.end() ) {
                                        icalconst = (*icalit);
                                } else {
                                        edm::LogError("EcalRecHitError") << "No intercalib const found for xtal " << EEDetId(it->id()) << "! something wrong with EcalIntercalibConstants in your DB? ";
                                }
                        }
                        // get laser coefficient
                        float lasercalib = 1;
                        if (doLaserCorrections_) {
                                lasercalib = pLaser->getLaserCorrection( EEDetId(it->id()), evt.time() );
                        }

                        // make the rechit and put in the output collection
                        // must implement op= for EcalRecHit
                        //EcalRecHit aHit( EEalgo_->makeRecHit(*it, icalconst * lasercalib) );

			if(doIntercalibInverse_){
			  icalconst = 1.0/icalconst;
			}
			if (doLaserCorrectionsInverse_){
			  lasercalib = 1.0/lasercalib;
			}
			
                        EcalRecHit aHit( (*it).id(), (*it).energy() * agc_ee * icalconst * lasercalib, (*it).time() );
                        EERecalibRecHits->push_back( aHit );
                }
        }
        // put the collection of recunstructed hits in the event   
        LogInfo("EcalRecalibRecHitInfo") << "total # EB re-calibrated rechits: " << EBRecalibRecHits->size();
        LogInfo("EcalRecalibRecHitInfo") << "total # EE re-calibrated rechits: " << EERecalibRecHits->size();

        evt.put( EBRecalibRecHits, EBRecalibRecHitCollection_ );
        evt.put( EERecalibRecHits, EERecalibRecHitCollection_ );
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( EcalRecalibRecHitProducer );
