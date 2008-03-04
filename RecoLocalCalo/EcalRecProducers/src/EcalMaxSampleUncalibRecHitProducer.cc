/** \class EcalMaxSampleUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes 
 *
 *  $Id: EcalMaxSampleUncalibRecHitProducer.cc,v 1.24 2007/08/06 15:03:39 innocent Exp $
 *  $Date: 2007/08/06 15:03:39 $
 *  $Revision: 1.1 $
 *  \author G. Franzoni, E. Di Marco
 *
 */
#include "RecoLocalCalo/EcalRecProducers/interface/EcalMaxSampleUncalibRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/Common/interface/Handle.h"

#include <iostream>
#include <iomanip>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <vector>

EcalMaxSampleUncalibRecHitProducer::EcalMaxSampleUncalibRecHitProducer(const edm::ParameterSet& ps) {

  EBdigiCollection_ = ps.getParameter<edm::InputTag>("EBdigiCollection");
  EEdigiCollection_ = ps.getParameter<edm::InputTag>("EEdigiCollection");
  EBhitCollection_  = ps.getParameter<std::string>("EBhitCollection");
  EEhitCollection_  = ps.getParameter<std::string>("EEhitCollection");
  produces< EBUncalibratedRecHitCollection >(EBhitCollection_);
  produces< EEUncalibratedRecHitCollection >(EEhitCollection_);
}

EcalMaxSampleUncalibRecHitProducer::~EcalMaxSampleUncalibRecHitProducer() {
}

void
EcalMaxSampleUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  using namespace edm;

  Handle< EBDigiCollection > pEBDigis;
  Handle< EEDigiCollection > pEEDigis;

  const EBDigiCollection* EBdigis =0;
  const EEDigiCollection* EEdigis =0;

  try {
    evt.getByLabel( EBdigiCollection_, pEBDigis);
    EBdigis = pEBDigis.product(); // get a ptr to the produc
    edm::LogInfo("EcalUncalibRecHitInfo") << "total # EBdigis: " << EBdigis->size() ;
  } catch (...) {
    // edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EBdigiCollection_.c_str() ;
  }

  try {
    evt.getByLabel( EEdigiCollection_, pEEDigis);
    EEdigis = pEEDigis.product(); // get a ptr to the product
    edm::LogInfo("EcalUncalibRecHitInfo") << "total # EEdigis: " << EEdigis->size() ;
  } catch (...) {
    //edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EEdigiCollection_.c_str() ;
  }

  // collection of reconstructed ampltudes to put in the event

  std::auto_ptr< EBUncalibratedRecHitCollection > EBuncalibRechits( new EBUncalibratedRecHitCollection );
  std::auto_ptr< EEUncalibratedRecHitCollection > EEuncalibRechits( new EEUncalibratedRecHitCollection );

  // loop over EB digis
  if (EBdigis)
    {
      EBuncalibRechits->reserve(EBdigis->size());
      for(EBDigiCollection::const_iterator itdg = EBdigis->begin(); itdg != EBdigis->end(); ++itdg) {

	EcalUncalibratedRecHit aHit =
	  EBalgo_.makeRecHit(*itdg, 0, 0, 0, 0 );
	EBuncalibRechits->push_back( aHit );

      }// end loop eb
    }// end if EB


  // loop over EB digis
  if (EEdigis)
    {
      EEuncalibRechits->reserve(EEdigis->size());

      for(EEDigiCollection::const_iterator itdg = EEdigis->begin(); itdg != EEdigis->end(); ++itdg) {

	EcalUncalibratedRecHit aHit =
	  EEalgo_.makeRecHit(*itdg, 0, 0, 0, 0 );
	EEuncalibRechits->push_back( aHit );

      }// loop EE
    }// if EE

  // put the collection of recunstructed hits in the event
  evt.put( EBuncalibRechits, EBhitCollection_ );
  evt.put( EEuncalibRechits, EEhitCollection_ );
}
