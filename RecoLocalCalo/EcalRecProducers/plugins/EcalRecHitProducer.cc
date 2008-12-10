/** \class EcalRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id: EcalRecHitProducer.cc,v 1.19 2008/02/20 14:28:38 meridian Exp $
 *  $Date: 2008/02/20 14:28:38 $
 *  $Revision: 1.19 $
 *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
 *
 **/
#include "RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerFactory.h"

EcalRecHitProducer::EcalRecHitProducer(const edm::ParameterSet& ps)
{
        ebUncalibRecHitCollection_ = ps.getParameter<edm::InputTag>("EBuncalibRecHitCollection");
        eeUncalibRecHitCollection_ = ps.getParameter<edm::InputTag>("EEuncalibRecHitCollection");
        ebRechitCollection_        = ps.getParameter<std::string>("EBrechitCollection");
        eeRechitCollection_        = ps.getParameter<std::string>("EErechitCollection");

        produces< EBRecHitCollection >(ebRechitCollection_);
        produces< EERecHitCollection >(eeRechitCollection_);

        std::string componentType = ps.getParameter<std::string>("algo");
        worker_ = EcalRecHitWorkerFactory::get()->create(componentType, ps);
}

EcalRecHitProducer::~EcalRecHitProducer()
{
}

void
EcalRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
        using namespace edm;

        Handle< EBUncalibratedRecHitCollection > pEBUncalibRecHits;
        Handle< EEUncalibratedRecHitCollection > pEEUncalibRecHits;

        const EBUncalibratedRecHitCollection*  ebUncalibRecHits = 0;
        const EEUncalibratedRecHitCollection*  eeUncalibRecHits = 0; 

        // get the barrel uncalib rechit collection
        if ( ebUncalibRecHitCollection_.label() != "" && ebUncalibRecHitCollection_.instance() != "" ) {
                evt.getByLabel( ebUncalibRecHitCollection_, pEBUncalibRecHits);
                if ( pEBUncalibRecHits.isValid() ) {
                        ebUncalibRecHits = pEBUncalibRecHits.product();
                        LogDebug("EcalRecHitDebug") << "total # EB uncalibrated rechits: " << ebUncalibRecHits->size();
                } else {
                        edm::LogError("EcalRecHitError") << "Error! can't get the product " << ebUncalibRecHitCollection_;
                }
        }

        if ( eeUncalibRecHitCollection_.label() != "" && eeUncalibRecHitCollection_.instance() != "" ) {
                evt.getByLabel( eeUncalibRecHitCollection_, pEEUncalibRecHits);
                if ( pEEUncalibRecHits.isValid() ) {
                        eeUncalibRecHits = pEEUncalibRecHits.product(); // get a ptr to the product
                        LogDebug("EcalRecHitDebug") << "total # EE uncalibrated rechits: " << eeUncalibRecHits->size();
                } else {
                        edm::LogError("EcalRecHitError") << "Error! can't get the product " << eeUncalibRecHitCollection_;
                }
        }

        // collection of rechits to put in the event
        std::auto_ptr< EBRecHitCollection > ebRecHits( new EBRecHitCollection );
        std::auto_ptr< EERecHitCollection > eeRecHits( new EERecHitCollection );

        worker_->set(es);

        if (ebUncalibRecHits)
        {
                // loop over uncalibrated rechits to make calibrated ones
                for(EBUncalibratedRecHitCollection::const_iterator it  = ebUncalibRecHits->begin(); it != ebUncalibRecHits->end(); ++it) {
                        worker_->run(evt, *it, *ebRecHits);
                }
        }

        if (eeUncalibRecHits)
        {
                // loop over uncalibrated rechits to make calibrated ones
                for(EEUncalibratedRecHitCollection::const_iterator it  = eeUncalibRecHits->begin(); it != eeUncalibRecHits->end(); ++it) {
                        worker_->run(evt, *it, *eeRecHits);
                }
        }

        // put the collection of recunstructed hits in the event   
        LogInfo("EcalRecHitInfo") << "total # EB calibrated rechits: " << ebRecHits->size();
        LogInfo("EcalRecHitInfo") << "total # EE calibrated rechits: " << eeRecHits->size();

        evt.put( ebRecHits, ebRechitCollection_ );
        evt.put( eeRecHits, eeRechitCollection_ );
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_ANOTHER_FWK_MODULE( EcalRecHitProducer );
