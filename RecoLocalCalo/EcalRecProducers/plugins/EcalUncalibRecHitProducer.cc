#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitProducer.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

EcalUncalibRecHitProducer::EcalUncalibRecHitProducer(const edm::ParameterSet& ps)
{
        ebDigiCollection_ = ps.getParameter<edm::InputTag>("EBdigiCollection");
        eeDigiCollection_ = ps.getParameter<edm::InputTag>("EEdigiCollection");
        ekDigiCollection_ = edm::InputTag("simEcalGlobalZeroSuppression","ekDigis"); //ps.getParameter<edm::InputTag>("EKdigiCollection");
        ebHitCollection_  = ps.getParameter<std::string>("EBhitCollection");
        eeHitCollection_  = ps.getParameter<std::string>("EEhitCollection");
        ekHitCollection_  = ps.getParameter<std::string>("EKhitCollection");
        produces< EBUncalibratedRecHitCollection >(ebHitCollection_);
        produces< EEUncalibratedRecHitCollection >(eeHitCollection_);
        produces< EKUncalibratedRecHitCollection >(ekHitCollection_);

        std::string componentType = ps.getParameter<std::string>("algo");
        worker_ = EcalUncalibRecHitWorkerFactory::get()->create(componentType, ps);
}

EcalUncalibRecHitProducer::~EcalUncalibRecHitProducer()
{
        delete worker_;
}

void
EcalUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

        using namespace edm;

        Handle< EBDigiCollection > pEBDigis;
        Handle< EEDigiCollection > pEEDigis;
        Handle< EKDigiCollection > pEKDigis;

        const EBDigiCollection* ebDigis =0;
        const EEDigiCollection* eeDigis =0;
        const EKDigiCollection* ekDigis =0;

        if ( ebDigiCollection_.label() != "" && ebDigiCollection_.instance() != "" ) {
                evt.getByLabel( ebDigiCollection_, pEBDigis);
                //evt.getByLabel( digiProducer_, pEBDigis);
                if ( pEBDigis.isValid() ) {
                        ebDigis = pEBDigis.product(); // get a ptr to the produc
                        edm::LogInfo("EcalUncalibRecHitInfo") << "total # ebDigis: " << ebDigis->size() ;
                } else {
                        edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << ebDigiCollection_;
                }
        }

        if ( eeDigiCollection_.label() != "" && eeDigiCollection_.instance() != "" ) {
                evt.getByLabel( eeDigiCollection_, pEEDigis);
                //evt.getByLabel( digiProducer_, pEEDigis);
                if ( pEEDigis.isValid() ) {
                        eeDigis = pEEDigis.product(); // get a ptr to the product
                        edm::LogInfo("EcalUncalibRecHitInfo") << "total # eeDigis: " << eeDigis->size() ;
                } else {
                        edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << eeDigiCollection_;
                }
        }

        if ( ekDigiCollection_.label() != "" && ekDigiCollection_.instance() != "" ) {
	  evt.getByLabel( ekDigiCollection_, pEKDigis);
	  if ( pEKDigis.isValid() ) {
	    ekDigis = pEKDigis.product(); // get a ptr to the product
	    edm::LogInfo("EcalUncalibRecHitInfo") << "total # ekDigis: " << ekDigis->size() ;
	  } else {
	    edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << ekDigiCollection_;
	  }
        }

        // tranparently get things from event setup
        worker_->set(es);

        // prepare output
        std::auto_ptr< EBUncalibratedRecHitCollection > ebUncalibRechits( new EBUncalibratedRecHitCollection );
        std::auto_ptr< EEUncalibratedRecHitCollection > eeUncalibRechits( new EEUncalibratedRecHitCollection );
        std::auto_ptr< EKUncalibratedRecHitCollection > ekUncalibRechits( new EKUncalibratedRecHitCollection );

        // loop over EB digis
        if (ebDigis)
        {
                ebUncalibRechits->reserve(ebDigis->size());
                for(EBDigiCollection::const_iterator itdg = ebDigis->begin(); itdg != ebDigis->end(); ++itdg) {
                        worker_->run(evt, itdg, *ebUncalibRechits);
                }
        }

        // loop over EB digis
        if (eeDigis)
        {
                eeUncalibRechits->reserve(eeDigis->size());
                for(EEDigiCollection::const_iterator itdg = eeDigis->begin(); itdg != eeDigis->end(); ++itdg) {
                        worker_->run(evt, itdg, *eeUncalibRechits);
                }
        }

        // loop over EK digis
        if (ekDigis){
	  ekUncalibRechits->reserve(ekDigis->size());
	  for(EKDigiCollection::const_iterator itdg = ekDigis->begin(); itdg != ekDigis->end(); ++itdg) {
	    worker_->run(evt, itdg, *ekUncalibRechits);
	  }
        }

        // put the collection of recunstructed hits in the event
        evt.put( ebUncalibRechits, ebHitCollection_ );
        evt.put( eeUncalibRechits, eeHitCollection_ );
        evt.put( ekUncalibRechits, ekHitCollection_ );
}

#include "FWCore/Framework/interface/MakerMacros.h"                                                                                                            
DEFINE_FWK_MODULE( EcalUncalibRecHitProducer );
