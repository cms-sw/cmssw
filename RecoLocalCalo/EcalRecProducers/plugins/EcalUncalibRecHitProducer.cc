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
        ebHitCollection_  = ps.getParameter<std::string>("EBhitCollection");
        eeHitCollection_  = ps.getParameter<std::string>("EEhitCollection");
        produces< EBUncalibratedRecHitCollection >(ebHitCollection_);
        produces< EEUncalibratedRecHitCollection >(eeHitCollection_);

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

        const EBDigiCollection* ebDigis =0;
        const EEDigiCollection* eeDigis =0;

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

        // tranparently get things from event setup
        worker_->set(es);

        // prepare output
        std::auto_ptr< EBUncalibratedRecHitCollection > ebUncalibRechits( new EBUncalibratedRecHitCollection );
        std::auto_ptr< EEUncalibratedRecHitCollection > eeUncalibRechits( new EEUncalibratedRecHitCollection );

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

        // put the collection of recunstructed hits in the event
        evt.put( ebUncalibRechits, ebHitCollection_ );
        evt.put( eeUncalibRechits, eeHitCollection_ );
}

#include "FWCore/Framework/interface/MakerMacros.h"                                                                                                            
DEFINE_FWK_MODULE( EcalUncalibRecHitProducer );
