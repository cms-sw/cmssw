#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

EcalUncalibRecHitProducer::EcalUncalibRecHitProducer(const edm::ParameterSet& ps)
{
        ebHitCollection_  = ps.getParameter<std::string>("EBhitCollection");
        eeHitCollection_  = ps.getParameter<std::string>("EEhitCollection");
        produces< EBUncalibratedRecHitCollection >(ebHitCollection_);
        produces< EEUncalibratedRecHitCollection >(eeHitCollection_);

	ebDigiCollectionToken_ = consumes<EBDigiCollection>(ps.getParameter<edm::InputTag>("EBdigiCollection"));
	
	eeDigiCollectionToken_ = consumes<EEDigiCollection>(ps.getParameter<edm::InputTag>("EEdigiCollection"));

        std::string componentType = ps.getParameter<std::string>("algo");
	edm::ConsumesCollector c{consumesCollector()};
        worker_ = EcalUncalibRecHitWorkerFactory::get()->create(componentType, ps, c);
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


	evt.getByToken( ebDigiCollectionToken_, pEBDigis);		
	ebDigis = pEBDigis.product(); // get a ptr to the produc
	edm::LogInfo("EcalUncalibRecHitInfo") << "total # ebDigis: " << ebDigis->size() ;
                    
	evt.getByToken( eeDigiCollectionToken_, pEEDigis);            
	eeDigis = pEEDigis.product(); // get a ptr to the product
	edm::LogInfo("EcalUncalibRecHitInfo") << "total # eeDigis: " << eeDigis->size() ;
        

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
