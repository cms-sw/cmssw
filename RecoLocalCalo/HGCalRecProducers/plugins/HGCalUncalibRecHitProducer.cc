#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalUncalibRecHitProducer.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalUncalibRecHitWorkerFactory.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

HGCalUncalibRecHitProducer::HGCalUncalibRecHitProducer(const edm::ParameterSet& ps)
{
        eeDigiCollection_ = ps.getParameter<edm::InputTag>("HGCEEdigiCollection");
        hefDigiCollection_ = ps.getParameter<edm::InputTag>("HGCHEFdigiCollection");
        hebDigiCollection_ = ps.getParameter<edm::InputTag>("HGCHEBdigiCollection");
        eeHitCollection_  = ps.getParameter<std::string>("HGCEEhitCollection");
        hefHitCollection_  = ps.getParameter<std::string>("HGCHEFhitCollection");
        hebHitCollection_  = ps.getParameter<std::string>("HGCHEBhitCollection");
        produces< HGCeeUncalibratedRecHitCollection >(eeHitCollection_);
        produces< HGChefUncalibratedRecHitCollection >(hefHitCollection_);
        produces< HGChebUncalibratedRecHitCollection >(hebHitCollection_);

        std::string componentType = ps.getParameter<std::string>("algo");
        worker_ = HGCalUncalibRecHitWorkerFactory::get()->create(componentType, ps);
}

HGCalUncalibRecHitProducer::~HGCalUncalibRecHitProducer()
{
        delete worker_;
}

void
HGCalUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

        using namespace edm;

        Handle< HGCEEDigiCollection > pHGCEEDigis;
        Handle< HGCHEDigiCollection > pHGCHEFDigis;
        Handle< HGCHEDigiCollection > pHGCHEBDigis;

        const HGCEEDigiCollection* eeDigis =0;
        const HGCHEDigiCollection* hefDigis =0;
        const HGCHEDigiCollection* hebDigis =0;


        if ( eeDigiCollection_.label() != "" && eeDigiCollection_.instance() != "" ) {
                evt.getByLabel( eeDigiCollection_, pHGCEEDigis);
                if ( pHGCEEDigis.isValid() ) {
                        eeDigis = pHGCEEDigis.product(); // get a ptr to the product
                        edm::LogInfo("HGCalUncalibRecHitInfo") << "total # eeDigis: " << eeDigis->size() ;
                } else {
                        edm::LogInfo("HGCalUncalibRecHitInfo") << "Info!? can't get the product " << eeDigiCollection_;
                }
        }

        if ( hefDigiCollection_.label() != "" && hefDigiCollection_.instance() != "" ) {
                evt.getByLabel( hefDigiCollection_, pHGCHEFDigis);
                if ( pHGCHEFDigis.isValid() ) {
                        hefDigis = pHGCHEFDigis.product(); // get a ptr to the product
                        edm::LogInfo("HGCalUncalibRecHitInfo") << "total # hefDigis: " << hefDigis->size() ;
                } else {
                        edm::LogInfo("HGCalUncalibRecHitInfo") << "Info!? can't get the product " << hefDigiCollection_;
                }
        }

        if ( hebDigiCollection_.label() != "" && hebDigiCollection_.instance() != "" ) {
                evt.getByLabel( hebDigiCollection_, pHGCHEBDigis);
                if ( pHGCHEBDigis.isValid() ) {
                        hebDigis = pHGCHEBDigis.product(); // get a ptr to the product
                        edm::LogInfo("HGCalUncalibRecHitInfo") << "total # hebDigis: " << hebDigis->size() ;
                } else {
                        edm::LogInfo("HGCalUncalibRecHitInfo") << "Info!? can't get the product " << hebDigiCollection_;
                }
        }

        // tranparently get things from event setup
        worker_->set(es);

        // prepare output
        std::auto_ptr< HGCeeUncalibratedRecHitCollection > eeUncalibRechits( new HGCeeUncalibratedRecHitCollection );
        std::auto_ptr< HGChefUncalibratedRecHitCollection > hefUncalibRechits( new HGChefUncalibratedRecHitCollection );
        std::auto_ptr< HGChefUncalibratedRecHitCollection > hebUncalibRechits( new HGChebUncalibratedRecHitCollection );

        // loop over HGCEE digis
        if (eeDigis)
        {
                eeUncalibRechits->reserve(eeDigis->size());
                for(HGCEEDigiCollection::const_iterator itdg = eeDigis->begin(); itdg != eeDigis->end(); ++itdg) {
                        worker_->run1(evt, itdg, *eeUncalibRechits);
                }
        }

        // loop over HGCHEF digis
        if (hefDigis)
        {
                hefUncalibRechits->reserve(hefDigis->size());
                for(HGCHEDigiCollection::const_iterator itdg = hefDigis->begin(); itdg != hefDigis->end(); ++itdg) {
                        worker_->run2(evt, itdg, *hefUncalibRechits);
                }
        }

        // loop over HGCHEB digis
        if (hebDigis)
        {
                hebUncalibRechits->reserve(hebDigis->size());
                for(HGCHEDigiCollection::const_iterator itdg = hebDigis->begin(); itdg != hebDigis->end(); ++itdg) {
                        worker_->run3(evt, itdg, *hebUncalibRechits);
                }
        }


        // put the collection of recunstructed hits in the event
        evt.put( eeUncalibRechits, eeHitCollection_ );
        evt.put( hefUncalibRechits, hefHitCollection_ );
        evt.put( hebUncalibRechits, hebHitCollection_ );
}

#include "FWCore/Framework/interface/MakerMacros.h"                                                                                                            
DEFINE_FWK_MODULE( HGCalUncalibRecHitProducer );
