#include "RecoLocalCalo/EcalRecProducers/plugins/ESRecHitProducer.h"

#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerFactory.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// ESRecHitProducer author : Chia-Ming, Kuo

ESRecHitProducer::ESRecHitProducer(edm::ParameterSet const& ps)
{
        digiCollection_ = ps.getParameter<edm::InputTag>("ESdigiCollection");
        rechitCollection_ = ps.getParameter<std::string>("ESrechitCollection");
        produces<ESRecHitCollection>(rechitCollection_);

        std::string componentType = ps.getParameter<std::string>("algo");
        worker_ = ESRecHitWorkerFactory::get()->create( componentType, ps );
}

ESRecHitProducer::~ESRecHitProducer()
{
        //  delete algo_;
}

void ESRecHitProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
        // Get input
        edm::Handle<ESDigiCollection> digiHandle;  
        const ESDigiCollection* digi=0;
        //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
        e.getByLabel( digiCollection_, digiHandle);
        digi=digiHandle.product();

        edm::LogInfo("ESRecHitInfo") << "total # ESdigis: " << digi->size() ;  
        // Create empty output
        std::auto_ptr<ESRecHitCollection> rec(new ESRecHitCollection );
        rec->reserve(digi->size()); 

        // when algo parameters will be taken from the DB
        // the set will retrieve appropriate field from the EventSetup
        worker_->set( es );

        // run the algorithm
        ESDigiCollection::const_iterator i;
        for (i=digi->begin(); i!=digi->end(); i++) {    
                //rec->push_back(algo_->reconstruct(*i));
                worker_->run( e, i, *rec );

        }

        e.put(rec,rechitCollection_);
}

#include "FWCore/Framework/interface/MakerMacros.h"                                                                                                            
DEFINE_ANOTHER_FWK_MODULE( ESRecHitProducer );
