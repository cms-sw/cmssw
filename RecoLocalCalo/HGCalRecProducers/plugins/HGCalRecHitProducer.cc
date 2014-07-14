/** \class HGCalRecHitProducer
 *   produce HGCAL rechits from uncalibrated rechits
 *
 *  based on Ecal code
 *
 *  \author Valery Andreev
 *
 **/
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerFactory.h"


HGCalRecHitProducer::HGCalRecHitProducer(const edm::ParameterSet& ps)
{
        eeUncalibRecHitCollection_ = ps.getParameter<edm::InputTag>("HGCEEuncalibRecHitCollection");
        hefUncalibRecHitCollection_ = ps.getParameter<edm::InputTag>("HGCHEFuncalibRecHitCollection");
        hebUncalibRecHitCollection_ = ps.getParameter<edm::InputTag>("HGCHEBuncalibRecHitCollection");
        eeRechitCollection_        = ps.getParameter<std::string>("HGCEErechitCollection");
        hefRechitCollection_        = ps.getParameter<std::string>("HGCHEFrechitCollection");
        hebRechitCollection_        = ps.getParameter<std::string>("HGCHEBrechitCollection");

        produces< HGCeeRecHitCollection >(eeRechitCollection_);
        produces< HGChefRecHitCollection >(hefRechitCollection_);
        produces< HGChebRecHitCollection >(hebRechitCollection_);

        std::string componentType = ps.getParameter<std::string>("algo");
        worker_ = HGCalRecHitWorkerFactory::get()->create(componentType, ps);

}

HGCalRecHitProducer::~HGCalRecHitProducer()
{
        delete worker_;
}

void
HGCalRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
        using namespace edm;

        Handle< HGCeeUncalibratedRecHitCollection > pHGCeeUncalibRecHits;
        Handle< HGChefUncalibratedRecHitCollection > pHGChefUncalibRecHits;
        Handle< HGChebUncalibratedRecHitCollection > pHGChebUncalibRecHits;

        const HGCeeUncalibratedRecHitCollection*  eeUncalibRecHits = 0;
        const HGChefUncalibratedRecHitCollection*  hefUncalibRecHits = 0; 
        const HGChebUncalibratedRecHitCollection*  hebUncalibRecHits = 0; 

        // get the HGC uncalib rechit collection
        if ( eeUncalibRecHitCollection_.label() != "" && eeUncalibRecHitCollection_.instance() != "" ) {
                evt.getByLabel( eeUncalibRecHitCollection_, pHGCeeUncalibRecHits);
                if ( pHGCeeUncalibRecHits.isValid() ) {
                        eeUncalibRecHits = pHGCeeUncalibRecHits.product();
                        LogDebug("HGCalRecHitDebug") << "total # HGCee uncalibrated rechits: " << eeUncalibRecHits->size();
                } else {
                        edm::LogInfo("HGCalRecHitInfo") << "Info!? can't get the product " << eeUncalibRecHitCollection_;
                }
        }

        if ( hefUncalibRecHitCollection_.label() != "" && hefUncalibRecHitCollection_.instance() != "" ) {
                evt.getByLabel( hefUncalibRecHitCollection_, pHGChefUncalibRecHits);
                if ( pHGChefUncalibRecHits.isValid() ) {
                        hefUncalibRecHits = pHGChefUncalibRecHits.product();
                        LogDebug("HGCalRecHitDebug") << "total # HGChef uncalibrated rechits: " << hefUncalibRecHits->size();
                } else {
                        edm::LogInfo("HGCalRecHitInfo") << "Info!? can't get the product " << hefUncalibRecHitCollection_;
                }
        }

        if ( hebUncalibRecHitCollection_.label() != "" && hebUncalibRecHitCollection_.instance() != "" ) {
                evt.getByLabel( hebUncalibRecHitCollection_, pHGChebUncalibRecHits);
                if ( pHGChebUncalibRecHits.isValid() ) {
                        hebUncalibRecHits = pHGChebUncalibRecHits.product();
                        LogDebug("HGCalRecHitDebug") << "total # HGCheb uncalibrated rechits: " << hebUncalibRecHits->size();
                } else {
                        edm::LogInfo("HGCalRecHitInfo") << "Info!? can't get the product " << hebUncalibRecHitCollection_;
                }
        }

        // collection of rechits to put in the event
        std::auto_ptr< HGCeeRecHitCollection > eeRecHits( new HGCeeRecHitCollection );
        std::auto_ptr< HGChefRecHitCollection > hefRecHits( new HGChefRecHitCollection );
        std::auto_ptr< HGChebRecHitCollection > hebRecHits( new HGChebRecHitCollection );


        worker_->set(es);

         if (eeUncalibRecHits)
        {
                // loop over uncalibrated rechits to make calibrated ones
                for(HGCeeUncalibratedRecHitCollection::const_iterator it  = eeUncalibRecHits->begin(); it != eeUncalibRecHits->end(); ++it) {
                        worker_->run(evt, *it, *eeRecHits);
                }
        }

        if (hefUncalibRecHits)
        {
                // loop over uncalibrated rechits to make calibrated ones
                for(HGChefUncalibratedRecHitCollection::const_iterator it  = hefUncalibRecHits->begin(); it != hefUncalibRecHits->end(); ++it) {
                        worker_->run(evt, *it, *hefRecHits);
                }
        }

        if (hebUncalibRecHits)
        {
                // loop over uncalibrated rechits to make calibrated ones
                for(HGChebUncalibratedRecHitCollection::const_iterator it  = hebUncalibRecHits->begin(); it != hebUncalibRecHits->end(); ++it) {
                        worker_->run(evt, *it, *hebRecHits);
                }
        }

        // sort collections before attempting recovery, to avoid insertion of double recHits
        eeRecHits->sort();
        hefRecHits->sort();
        hebRecHits->sort();
        

        // put the collection of recunstructed hits in the event   
        LogInfo("HGCalRecHitInfo") << "total # HGCee calibrated rechits: " << eeRecHits->size();
        LogInfo("HGCalRecHitInfo") << "total # HGChef calibrated rechits: " << hefRecHits->size();
        LogInfo("HGCalRecHitInfo") << "total # HGCheb calibrated rechits: " << hebRecHits->size();

        evt.put( eeRecHits, eeRechitCollection_ );
        evt.put( hefRecHits, hefRechitCollection_ );
        evt.put( hebRecHits, hebRechitCollection_ );
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( HGCalRecHitProducer );
