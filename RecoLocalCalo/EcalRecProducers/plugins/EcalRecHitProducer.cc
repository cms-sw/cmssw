/** \class EcalRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
 *
 **/
#include "RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitProducer.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalCleaningAlgo.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerFactory.h"


EcalRecHitProducer::EcalRecHitProducer(const edm::ParameterSet& ps)
{
       
        ebRechitCollection_        = ps.getParameter<std::string>("EBrechitCollection");
        eeRechitCollection_        = ps.getParameter<std::string>("EErechitCollection");

        recoverEBIsolatedChannels_   = ps.getParameter<bool>("recoverEBIsolatedChannels");
        recoverEEIsolatedChannels_   = ps.getParameter<bool>("recoverEEIsolatedChannels");
        recoverEBVFE_                = ps.getParameter<bool>("recoverEBVFE");
        recoverEEVFE_                = ps.getParameter<bool>("recoverEEVFE");
        recoverEBFE_                 = ps.getParameter<bool>("recoverEBFE");
        recoverEEFE_                 = ps.getParameter<bool>("recoverEEFE");
        killDeadChannels_            = ps.getParameter<bool>("killDeadChannels");

   
        produces< EBRecHitCollection >(ebRechitCollection_);
        produces< EERecHitCollection >(eeRechitCollection_);

	
	ebUncalibRecHitToken_ = 
	  consumes<EBUncalibratedRecHitCollection>( ps.getParameter<edm::InputTag>("EBuncalibRecHitCollection"));
	
	eeUncalibRecHitToken_ = 
	  consumes<EEUncalibratedRecHitCollection>( ps.getParameter<edm::InputTag>("EEuncalibRecHitCollection"));

        ebDetIdToBeRecoveredToken_ = 
	  consumes<std::set<EBDetId>>(ps.getParameter<edm::InputTag>("ebDetIdToBeRecovered"));  
	
	eeDetIdToBeRecoveredToken_=        
	  consumes<std::set<EEDetId>>(ps.getParameter<edm::InputTag>("eeDetIdToBeRecovered"));

	ebFEToBeRecoveredToken_ = consumes<std::set<EcalTrigTowerDetId>>(ps.getParameter<edm::InputTag>("ebFEToBeRecovered"));

        eeFEToBeRecoveredToken_= consumes<std::set<EcalScDetId>>( ps.getParameter<edm::InputTag>("eeFEToBeRecovered"))   ;

        std::string componentType = ps.getParameter<std::string>("algo");
	edm::ConsumesCollector c{consumesCollector()};
        worker_ = EcalRecHitWorkerFactory::get()->create(componentType, ps, c);

        // to recover problematic channels
        componentType = ps.getParameter<std::string>("algoRecover");
        workerRecover_ = EcalRecHitWorkerFactory::get()->create(componentType, ps, c);

	edm::ParameterSet cleaningPs = 
	  ps.getParameter<edm::ParameterSet>("cleaningConfig");
	cleaningAlgo_ = new EcalCleaningAlgo(cleaningPs);
}

EcalRecHitProducer::~EcalRecHitProducer()
{
        delete worker_;
        delete workerRecover_;
	delete cleaningAlgo_;
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
       
	evt.getByToken( ebUncalibRecHitToken_, pEBUncalibRecHits);
	ebUncalibRecHits = pEBUncalibRecHits.product();
	LogDebug("EcalRecHitDebug") << "total # EB uncalibrated rechits: " << ebUncalibRecHits->size();
        

       
	evt.getByToken( eeUncalibRecHitToken_, pEEUncalibRecHits);
	eeUncalibRecHits = pEEUncalibRecHits.product(); // get a ptr to the product
	LogDebug("EcalRecHitDebug") << "total # EE uncalibrated rechits: " << eeUncalibRecHits->size();
       
        // collection of rechits to put in the event
        std::auto_ptr< EBRecHitCollection > ebRecHits( new EBRecHitCollection );
        std::auto_ptr< EERecHitCollection > eeRecHits( new EERecHitCollection );

        worker_->set(es);

        if ( recoverEBIsolatedChannels_ || recoverEEIsolatedChannels_
                || recoverEBFE_ || recoverEEFE_
                || recoverEBVFE_ || recoverEEVFE_
                || killDeadChannels_ ) {
                workerRecover_->set(es);
        }

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

        // sort collections before attempting recovery, to avoid insertion of double recHits
        ebRecHits->sort();
        eeRecHits->sort();
        
        if ( recoverEBIsolatedChannels_ || recoverEBFE_ || killDeadChannels_ )
        {
                edm::Handle< std::set<EBDetId> > pEBDetId;
                const std::set<EBDetId> * detIds = 0;
		evt.getByToken( ebDetIdToBeRecoveredToken_, pEBDetId);
		detIds = pEBDetId.product();
                       

                if ( detIds ) {
                        edm::ESHandle<EcalChannelStatus> chStatus;
                        es.get<EcalChannelStatusRcd>().get(chStatus);
                        for( std::set<EBDetId>::const_iterator it = detIds->begin(); it != detIds->end(); ++it ) {
                                // get channel status map to treat dead VFE separately
                                EcalChannelStatusMap::const_iterator chit = chStatus->find( *it );
                                EcalChannelStatusCode chStatusCode = 1;
                                if ( chit != chStatus->end() ) {
                                        chStatusCode = *chit;
                                } else {
                                        edm::LogError("EcalRecHitProducerError") << "No channel status found for xtal "
                                                << (*it).rawId()
                                                << "! something wrong with EcalChannelStatus in your DB? ";
                                }
                                EcalUncalibratedRecHit urh;
                                if ( (chStatusCode.getStatusCode() & 0x001F) == 12 ) { // dead VFE (from DB info)
                                        // uses the EcalUncalibratedRecHit to pass the DetId info
                                        urh = EcalUncalibratedRecHit( *it, 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EB_VFE );
                                        if ( recoverEBVFE_ || killDeadChannels_ ) workerRecover_->run( evt, urh, *ebRecHits );
                                } else {
                                        // uses the EcalUncalibratedRecHit to pass the DetId info
                                        urh = EcalUncalibratedRecHit( *it, 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EB_single );
                                        if ( recoverEBIsolatedChannels_ || killDeadChannels_ ) workerRecover_->run( evt, urh, *ebRecHits );
                                }
                                
                        }
                }
        }

        if ( recoverEEIsolatedChannels_ || recoverEEVFE_ || killDeadChannels_ )
        {
                edm::Handle< std::set<EEDetId> > pEEDetId;
                const std::set<EEDetId> * detIds = 0;
               
		evt.getByToken( eeDetIdToBeRecoveredToken_, pEEDetId);
		detIds = pEEDetId.product();
	        
                if ( detIds ) {
                        edm::ESHandle<EcalChannelStatus> chStatus;
                        es.get<EcalChannelStatusRcd>().get(chStatus);
                        for( std::set<EEDetId>::const_iterator it = detIds->begin(); it != detIds->end(); ++it ) {
                                // get channel status map to treat dead VFE separately
                                EcalChannelStatusMap::const_iterator chit = chStatus->find( *it );
                                EcalChannelStatusCode chStatusCode = 1;
                                if ( chit != chStatus->end() ) {
                                        chStatusCode = *chit;
                                } else {
                                        edm::LogError("EcalRecHitProducerError") << "No channel status found for xtal "
                                                << (*it).rawId()
                                                << "! something wrong with EcalChannelStatus in your DB? ";
                                }
                                EcalUncalibratedRecHit urh;
                                if ( (chStatusCode.getStatusCode() & 0x001F) == 12 ) { // dead VFE (from DB info)
                                        // uses the EcalUncalibratedRecHit to pass the DetId info
                                        urh = EcalUncalibratedRecHit( *it, 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EE_VFE );
                                        if ( recoverEEVFE_ || killDeadChannels_ ) workerRecover_->run( evt, urh, *eeRecHits );
                                } else {
                                        // uses the EcalUncalibratedRecHit to pass the DetId info
                                        urh = EcalUncalibratedRecHit( *it, 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EE_single );
                                        if ( recoverEEIsolatedChannels_ || killDeadChannels_ ) workerRecover_->run( evt, urh, *eeRecHits );
                                }
                        }
                }
        }

        if ( recoverEBFE_ || killDeadChannels_ )
        {
                edm::Handle< std::set<EcalTrigTowerDetId> > pEBFEId;
                const std::set<EcalTrigTowerDetId> * ttIds = 0;
               
		evt.getByToken( ebFEToBeRecoveredToken_, pEBFEId);
		ttIds = pEBFEId.product();
                
                if ( ttIds ) {
                        for( std::set<EcalTrigTowerDetId>::const_iterator it = ttIds->begin(); it != ttIds->end(); ++it ) {
                                // uses the EcalUncalibratedRecHit to pass the DetId info
                                int ieta = (((*it).ietaAbs()-1)*5+1)*(*it).zside(); // from EcalTrigTowerConstituentsMap
                                int iphi = ((*it).iphi()-1)*5+11;                   // from EcalTrigTowerConstituentsMap
                                if( iphi <= 0 ) iphi += 360;                        // from EcalTrigTowerConstituentsMap
                                EcalUncalibratedRecHit urh( EBDetId(ieta, iphi, EBDetId::ETAPHIMODE), 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EB_FE );
                                workerRecover_->run( evt, urh, *ebRecHits );
                        }
                }
        }

        if ( recoverEEFE_ || killDeadChannels_ )
        {
                edm::Handle< std::set<EcalScDetId> > pEEFEId;
                const std::set<EcalScDetId> * scIds = 0;
            
		evt.getByToken( eeFEToBeRecoveredToken_, pEEFEId);
		scIds = pEEFEId.product();
	
	        
                if ( scIds ) {
                        for( std::set<EcalScDetId>::const_iterator it = scIds->begin(); it != scIds->end(); ++it ) {
                                // uses the EcalUncalibratedRecHit to pass the DetId info
                                if (EEDetId::validDetId( ((*it).ix()-1)*5+1, ((*it).iy()-1)*5+1, (*it).zside() )) {
                                        EcalUncalibratedRecHit urh( EEDetId( ((*it).ix()-1)*5+1, ((*it).iy()-1)*5+1, (*it).zside() ), 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EE_FE );
                                        workerRecover_->run( evt, urh, *eeRecHits );
                                }
                        }
                }
        }

	// without re-sorting, find (used below in cleaning) will lead
        // to undefined results
	ebRecHits->sort();
        eeRecHits->sort();
	
	// apply spike cleaning
	if (cleaningAlgo_){
	  cleaningAlgo_->setFlags(*ebRecHits);
	  cleaningAlgo_->setFlags(*eeRecHits);
	}


        // put the collection of recunstructed hits in the event   
        LogInfo("EcalRecHitInfo") << "total # EB calibrated rechits: " << ebRecHits->size();
        LogInfo("EcalRecHitInfo") << "total # EE calibrated rechits: " << eeRecHits->size();

        evt.put( ebRecHits, ebRechitCollection_ );
        evt.put( eeRecHits, eeRechitCollection_ );
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( EcalRecHitProducer );
