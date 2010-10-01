#include "RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitWorkerRecover.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryAlgos.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalRecHitWorkerRecover::EcalRecHitWorkerRecover(const edm::ParameterSet&ps) :
        EcalRecHitWorkerBaseClass(ps)
{
        rechitMaker_ = new EcalRecHitSimpleAlgo();
        // isolated channel recovery
        singleRecoveryMethod_    = ps.getParameter<std::string>("singleChannelRecoveryMethod");
        singleRecoveryThreshold_ = ps.getParameter<double>("singleChannelRecoveryThreshold");
        killDeadChannels_        = ps.getParameter<bool>("killDeadChannels");
        recoverEBIsolatedChannels_ = ps.getParameter<bool>("recoverEBIsolatedChannels");
        recoverEEIsolatedChannels_ = ps.getParameter<bool>("recoverEEIsolatedChannels");
        recoverEBVFE_ = ps.getParameter<bool>("recoverEBVFE");
        recoverEEVFE_ = ps.getParameter<bool>("recoverEEVFE");
        recoverEBFE_ = ps.getParameter<bool>("recoverEBFE");
        recoverEEFE_ = ps.getParameter<bool>("recoverEEFE");

        tpDigiCollection_        = ps.getParameter<edm::InputTag>("triggerPrimitiveDigiCollection");
        logWarningEtThreshold_EB_FE_ = ps.getParameter<double>("logWarningEtThreshold_EB_FE");
        logWarningEtThreshold_EE_FE_ = ps.getParameter<double>("logWarningEtThreshold_EE_FE");
}


void EcalRecHitWorkerRecover::set(const edm::EventSetup& es)
{
        //es.get<EcalIntercalibConstantsRcd>().get(ical);
        //es.get<EcalTimeCalibConstantsRcd>().get(itime);
        //es.get<EcalADCToGeVConstantRcd>().get(agc);
        //es.get<EcalChannelStatusRcd>().get(chStatus);
        es.get<EcalLaserDbRecord>().get(laser);
        es.get<CaloTopologyRecord>().get(caloTopology_);
        ecalScale_.setEventSetup( es );
        es.get<EcalMappingRcd>().get(pEcalMapping_);
        ecalMapping_ = pEcalMapping_.product();
        // geometry...
        es.get<EcalBarrelGeometryRecord>().get("EcalBarrel",pEBGeom_);
        es.get<EcalEndcapGeometryRecord>().get("EcalEndcap",pEEGeom_);
        ebGeom_ = pEBGeom_.product();
        eeGeom_ = pEEGeom_.product();
        es.get<IdealGeometryRecord>().get(ttMap_);
        recoveredDetIds_EB_.clear();
        recoveredDetIds_EE_.clear();
}


bool
EcalRecHitWorkerRecover::run( const edm::Event & evt,
                const EcalUncalibratedRecHit& uncalibRH,
                EcalRecHitCollection & result )
{
        DetId detId=uncalibRH.id();
        uint32_t flags = uncalibRH.recoFlag();

        // get laser coefficient
        //float lasercalib = laser->getLaserCorrection( detId, evt.time());

        // killDeadChannels_ = true, means explicitely kill dead channels even if the recovered energies are computed in the code
        // if you don't want to store the recovered energies in the rechit you can produce LogWarnings if logWarningEtThreshold_EB(EE)_FE>0 
	// logWarningEtThreshold_EB(EE)_FE_<0 will not compute the recovered energies at all (faster)
	// Revovery in the EE is not tested, recovered energies may not make sense
        // EE recovery computation is not tested against segmentation faults, use with caution even if you are going to killDeadChannels=true

        if ( killDeadChannels_ ) {
                if (    (flags == EcalRecHitWorkerRecover::EB_single && !recoverEBIsolatedChannels_)
                     || (flags == EcalRecHitWorkerRecover::EE_single && !recoverEEIsolatedChannels_)
                     || (flags == EcalRecHitWorkerRecover::EB_VFE && !recoverEBVFE_)
                     || (flags == EcalRecHitWorkerRecover::EE_VFE && !recoverEEVFE_)
                     ) {
                        EcalRecHit hit( detId, 0., 0., EcalRecHit::kDead );
                        hit.setFlagBits( (0x1 << EcalRecHit::kDead) ) ;
                        insertRecHit( hit, result); // insert trivial rechit with kDead flag
                        return true;
                } 
                if ( flags == EcalRecHitWorkerRecover::EB_FE && !recoverEBFE_) {
                        EcalTrigTowerDetId ttDetId( ((EBDetId)detId).tower() );
                        std::vector<DetId> vid = ttMap_->constituentsOf( ttDetId );
                        for ( std::vector<DetId>::const_iterator dit = vid.begin(); dit != vid.end(); ++dit ) {
                                EcalRecHit hit( (*dit), 0., 0., EcalRecHit::kDead );
                                hit.setFlagBits( (0x1 << EcalRecHit::kDead) ) ;
                                insertRecHit( hit, result ); // insert trivial rechit with kDead flag
                        }
			if(logWarningEtThreshold_EB_FE_<0)return true; // if you don't want log warning just return true
                }
                if ( flags == EcalRecHitWorkerRecover::EE_FE && !recoverEEFE_) {
                        EEDetId id( detId );
                        EcalScDetId sc( 1+(id.ix()-1)/5, 1+(id.iy()-1)/5, id.zside() );
                        std::vector<DetId> eeC;
                        for(int dx=1; dx<=5; ++dx){
                                for(int dy=1; dy<=5; ++dy){
                                        int ix = (sc.ix()-1)*5 + dx;
                                        int iy = (sc.iy()-1)*5 + dy;
                                        int iz = sc.zside();
                                        if(EEDetId::validDetId(ix, iy, iz)){
                                                eeC.push_back(EEDetId(ix, iy, iz));
                                        }
                                }
                        }
                        for ( size_t i = 0; i < eeC.size(); ++i ) {
                                EcalRecHit hit( eeC[i], 0., 0., EcalRecHit::kDead );
                                hit.setFlagBits( (0x1 << EcalRecHit::kDead) ) ;
                                insertRecHit( hit, result ); // insert trivial rechit with kDead flag
                        }
		   	if(logWarningEtThreshold_EE_FE_<0)   return true; // if you don't want log warning just return true
                }
        }

        if ( flags == EcalRecHitWorkerRecover::EB_single ) {
                // recover as single dead channel
                const EcalRecHitCollection * hit_collection = &result;
                EcalDeadChannelRecoveryAlgos deadChannelCorrector(caloTopology_.product());

                // channel recovery
                EcalRecHit hit = deadChannelCorrector.correct( detId, hit_collection, singleRecoveryMethod_, singleRecoveryThreshold_ );
                EcalRecHitCollection::const_iterator ti = result.find( detId );
                if ( hit.energy() != 0 ) {
                        hit.setFlags( EcalRecHit::kNeighboursRecovered );
                        hit.setFlagBits( (0x1 << EcalRecHit::kNeighboursRecovered) ) ;
                } else {
                        // recovery failed
                        hit.setFlags( EcalRecHit::kDead );
                        hit.setFlagBits( (0x1 << EcalRecHit::kDead) ) ;
                }
                insertRecHit( hit, result );
        } else if ( flags == EcalRecHitWorkerRecover::EB_VFE ) {
                // recover as dead VFE
                EcalRecHit hit( detId, 0., 0., EcalRecHit::kDead );
                hit.setFlagBits( (0x1 << EcalRecHit::kDead) ) ;
                // recovery not implemented
                insertRecHit( hit, result );
        } else if ( flags == EcalRecHitWorkerRecover::EB_FE ) {
                // recover as dead TT
                EcalTrigTowerDetId ttDetId( ((EBDetId)detId).tower() );
                edm::Handle<EcalTrigPrimDigiCollection> pTPDigis;
                evt.getByLabel(tpDigiCollection_, pTPDigis);
                const EcalTrigPrimDigiCollection * tpDigis = 0;
                if ( pTPDigis.isValid() ) {
                        tpDigis = pTPDigis.product();
                } else {
                        edm::LogError("EcalRecHitWorkerRecover") << "Can't get the product " << tpDigiCollection_.instance() 
                                << " with label " << tpDigiCollection_.label();
                        return false;
                }
                EcalTrigPrimDigiCollection::const_iterator tp = tpDigis->find( ttDetId );
                // recover the whole trigger tower
                if ( tp != tpDigis->end() ) {
                        //std::vector<DetId> vid = ecalMapping_->dccTowerConstituents( ecalMapping_->DCCid( ttDetId ), ecalMapping_->iTT( ttDetId ) );
                        std::vector<DetId> vid = ttMap_->constituentsOf( ttDetId );
                        float tpEt  = ecalScale_.getTPGInGeV( tp->compressedEt(), tp->id() );
                        float tpEtThreshEB = logWarningEtThreshold_EB_FE_;
                        if(tpEt>tpEtThreshEB){
                                edm::LogWarning("EnergyInDeadEB_FE")<<"TP energy in the dead TT = "<<tpEt<<" at "<<ttDetId;
                        }
                        if ( !killDeadChannels_ || recoverEBFE_ ) {  
                                // democratic energy sharing
                                for ( std::vector<DetId>::const_iterator dit = vid.begin(); dit != vid.end(); ++dit ) {
                                        float theta = 0;
                                        theta = ebGeom_->getGeometry(*dit)->getPosition().theta();
                                        float tpEt  = ecalScale_.getTPGInGeV( tp->compressedEt(), tp->id() );
                                        EcalRecHit hit( *dit, tpEt / (float)vid.size() / sin(theta), 0., EcalRecHit::kTowerRecovered );
                                        hit.setFlagBits( (0x1 << EcalRecHit::kTowerRecovered) ) ;
                                        if ( tp->compressedEt() == 0xFF ) hit.setFlagBits( (0x1 << EcalRecHit::kTPSaturated) );
                                        if ( tp->l1aSpike() ) hit.setFlagBits( (0x1 << EcalRecHit::kL1SpikeFlag) );
                                        insertRecHit( hit, result );
                                }
                        }
                } else {
                        // tp not found => recovery failed
                        std::vector<DetId> vid = ttMap_->constituentsOf( ttDetId );
                        for ( std::vector<DetId>::const_iterator dit = vid.begin(); dit != vid.end(); ++dit ) {
                                EcalRecHit hit( detId, 0., 0., EcalRecHit::kDead );
                                hit.setFlagBits( (0x1 << EcalRecHit::kDead) ) ;
                                EcalRecHitCollection::iterator it = result.find( *dit );
                                insertRecHit( hit, result );
                        }
                }
        } else if ( flags == EcalRecHitWorkerRecover::EE_FE ) {
                        // Structure for recovery:
                        // ** SC --> EEDetId constituents (eeC) --> associated Trigger Towers (aTT) --> EEDetId constituents (aTTC)
                        // ** energy for a SC EEDetId = [ sum_aTT(energy) - sum_aTTC(energy) ] / N_eeC
                        // .. i.e. the total energy of the TTs covering the SC minus 
                        // .. the energy of the recHits in the TTs but not in the SC
                        //std::vector<DetId> vid = ecalMapping_->dccTowerConstituents( ecalMapping_->DCCid( ttDetId ), ecalMapping_->iTT( ttDetId ) );
			// due to lack of implementation of the EcalTrigTowerDetId ix,iy methods in EE we compute Et recovered energies (in EB we compute E)
                        // --- RECOVERY NOT YET VALIDATED
                        EEDetId eeId( detId );
                        EcalScDetId sc( (eeId.ix()-1)/5+1, (eeId.iy()-1)/5+1, eeId.zside() );
                        std::vector<DetId> eeC;
                        for(int dx=1; dx<=5; ++dx){
                                for(int dy=1; dy<=5; ++dy){
                                        int ix = (sc.ix()-1)*5 + dx;
                                        int iy = (sc.iy()-1)*5 + dy;
                                        int iz = sc.zside();
                                        if(EEDetId::validDetId(ix, iy, iz)){
                                                eeC.push_back(EEDetId(ix, iy, iz));
                                        }
                                }
                        }
                        
                        edm::Handle<EcalTrigPrimDigiCollection> pTPDigis;
                        evt.getByLabel(tpDigiCollection_, pTPDigis);
                        const EcalTrigPrimDigiCollection * tpDigis = 0;
                        if ( pTPDigis.isValid() ) {
                                tpDigis = pTPDigis.product();
                        } else {
                                edm::LogError("EcalRecHitWorkerRecover") << "Can't get the product " << tpDigiCollection_.instance() 
                                        << " with label " << tpDigiCollection_.label();
                                return false;
                        }
                        // associated trigger towers
                        std::set<EcalTrigTowerDetId> aTT;
                        for ( size_t i = 0; i < eeC.size(); ++i ) {
                                aTT.insert( ttMap_->towerOf( eeC[i] ) );
                        }
                        // associated trigger towers: total energy
                        float totE = 0;
                        // associated trigger towers: EEDetId constituents
                        std::set<DetId> aTTC;
                        bool atLeastOneTPSaturated = false;
                        for ( std::set<EcalTrigTowerDetId>::const_iterator it = aTT.begin(); it != aTT.end(); ++it ) {
                                // add the energy of this trigger tower
                                EcalTrigPrimDigiCollection::const_iterator itTP = tpDigis->find( *it );
                                if ( itTP != tpDigis->end() ) {
                                        EcalTrigTowerDetId ttId = itTP->id();
        //                                EEDetId eeId( ttId.ieta(), ttId.iphi(), ttId.zside() );// this produces segmentation fault, should use ttId.ix(),iy() but these are not implmented
          //                              float theta = eeGeom_->getGeometry( eeId )->getPosition().theta();
                                        float theta = 1.0; // do not calculate theta, actually will use Et instead of E
                                        totE += ecalScale_.getTPGInGeV( itTP->compressedEt(), itTP->id() ) / sin(theta);
                                        if ( itTP->compressedEt() == 0xFF ) atLeastOneTPSaturated = true;
                                }
                                // get the trigger tower constituents
                                std::vector<DetId> v = ttMap_->constituentsOf( *it );
                                for ( size_t j = 0; j < v.size(); ++j ) {
                                        aTTC.insert( v[j] );
                                }
                        }
                        // remove crystals of dead SC
                        // (this step is not needed if sure that SC crystals are not 
                        // in the recHit collection)
                        for ( size_t i = 0; i < eeC.size(); ++i ) {
                                aTTC.erase( eeC[i] );
                        }
                        // compute the total energy for the dead SC
                        const EcalRecHitCollection * hits = &result;
                        for ( std::set<DetId>::const_iterator it = aTTC.begin(); it != aTTC.end(); ++it ) {
                                EcalRecHitCollection::const_iterator jt = hits->find( *it );
                                if ( jt != hits->end() ) {
                                        //float theta = eeGeom_->getGeometry( *it )->getPosition().theta();
                                        float theta = 1; // use Et instead of E, consistent with the Et estimation of the associated TT
                                        totE -= (*jt).energy() / sin(theta);
                                }
                        }
                        

			float scEt = totE;
			float scEtThreshEE = logWarningEtThreshold_EE_FE_;
			if(scEt>scEtThreshEE){
				edm::LogWarning("EnergyInDeadEE_FE")<<"TP energy in the dead TT = "<<scEt<<" at "<<sc;
			}

                        // assign the energy to the SC crystals
			if ( !killDeadChannels_ || recoverEEFE_ ) {
				for ( size_t i = 0; i < eeC.size(); ++i ) {
					EcalRecHit hit( eeC[i], 0., 0., EcalRecHit::kDead ); 
					hit.setFlagBits( (0x1 << EcalRecHit::kDead) ) ;
                                        
                                        hit = EcalRecHit( eeC[i], totE / (float)eeC.size(), 0., EcalRecHit::kTowerRecovered );
					insertRecHit( hit, result );
                                }
                        }
        }
        return true;
}


void EcalRecHitWorkerRecover::insertRecHit( const EcalRecHit &hit, EcalRecHitCollection &collection )
{
        // skip already inserted DetId's and raise a log warning
        if ( alreadyInserted( hit.id() ) ) {
	  //        edm::LogWarning("EcalRecHitWorkerRecover") << "DetId already recovered! Skipping...";
                return;
        }
        EcalRecHitCollection::iterator it = collection.find( hit.id() );
        if ( it == collection.end() ) {
                // insert the hit in the collection
                collection.push_back( hit );
        } else {
                // overwrite existing recHit
                *it = hit;
        }
        if ( hit.id().subdetId() == EcalBarrel ) {
                recoveredDetIds_EB_.insert( hit.id() );
        } else if ( hit.id().subdetId() == EcalEndcap ) {
                recoveredDetIds_EE_.insert( hit.id() );
        } else {
                edm::LogError("EcalRecHitWorkerRecover::InvalidDetId") << "Invalid DetId " << hit.id().rawId();
        }
}


bool EcalRecHitWorkerRecover::alreadyInserted( const DetId & id )
{
        bool res = false;
        if ( id.subdetId() == EcalBarrel ) {
                res = ( recoveredDetIds_EB_.find( id ) != recoveredDetIds_EB_.end() );
        } else if ( id.subdetId() == EcalEndcap ) {
                res = ( recoveredDetIds_EE_.find( id ) != recoveredDetIds_EE_.end() );
        } else {
                edm::LogError("EcalRecHitWorkerRecover::InvalidDetId") << "Invalid DetId " << id.rawId();
        }
        return res;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalRecHitWorkerFactory, EcalRecHitWorkerRecover, "EcalRecHitWorkerRecover" );
