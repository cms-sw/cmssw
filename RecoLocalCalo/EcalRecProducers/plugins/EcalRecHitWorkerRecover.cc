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
        tpDigiCollection_        = ps.getParameter<edm::InputTag>("triggerPrimitiveDigiCollection");
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

        if ( flags == EcalRecHitWorkerRecover::EB_single ) {
                // recover as single dead channel
                const EcalRecHitCollection * hit_collection = &result;
                EcalDeadChannelRecoveryAlgos deadChannelCorrector(caloTopology_.product());

                // channel recovery
                EcalRecHit hit( detId, 0., 0., EcalRecHit::kDead );
                if ( killDeadChannels_ ) {
                        // do nothing, the channel will be put in the collection
                        // with zero energy and the flag EcalRecHit::kDead
                } else {
                        // handle single dead channel
                        hit = deadChannelCorrector.correct( detId, hit_collection, singleRecoveryMethod_, singleRecoveryThreshold_ );
                        if ( hit.energy() != 0 ) {
                                hit.setFlags( EcalRecHit::kNeighboursRecovered );
                        }
                }
                EcalRecHitCollection::iterator it = result.find( detId );
                if ( it == result.end() ) {
                        result.push_back( hit );
                } else {
                        *it = hit;
                }
        } else if ( flags == EcalRecHitWorkerRecover::EB_VFE ) {
                // recover as dead VFE
        } else if ( flags == EcalRecHitWorkerRecover::EB_FE || flags == EcalRecHitWorkerRecover::EE_FE ) {
                // recover as dead TT
                EcalTrigTowerDetId ttDetId( detId.rawId() );
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
                if ( tp->id().subDet() == EcalBarrel ) {
                        // recover the whole trigger tower
                        if ( tp != tpDigis->end() ) {
                                //std::vector<DetId> vid = ecalMapping_->dccTowerConstituents( ecalMapping_->DCCid( ttDetId ), ecalMapping_->iTT( ttDetId ) );
                                std::vector<DetId> vid = ttMap_->constituentsOf( ttDetId );
                                // democratic energy sharing
                                for ( std::vector<DetId>::const_iterator dit = vid.begin(); dit != vid.end(); ++dit ) {
                                        float theta = 0;
                                        theta = ebGeom_->getGeometry(*dit)->getPosition().theta();
                                        float tpEt  = ecalScale_.getTPGInGeV( tp->compressedEt(), tp->id() );
                                        // set the recHit to 0 energy and kDead flag.
                                        // If recovery is false but kill dead channels is true
                                        EcalRecHit hit( detId, 0., 0., EcalRecHit::kDead );
                                        if ( !killDeadChannels_ ) {
                                                hit = EcalRecHit( *dit, tpEt / (float)vid.size() / sin(theta), 0., EcalRecHit::kTowerRecovered );
                                        }
                                        EcalRecHitCollection::iterator it = result.find( *dit );
                                        if ( it == result.end() ) {
                                                // insert the hit in the collection
                                                result.push_back( hit );
                                        } else {
                                                // overwrite existing recHit
                                                *it = hit;
                                        }
                                }
                        } else {
                                //error...
                        }
                } else if ( tp->id().subDet() == EcalEndcap ) {
                        // Structure for recovery:
                        // ** SC --> EEDetId constituents (eeC) --> associated Trigger Towers (aTT) --> EEDetId constituents (aTTC)
                        // ** energy for a SC EEDetId = [ sum_aTT(energy) - sum_aTTC(energy) ] / N_eeC
                        // .. i.e. the total energy of the TTs covering the SC minus 
                        // .. the energy of the recHits in the TTs but not in the SC
                        //std::vector<DetId> vid = ecalMapping_->dccTowerConstituents( ecalMapping_->DCCid( ttDetId ), ecalMapping_->iTT( ttDetId ) );
                        // --- RECOVERY NOT YET VALIDATED
                        EcalScDetId sc( detId );
                        std::vector<DetId> eeC;
                        for(int dx=1; dx<=5; ++dx){
                                for(int dy=1; dy<=5; ++dy){
                                        int ix = (sc.ix()-1)/5 + dx;
                                        int iy = (sc.iy()-1)/5 + dy;
                                        int iz = sc.zside();
                                        if(EEDetId::validDetId(ix, iy, iz)){
                                                eeC.push_back(EEDetId(ix, iy, iz));
                                        }
                                }
                        }
                        /*
                        // associated trigger towers
                        std::set<EcalTrigTowerDetId> aTT;
                        float totE = 0;
                        for ( size_t i = 0; i < eeC.size(); ++i ) {
                                float theta = eeGeom_->getGeometry( eeC[i] )->getPosition().theta();
                                totE += ecalScale_.getTPGInGeV( tp->compressedEt(), tp->id() ) / sin(theta);
                                aTT.insert( ttMap_->towerOf( eeC[i] ) );
                        }
                        // associated trigger towers: EEDetId constituents
                        std::set<DetId> aTTC;
                        for ( std::set<EcalTrigTowerDetId>::const_iterator it = aTT.begin(); it != aTT.end(); ++it ) {
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
                                        float theta = eeGeom_->getGeometry( *it )->getPosition().theta();
                                        totE -= (*jt).energy() / sin(theta);
                                }
                        }
                        */
                        // assign the energy to the SC crystals
                        for ( size_t i = 0; i < eeC.size(); ++i ) {
                                EcalRecHit hit( detId, 0., 0., EcalRecHit::kDead );
                                if ( !killDeadChannels_ ) {
                                        // not yet validated
                                        //hit = EcalRecHit( eeC[i], totE / (float)eeC.size(), 0., EcalRecHit::kTowerRecovered );
                                }
                                EcalRecHitCollection::iterator it = result.find( eeC[i] );
                                if ( it == result.end() ) {
                                        // insert the hit in the collection
                                        result.push_back( hit );
                                } else {
                                        // overwrite existing recHit
                                        *it = hit;
                                }
                        }
                }
        }
        return true;
}


#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalRecHitWorkerFactory, EcalRecHitWorkerRecover, "EcalRecHitWorkerRecover" );
