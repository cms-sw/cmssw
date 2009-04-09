#include "RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitWorkerRecover.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
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
        tpDigiCollection_        = ps.getParameter<edm::InputTag>("triggerPrimitiveDigiCollection");
}


void EcalRecHitWorkerRecover::set(const edm::EventSetup& es)
{
        //es.get<EcalIntercalibConstantsRcd>().get(ical);
        //es.get<EcalTimeCalibConstantsRcd>().get(itime);
        //es.get<EcalADCToGeVConstantRcd>().get(agc);
        ////es.get<EcalChannelStatusRcd>().get(chStatus);
        es.get<EcalLaserDbRecord>().get(laser);
        es.get<CaloTopologyRecord>().get(caloTopology_);
        ecalScale_.setEventSetup( es );
        es.get<EcalMappingRcd>().get(pEcalMapping_);
        ecalMapping_ = pEcalMapping_.product();
        // geometry...
        es.get<EcalEndcapGeometryRecord>().get("EcalEndcap",pEBGeom_);
        es.get<EcalBarrelGeometryRecord>().get("EcalBarrel",pEEGeom_);
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
        float lasercalib = laser->getLaserCorrection( detId, evt.time());

        if ( flags == 1 ) {
                // recover as single dead channel
                const EcalRecHitCollection * hit_collection = &result;
                EcalDeadChannelRecoveryAlgos deadChannelCorrector(*caloTopology_);
                EcalRecHit hit = deadChannelCorrector.correct( detId, hit_collection, singleRecoveryMethod_, singleRecoveryThreshold_ );
                hit.setFlags( EcalRecHit::kNeighboursRecovered );
                result.push_back( hit );
        } else if ( flags == 2 ) {
                // recover as dead TT
                const EcalRecHitCollection * hits = &result;
                EcalRecHitCollection::const_iterator it = hits->find( detId );
                if ( it == hits->end() ) {
                        EcalTrigTowerDetId ttDetId( detId );
                        edm::Handle<EcalTrigPrimDigiCollection> pTPDigis;
                        evt.getByLabel(tpDigiCollection_, pTPDigis);
                        const EcalTrigPrimDigiCollection * tpDigis = pTPDigis.product();
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
                                                EcalRecHit hit( *dit, tpEt / (float)vid.size(), 0., EcalRecHit::kTowerRecovered );
                                                // paranoic: verify the hit is not in the collection
                                                if ( result.find( *dit ) == result.end() ) {
                                                        // insert the hit in the collection
                                                        result.push_back( hit );
                                                } else {
                                                        // error
                                                }
                                        }
                                } else {
                                        //error...
                                }
                        } else if ( tp->id().subDet() == EcalEndcap ) {
                                // FIXME -- too many incoherences with EE numbering schema -- implementation will follow
                                // Structure for recovery:
                                // ** SC --> EEDetId constituents (eeC) --> associated Trigger Towers (aTT) --> EEDetId constituents (aTTC)
                                // ** energy for a SC EEDetId = [ sum_aTT(energy) - sum_aTTC(energy) ] / N_eeC
                                // .. i.e. the total energy of the TTs covering the SC minus 
                                // .. the energy of the recHits in the TTs but not in the SC
                                //std::vector<DetId> vid = ecalMapping_->dccTowerConstituents( ecalMapping_->DCCid( ttDetId ), ecalMapping_->iTT( ttDetId ) );
                                //theta = eeGeom_->getGeometry(*dit)->getPosition().theta();
                        }
                } else {
                        // dead channel is in recHit collection ?!?
                }
        }
        return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalRecHitWorkerFactory, EcalRecHitWorkerRecover, "EcalRecHitWorkerRecover" );
