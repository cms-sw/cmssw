#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

int EcalSeverityLevelAlgo::severityLevel( const DetId id, 
                const EcalRecHitCollection & recHits, 
                const EcalChannelStatus & chStatus,
                float recHitEtThreshold,
                SpikeId spId,
		float spIdThreshold,
		float recHitEnergyThresholdForTiming,
		float recHitEnergyThresholdForEE
                )
{

        // get DB flag
        uint16_t dbStatus = retrieveDBStatus( id, chStatus );
        // get recHit flags
        EcalRecHitCollection::const_iterator it = recHits.find( id );
        if ( it == recHits.end() ) {
                // the channel is not in the recHit collection:
                // dead or zero-suppressed?
                if ( dbStatus >= 10 ) { // originally dead
                        return kBad;
                } else if ( dbStatus > 0 && dbStatus < 10 ) {
                        // zero-suppressed and originally problematic
                        return kProblematic;
                } else if ( dbStatus == 0 ) {
                        // zero-suppressed and originally good
                        return kGood;
                }
        } else {
                // the channel is in the recHit collection
                // .. is it a spike?
                // check the topology
                if ( id.subdetId() == EcalBarrel && (spikeFromNeighbours(id, recHits, recHitEtThreshold, spId) > spIdThreshold)  ) return kWeird;
                // check the timing (currently only a trivial check)
		if ( id.subdetId() == EcalBarrel && spikeFromTiming(*it, recHitEnergyThresholdForTiming) ) return kTime;
                // filtering on VPT discharges in the endcap
                if ( id.subdetId() == EcalEndcap && spId == kSwissCross && ( 1-swissCross(id, recHits, recHitEnergyThresholdForEE, spId) < 0.02*log(recHitE(id, recHits)/4.) )  ) return kWeird;

                // .. not a spike, return the normal severity level
                return severityLevel( *it, chStatus );
        }
        return kGood;
}

int EcalSeverityLevelAlgo::severityLevel( const EcalRecHit &recHit, 
                const EcalChannelStatus &chStatus )
{
        // the channel is there, check its flags
        // and combine with DB (not needed at the moment)
        uint32_t rhFlag = recHit.recoFlag();
        uint16_t dbStatus = retrieveDBStatus( recHit.id(), chStatus );
        return severityLevel( rhFlag, dbStatus );
}

int EcalSeverityLevelAlgo::severityLevel( uint32_t rhFlag, uint16_t chStatus )
{
        // DB info currently not used at this level
        if       (  rhFlag == EcalRecHit::kPoorReco 
                 || rhFlag == EcalRecHit::kOutOfTime
                 || rhFlag == EcalRecHit::kNoisy
                 || rhFlag == EcalRecHit::kPoorCalib 
                 || rhFlag == EcalRecHit::kFaultyHardware
                 ) {
                // problematic
                return kProblematic;
        } else if ( rhFlag == EcalRecHit::kLeadingEdgeRecovered
                 || rhFlag == EcalRecHit::kNeighboursRecovered
                 || rhFlag == EcalRecHit::kTowerRecovered
                 ) {
                // recovered
                return kRecovered;
        } else if ( rhFlag == EcalRecHit::kDead
                 || rhFlag == EcalRecHit::kSaturated
                 //|| rhFlag == EcalRecHit::kFake // will be uncommented when validated
                 || rhFlag == EcalRecHit::kFakeNeighbours
                 || rhFlag == EcalRecHit::kKilled ) {
                // recovery failed (or not tried) or signal is fake or channel
                // is dead
                return kBad;
        }
        // good
        return kGood;
}

uint16_t EcalSeverityLevelAlgo::retrieveDBStatus( const DetId id, const EcalChannelStatus &chStatus )
{
        EcalChannelStatus::const_iterator chIt = chStatus.find( id );
        uint16_t dbStatus = 0;
        if ( chIt != chStatus.end() ) {
                dbStatus = chIt->getStatusCode();
        } else {
                edm::LogError("EcalSeverityLevelError") << "No channel status found for xtal " 
                        << id.rawId() 
                        << "! something wrong with EcalChannelStatus in your DB? ";
        }
        return dbStatus;
}

float EcalSeverityLevelAlgo::spikeFromNeighbours( const DetId id,
                                                  const EcalRecHitCollection & recHits,
                                                  float recHitThreshold,
                                                  SpikeId spId
                                                  )
{
  switch( spId ) {
  case kE1OverE9:
    return E1OverE9( id, recHits, recHitThreshold );
    break;
  case kSwissCross:
    return swissCross( id, recHits, recHitThreshold , true);
    break;
  case kSwissCrossBordersIncluded:
    return swissCross( id, recHits, recHitThreshold , false);
    break;
  default:
    edm::LogInfo("EcalSeverityLevelAlgo") << "Algorithm number " << spId
					  << " not known. Please check the enum in EcalSeverityLevelAlgo.h";
    break;
    
  }
        return 0;
}

float EcalSeverityLevelAlgo::E1OverE9( const DetId id, const EcalRecHitCollection & recHits, float recHitEtThreshold )
{
        // compute E1 over E9
        if ( id.subdetId() == EcalBarrel ) {
                // select recHits with Et above recHitEtThreshold
                if ( recHitApproxEt( id, recHits ) < recHitEtThreshold ) return 0;
                EBDetId ebId( id );
                float s9 = 0;
                for ( int deta = -1; deta <= +1; ++deta ) {
                        for ( int dphi = -1; dphi <= +1; ++dphi ) {
                                s9 += recHitE( id, recHits, deta, dphi );
                        }
                }
                return recHitE(id, recHits) / s9;
        } else if( id.subdetId() == EcalEndcap ) {
                // select recHits with Et above recHitEtThreshold
                if ( recHitApproxEt( id, recHits ) < recHitEtThreshold ) return 0;
                EEDetId eeId( id );
                float s9 = 0;
                for ( int dx = -1; dx <= +1; ++dx ) {
                        for ( int dy = -1; dy <= +1; ++dy ) {
                                s9 += recHitE( id, recHits, dx, dy );
                        }
                }
                return recHitE(id, recHits) / s9;

        }
        return 0;
}

float EcalSeverityLevelAlgo::swissCross( const DetId id, const EcalRecHitCollection & recHits, float recHitThreshold , bool avoidIeta85)
{
        // compute swissCross
        if ( id.subdetId() == EcalBarrel ) {
                EBDetId ebId( id );
                // avoid recHits at |eta|=85 where one side of the neighbours is missing
                // (may improve considering also eta module borders, but no
                // evidence for the time being that there the performance is
                // different)
                if ( abs(ebId.ieta())==85 && avoidIeta85) return 0;
                // select recHits with Et above recHitThreshold
                if ( recHitApproxEt( id, recHits ) < recHitThreshold ) return 0;
                float s4 = 0;
                float e1 = recHitE( id, recHits );
                // protect against nan (if 0 threshold is given above)
                if ( e1 == 0 ) return 0;
                s4 += recHitE( id, recHits,  1,  0 );
                s4 += recHitE( id, recHits, -1,  0 );
                s4 += recHitE( id, recHits,  0,  1 );
                s4 += recHitE( id, recHits,  0, -1 );
                return 1 - s4 / e1;
        } else if ( id.subdetId() == EcalEndcap ) {
                EEDetId eeId( id );
                // select recHits with E above recHitThreshold
                float e1 = recHitE( id, recHits );
                if ( e1 < recHitThreshold ) return 0;
                float s4 = 0;
                // protect against nan (if 0 threshold is given above)
                if ( e1 == 0 ) return 0;
                s4 += recHitE( id, recHits,  1,  0 );
                s4 += recHitE( id, recHits, -1,  0 );
                s4 += recHitE( id, recHits,  0,  1 );
                s4 += recHitE( id, recHits,  0, -1 );
                return 1 - s4 / e1;
        }
        return 0;
}

float EcalSeverityLevelAlgo::recHitE( const DetId id, const EcalRecHitCollection & recHits,
                                           int di, int dj )
{
        // in the barrel:   di = dEta   dj = dPhi
        // in the endcap:   di = dX     dj = dY
  
        DetId nid;
        if( id.subdetId() == EcalBarrel) nid = EBDetId::offsetBy( id, di, dj );
        else if( id.subdetId() == EcalEndcap) nid = EEDetId::offsetBy( id, di, dj );

        return ( nid == DetId(0) ? 0 : recHitE( nid, recHits ) );
}

float EcalSeverityLevelAlgo::recHitE( const DetId id, const EcalRecHitCollection &recHits )
{
        if ( id == DetId(0) ) {
                return 0;
        } else {
                EcalRecHitCollection::const_iterator it = recHits.find( id );
                if ( it != recHits.end() ) return (*it).energy();
        }
        return 0;
}


float EcalSeverityLevelAlgo::recHitApproxEt( const DetId id, const EcalRecHitCollection &recHits )
{
        // for the time being works only for the barrel
        if ( id.subdetId() == EcalBarrel ) {
                return recHitE( id, recHits ) / cosh( EBDetId::approxEta( id ) );
        }
        return 0;
}


bool EcalSeverityLevelAlgo::spikeFromTiming( const EcalRecHit &recHit, float recHitEnergyThreshold)
{
        if ( recHit.energy() < recHitEnergyThreshold )     return false;
        if ( recHit.recoFlag() == EcalRecHit::kOutOfTime ) return true;
        return false;
}
