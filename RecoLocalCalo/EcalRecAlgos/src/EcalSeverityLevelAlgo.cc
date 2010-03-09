#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

int EcalSeverityLevelAlgo::severityLevel( const DetId id, 
                const EcalRecHitCollection & recHits, 
                const EcalChannelStatus & chStatus,
                SpikeId sp,
                float spIdThreshold
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
                if ( spikeFromNeighbours(id, recHits) > spIdThreshold  ) return kWeird;
                // .. not a spike, return the normal severity level
                return severityLevel( *it, chStatus );
        }
        return 0;
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
                                                  SpikeId spId
                                                  )
{
        switch( spId ) {
                case kE1OverE9:
                        return E1OverE9( id, recHits );
                        break;
                case kSwissCross:
                        return swissCross( id, recHits );
                        break;
                default:
                        edm::LogInfo("EcalSeverityLevelAlgo") << "Algorithm number " << spId
                                << " not known. Please check the enum in EcalSeverityLevelAlgo.h";
                        break;

        }
        return 0;
}

float EcalSeverityLevelAlgo::E1OverE9( const DetId id, const EcalRecHitCollection & recHits )
{
        // compute E1 over E9
        if ( id.subdetId() == EcalBarrel ) {
                EBDetId ebId( id );
                float s9 = 0;
                for ( int deta = -1; deta <= +1; ++deta ) {
                        for ( int dphi = -1; dphi <= +1; ++dphi ) {
                                s9 += recHitEnergy( id, recHits, deta, dphi );
                        }
                }
                return recHitEnergy(id, recHits) / s9;
        }
        return 0;
}

float EcalSeverityLevelAlgo::swissCross( const DetId id, const EcalRecHitCollection & recHits )
{
        // compute swissCross
        if ( id.subdetId() == EcalBarrel ) {
                EBDetId ebId( id );
                float s4 = 0;
                float e1 = recHitEnergy( id, recHits );
                float approxEta = 0.017453292519943295 * ebId.ieta();
                // avoid recHits at |eta|=85 where one side of the neighbours is missing
                // (may improve considering also eta module borders, but no
                // evidence for the time being that there the performance is
                // different)
                if ( abs(ebId.ieta())==85 ) return 0;
                // select recHits above 5 GeV 
                if ( e1 / cosh( approxEta ) < 5 ) return 0;
                s4 += recHitEnergy( id, recHits,  1,  0 );
                s4 += recHitEnergy( id, recHits, -1,  0 );
                s4 += recHitEnergy( id, recHits,  0,  1 );
                s4 += recHitEnergy( id, recHits,  0, -1 );
                return 1 - s4 / e1;
        }
        return 0;
}

float EcalSeverityLevelAlgo::recHitEnergy( const DetId id, const EcalRecHitCollection & recHits,
                                           int dEta, int dPhi )
{
        DetId nid = EBDetId::offsetBy( id, dEta, dPhi );
        return ( nid == DetId(0) ? 0 : recHitEnergy( nid, recHits ) );
}

float EcalSeverityLevelAlgo::recHitEnergy( const DetId id, const EcalRecHitCollection &recHits )
{
        if ( id == DetId(0) ) {
                return 0;
        } else {
                EcalRecHitCollection::const_iterator it = recHits.find( id );
                if ( it != recHits.end() ) return (*it).energy();
        }
        return 0;
}
