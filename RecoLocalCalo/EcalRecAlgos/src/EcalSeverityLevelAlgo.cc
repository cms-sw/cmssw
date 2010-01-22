#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

int EcalSeverityLevelAlgo::severityLevel( const DetId id, 
                const EcalRecHitCollection &recHits, 
                const EcalChannelStatus &chStatus )
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
                 || rhFlag == EcalRecHit::kFake
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
                edm::LogError("EcalSeveritylevelError") << "No channel status found for xtal " 
                        << id.rawId() 
                        << "! something wrong with EcalChannelStatus in your DB? ";
        }
        return dbStatus;
}
