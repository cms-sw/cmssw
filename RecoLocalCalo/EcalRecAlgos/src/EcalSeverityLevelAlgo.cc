#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

int EcalSeverityLevelAlgo::severityLevel( const DetId id, 
                const EcalRecHitCollection &recHits, 
                const EcalChannelStatus &chStatus ) const
{
        // get DB flag
        uint16_t dbStatus = retrieveDBStatus( id, chStatus );
        // get recHit flags
        EcalRecHitCollection::const_iterator it = recHits.find( id );
        if ( it == recHits.end() ) {
                // the channel is not in the recHit collection:
                // dead or zero-suppressed?
                // at the moment the DB is binary: 0 = good, 1 = bad
                // FIXME: change 1 to the dead channel flag, when it will be defined
                if ( dbStatus == 1 ) {
                        return kBad;
                } else if ( dbStatus > 0 && dbStatus != 1 ) { // FIXME: at the moment useless because of binary DB
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
                const EcalChannelStatus &chStatus ) const
{
        // the channel is there, check its flags
        // and combine with DB (not needed at the moment)
        uint32_t rhFlag = recHit.recoFlag();
        uint16_t dbStatus = retrieveDBStatus( recHit.id(), chStatus );
        return severityLevel( rhFlag, dbStatus );
}

int EcalSeverityLevelAlgo::severityLevel( uint32_t rhFlag, uint16_t chStatus ) const
{
        // DB info currently not used at this level
        if ( rhFlag > 0 && rhFlag <= 4 ) {
                // problematic
                return kProblematic;
        } else if ( rhFlag > 4 && rhFlag <= 6 ) {
                // recovered
                return kRecovered;
        } else if ( rhFlag == 7 ) {
                // recovering failed
                return kBad;
        }
        // good
        return 0;
}

uint16_t EcalSeverityLevelAlgo::retrieveDBStatus( const DetId id, const EcalChannelStatus &chStatus ) const
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
