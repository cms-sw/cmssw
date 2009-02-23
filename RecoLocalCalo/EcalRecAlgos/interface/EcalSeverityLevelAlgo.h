#ifndef RecoLocalCalo_EcalRecAlgos_EcalSeverityLevelAlgo_hh
#define RecoLocalCalo_EcalRecAlgos_EcalSeverityLevelAlgo_hh

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

class EcalSeverityLevelAlgo {
        public:
                // give the severity level from the EcalRecHit flags + the DB information stored in EcalChannelStatus
                // Levels of severity:
                // - 0 --> good
                // - 1 --> problematic (e.g. noisy)
                // - 2 --> recovered (e.g. dead or saturated)
                // - 3 --> bad, not suitable to be used in the reconstruction

                enum EcalSeverityLevel { kGood, kProblematic, kRecovered, kBad };
                
                int severityLevel( const DetId , const EcalRecHitCollection &, const EcalChannelStatus & ) const;
                int severityLevel( const EcalRecHit &, const EcalChannelStatus & ) const;
                int severityLevel( uint32_t rhFlag, uint16_t dbStatus ) const;
        private:
                uint16_t retrieveDBStatus( const DetId , const EcalChannelStatus &chStatus ) const;
};

#endif
