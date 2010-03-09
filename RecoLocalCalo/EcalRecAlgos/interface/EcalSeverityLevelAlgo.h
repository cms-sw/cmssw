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
                // - 2 --> recovered (e.g. an originally dead or saturated)
                // - 3 --> weird (e.g. spike)
                // - 4 --> bad, not suitable to be used in the reconstruction

                enum EcalSeverityLevel { kGood=0, kProblematic, kRecovered, kWeird, kBad };

                enum SpikeId { kE1OverE9=0, kSwissCross };

                /** compute the severity level
                 */
                static int severityLevel( const DetId,
                                          const EcalRecHitCollection &,
                                          const EcalChannelStatus &,
                                          SpikeId spId = kSwissCross,
                                          float spIdThreshold = 0.95,
                                          float recHitEtThreshold = 5.
                                          );

                /** return the estimator of the signal being a spike
                 *  based on the topological information from the neighbours
                 */
                static float spikeFromNeighbours( const DetId id,
                                                  const EcalRecHitCollection &,
                                                  SpikeId spId = kSwissCross
                                                  );

                /** ratio between the crystal energy and the energy in the 3x3
                 *  matrix of crystal
                 */
                static float E1OverE9( const DetId id, const EcalRecHitCollection & );

                /** 1 - the ratio between the energy in the swiss cross around
                 * a crystal and the crystal energy (also called S4/S1, Rook)
                 */
                static float swissCross( const DetId id, const EcalRecHitCollection & );

        private:

                static int severityLevel( uint32_t rhFlag, uint16_t dbStatus );
                static int severityLevel( const EcalRecHit &, const EcalChannelStatus & );

                static uint16_t retrieveDBStatus( const DetId , const EcalChannelStatus &chStatus );

                /** return energy of a recHit (if in the collection)
                 */
                static float recHitEnergy( const DetId id, const EcalRecHitCollection &recHits );
                static float recHitEnergy( const DetId id, const EcalRecHitCollection & recHits, int dEta, int dPhi );
};

#endif
