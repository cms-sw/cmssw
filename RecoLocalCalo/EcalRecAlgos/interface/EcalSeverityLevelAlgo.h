#ifndef RecoLocalCalo_EcalRecAlgos_EcalSeverityLevelAlgo_hh
#define RecoLocalCalo_EcalRecAlgos_EcalSeverityLevelAlgo_hh

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

class EcalSeverityLevelAlgo {
        public:
                // give the severity level from the EcalRecHit flags + the DB information stored in EcalChannelStatus
                // Levels of severity:
                // - kGood        --> good channel
                // - kProblematic --> problematic (e.g. noisy)
                // - kRecovered   --> recovered (e.g. an originally dead or saturated)
                // - kTime        --> the channel is out of time (e.g. spike)
                // - kWeird       --> weird (e.g. spike)
                // - kBad         --> bad, not suitable to be used in the reconstruction

                enum EcalSeverityLevel { kGood=0, kProblematic, kRecovered, kTime, kWeird, kBad };

                enum SpikeId { kE1OverE9=0, kSwissCross, kSwissCrossBordersIncluded };

                /** compute the severity level
                 */
                static int severityLevel( const DetId,
                                          const EcalRecHitCollection &,
                                          const EcalChannelStatus &,
                                          float recHitEtThreshold = 5.,
                                          SpikeId spId = kSwissCross,
                                          float spIdThreshold = 0.95,
					  float recHitEnergyThresholdForTiming = 2.,
					  float recHitEnergyThresholdForEE = 1e+37
                                          );

                /** return the estimator of the signal being a spike
                 *  based on the topological information from the neighbours
                 */
                static float spikeFromNeighbours( const DetId id,
                                                  const EcalRecHitCollection &,
                                                  float recHitEtThreshold,
                                                  SpikeId spId
                                                  );

                /** ratio between the crystal energy and the energy in the 3x3
                 *  matrix of crystal
                 */
                static float E1OverE9( const DetId id, const EcalRecHitCollection &, float recHitEtThreshold = 0. );

                /** 1 - the ratio between the energy in the swiss cross around
                 * a crystal and the crystal energy (also called S4/S1, Rook)
                 */
                static float swissCross( const DetId id, const EcalRecHitCollection &, float recHitEtThreshold = 0. , bool avoidIeta85=true);

		/** return whether or not the rechit is a spike based on the kOutOfTime rechit flag
                 */
		static bool spikeFromTiming( const EcalRecHit &, float recHitEnergyThreshold );

        private:

                static int severityLevel( uint32_t rhFlag, uint16_t dbStatus );
                static int severityLevel( const EcalRecHit &, const EcalChannelStatus & );

                static uint16_t retrieveDBStatus( const DetId , const EcalChannelStatus &chStatus );

                /** return energy of a recHit (if in the collection)
                 */
                static float recHitE( const DetId id, const EcalRecHitCollection &recHits );
                static float recHitApproxEt( const DetId id, const EcalRecHitCollection &recHits );
                static float recHitE( const DetId id, const EcalRecHitCollection & recHits, int dEta, int dPhi );
};

#endif
