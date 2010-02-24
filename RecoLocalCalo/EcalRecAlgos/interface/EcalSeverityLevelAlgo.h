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

                enum EcalSeverityLevel { kGood, kProblematic, kRecovered, kWeird, kBad };

                enum SpikeId { kE1OverE9, kSwissCross };
                
                /** compute the severity level
                 */
                static int severityLevel( const DetId,
                                          const EcalRecHitCollection &,
                                          const EcalChannelStatus &,
                                          SpikeId spId = kE1OverE9,
                                          float threshold = 0.95
                                          );

                /** return the estimator of the signal being a spike
                 *  based on the topological information from the neighbours
                 */
                static float spikeFromNeighbours( const DetId id,
                                                  const EcalRecHitCollection &,
                                                  SpikeId spId = kE1OverE9
                                                  );

                /** ratio between the crystal energy and the energy in the 3x3
                 *  matrix of crystal
                 */
                static float E1OverE9( const DetId id, const EcalRecHitCollection & );
                static float E1OverE9New( const DetId id, const EcalRecHitCollection & );

                /** ratio between the crystal energy and the energy in the swiss cross
                 *  around
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

                // HOME MADE NAVIGATION - to be generalized
                // (and so moved in a more appropriate place)

                /** Converts a std CMSSW crystal eta index to a c-array index (starting from
                 * zero and without hole).
                 */
                static int iEta2cIndex(int iEta){
                        return (iEta<0)?iEta+85:iEta+84;
                }
                /** Converts a std CMSSW crystal phi index to a c-array index (starting from
                 * zero and without hole).
                 */
                static int iPhi2cIndex(int iPhi){
                        int iPhi0 = iPhi - 11;
                        if(iPhi0<0) iPhi0 += 360;
                        return iPhi0;
                }
                /** converse of iEta2cIndex() method.
                */
                static int cIndex2iEta(int i){
                        return (i<85)?i-85:i-84;
                }
                /** converse of iPhi2cIndex() method.
                */
                static int cIndex2iPhi(int i){
                        return 1+ ((i+10) % 360);
                }
                /** Converts a std CMSSW crystal x or y index to a c-array index (starting
                 * from zero and without hole).
                 */
                int iXY2cIndex(int iX) const{
                        return iX-1;
                }
                /** converse of iXY2cIndex() method.
                */
                int cIndex2iXY(int iX0) const{
                        return iX0+1;
                }
};

#endif
