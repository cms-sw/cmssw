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

                enum EcalSeverityLevel { kGood, kProblematic, kRecovered, kBad, kWeird };
                
                static int severityLevel( const DetId , const EcalRecHitCollection &, const EcalChannelStatus & );
                static int severityLevel( uint32_t rhFlag, uint16_t dbStatus );
                static int severityLevel( const EcalRecHit &, const EcalChannelStatus & );

        private:
                static uint16_t retrieveDBStatus( const DetId , const EcalChannelStatus &chStatus );

                /** return the estimator of the signal being a spike
                 *  based on the neighbours energies
                 */
                static float spikeFromNeighbours( const DetId id , const EcalRecHitCollection & );
                /** return energy of a recHit (if in the collection)
                 */
                static float recHitEnergy( const DetId id, const EcalRecHitCollection &recHits );

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
