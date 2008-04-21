#ifndef RecoEcal_EgammaCoreTools_EcalClusterTools_h
#define RecoEcal_EgammaCoreTools_EcalClusterTools_h

/** \class EcalClusterTools
 *  
 * various cluster tools (e.g. cluster shapes)
 *
 * \author Federico Ferri
 * 
 * \version $Id: 
 *
 */

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

class DetId;
class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;

class EcalClusterTools {
        public:
                EcalClusterTools() {};
                ~EcalClusterTools() {};

                static float e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                static float e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                static float e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                static float e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                static float e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                static float e2x5Right( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                static float e2x5Left( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                static float e2x5Top( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                static float e2x5Bottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology );
                static float eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );
                static float e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );

                static std::vector<float> energyBasketFractionEta( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );
                static std::vector<float> energyBasketFractionPhi( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );

                std::pair<float, float> etaPhiLat( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, float w0 = 0.1, bool logW = false );
                std::vector<float> covariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloSubdetectorTopology *topology, const CaloSubdetectorGeometry* geometry, float w0 = 0.1);
                
                static double zernike20( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, float w0, bool logW );
                static double zernike42( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, float w0, bool logW );

        private:
                struct EcalClusterEnergyDeposition
                { 
                        double deposited_energy;
                        double r;
                        double phi;
                };


                static std::pair<DetId, float> getMaximum( const std::vector<DetId> &v_id, const EcalRecHitCollection *recHits);
                static float recHitEnergy(DetId id, const EcalRecHitCollection *recHits);
                static float matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );

                static std::vector<EcalClusterEnergyDeposition> getEnergyDepTopology( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, float w0 = 0.1, bool logW = true  );
                static math::XYZVector meanClusterPosition( std::vector<DetId> v_id, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology *topology, const CaloSubdetectorGeometry *geometry );

                static double f00(double r) { return 1; }
                static double f11(double r) { return r; }
                static double f20(double r) { return 2.0*r*r-1.0; }
                static double f22(double r) { return r*r; }
                static double f31(double r) { return 3.0*r*r*r - 2.0*r; }
                static double f33(double r) { return r*r*r; }
                static double f40(double r) { return 6.0*r*r*r*r-6.0*r*r+1.0; }
                static double f42(double r) { return 4.0*r*r*r*r-3.0*r*r; }
                static double f44(double r) { return r*r*r*r; }
                static double f51(double r) { return 10.0*pow(r,5)-12.0*pow(r,3)+3.0*r; }
                static double f53(double r) { return 5.0*pow(r,5) - 4.0*pow(r,3); }
                static double f55(double r) { return pow(r,5); }

                static double absZernikeMoment( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, float w0, bool logW, int n, int m, double R0 = 6.6 );
                static double fast_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, float w0, bool logW,int n, int m, double R0 );
                static double calc_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, float w0, bool logW, int n, int m, double R0 );

                static double factorial(int n) {
                        double res = 1.;
                        for (int i = 2; i <= n; ++i) res *= i;
                        return res;
                }

};

#endif
