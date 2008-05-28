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
class CaloTopology;
class CaloGeometry;

class EcalClusterTools {
        public:
                EcalClusterTools() {};
                ~EcalClusterTools() {};

                // various energies in the matrix nxn surrounding the maximum energy crystal of the input cluster
                static float e1x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e3x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e1x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e5x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e2x5Right( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e2x5Left( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e2x5Top( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float e2x5Bottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                // energies in the crystal left, right, top, bottom w.r.t. to the most energetic crystal
                static float eLeft( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float eRight( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float eTop( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static float eBottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                // the energy of the most energetic crystal in the cluster
                static float eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );
                // the energy of the second most energetic crystal in the cluster
                static float e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );
                // get the DetId and the energy of the maximum energy crystal of the input cluster
                static std::pair<DetId, float> getMaximum( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits);

                static std::vector<float> energyBasketFractionEta( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );
                static std::vector<float> energyBasketFractionPhi( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );

                // return a vector v with v[0] = etaLat, v[1] = phiLat, v[2] = lat
                static std::vector<float> lat( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, bool logW = true, float w0 = 4.7 );
                // return a vector v with v[0] = covEtaEta, v[1] = covEtaPhi, v[2] = covPhiPhi
                static std::vector<float> covariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, const CaloGeometry* geometry, float w0 = 4.7);
                
                static double zernike20( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0 = 6.6, bool logW = true, float w0 = 4.7 );
                static double zernike42( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0 = 6.6, bool logW = true, float w0 = 4.7 );

                // get the detId's of a matrix centered in the maximum energy crystal = (0,0)
                // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
                static std::vector<DetId> matrixDetId( const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );
                // get the energy deposited in a matrix centered in the maximum energy crystal = (0,0)
                // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
                static float matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );
                // get the DetId and the energy of the maximum energy crystal in a vector of DetId
                static std::pair<DetId, float> getMaximum( const std::vector<DetId> &v_id, const EcalRecHitCollection *recHits);
                // get the energy of a DetId, return 0 if the DetId is not in the collection
                static float recHitEnergy(DetId id, const EcalRecHitCollection *recHits);

        private:
                struct EcalClusterEnergyDeposition
                { 
                        double deposited_energy;
                        double r;
                        double phi;
                };

                static std::vector<EcalClusterEnergyDeposition> getEnergyDepTopology( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, bool logW, float w0 );
                static math::XYZVector meanClusterPosition( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology *topology, const CaloGeometry *geometry );

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

                static double absZernikeMoment( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, int n, int m, double R0, bool logW, float w0 );
                static double fast_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, int n, int m, double R0, bool logW, float w0 );
                static double calc_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, int n, int m, double R0, bool logW, float w0 );

                static double factorial(int n) {
                        double res = 1.;
                        for (int i = 2; i <= n; ++i) res *= i;
                        return res;
                }

};

#endif
