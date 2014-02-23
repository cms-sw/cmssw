#ifndef RecoEcal_EgammaCoreTools_EcalClusterTools_h
#define RecoEcal_EgammaCoreTools_EcalClusterTools_h

/** \class EcalClusterTools
 *  
 * various cluster tools (e.g. cluster shapes)
 *
 * \author Federico Ferri
 *
 * editing author: M.B. Anderson
 * 
 *
 */

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
//#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
//includes for ShowerShape function to work
#include <vector>
#include <math.h>
#include <TMath.h>
#include <TMatrixT.h>
#include <TMatrixD.h>
#include <TVectorT.h>
#include <TVectorD.h>


class DetId;
class CaloTopology;
class CaloGeometry;

struct Cluster2ndMoments {

  // major and minor cluster moments wrt principale axes:
  float sMaj;
  float sMin;
  // angle between sMaj and phi:
  float alpha;

};

class EcalClusterTools {
        public:
                EcalClusterTools() {};
                ~EcalClusterTools() {};

                // various energies in the matrix nxn surrounding the maximum energy crystal of the input cluster
		//we use an eta/phi coordinate system rather than phi/eta
                //note e3x2 does not have a definate eta/phi geometry, it takes the maximum 3x2 block containing the 
                //seed regardless of whether that 3 in eta or phi
                static float e1x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );

static float e1x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

                static float e3x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );

static float e3x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

                static float e1x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
		//AA
		//Take into account severities
                static float e1x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

                static float e5x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e5x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

                static float e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );

                static float e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

                static float e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );

                static float e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology);
static float e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

                static float e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

                // energy in the 2x5 strip right of the max crystal (does not contain max crystal)
		// 2 crystals wide in eta, 5 wide in phi.
                static float e2x5Right( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e2x5Right( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );
                // energy in the 2x5 strip left of the max crystal (does not contain max crystal)

                static float e2x5Left( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e2x5Left( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );
                // energy in the 5x2 strip above the max crystal (does not contain max crystal)
		// 5 crystals wide in eta, 2 wide in phi.

                static float e2x5Top( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e2x5Top( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );
                // energy in the 5x2 strip below the max crystal (does not contain max crystal)                

                static float e2x5Bottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e2x5Bottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );
                // energy in a 2x5 strip containing the seed (max) crystal.
                // 2 crystals wide in eta, 5 wide in phi.
                // it is the maximum of either (1x5left + 1x5center) or (1x5right + 1x5center)
		static float e2x5Max( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float e2x5Max( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );

                // energies in the crystal left, right, top, bottom w.r.t. to the most energetic crystal
                static float eLeft( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float eLeft( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );

                static float eRight( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float eRight( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );

                static float eTop( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float eTop( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );

                static float eBottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
static float eBottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );
                // the energy of the most energetic crystal in the cluster

                static float eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );
static float eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );

                // the energy of the second most energetic crystal in the cluster
                static float e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );
static float e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );

                // get the DetId and the energy of the maximum energy crystal of the input cluster
                static std::pair<DetId, float> getMaximum( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits);
static std::pair<DetId, float> getMaximum( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

                static std::vector<float> energyBasketFractionEta( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits );
static std::vector<float> energyBasketFractionEta( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );

                static std::vector<float> energyBasketFractionPhi( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits);
static std::vector<float> energyBasketFractionPhi( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

                // return a vector v with v[0] = etaLat, v[1] = phiLat, v[2] = lat
                static std::vector<float> lat( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, bool logW = true, float w0 = 4.7 );
static std::vector<float> lat( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv, bool logW = true,float w0 = 4.7);

                // return a vector v with v[0] = covEtaEta, v[1] = covEtaPhi, v[2] = covPhiPhi

                static std::vector<float> covariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, const CaloGeometry* geometry, float w0 = 4.7);
static std::vector<float> covariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, const CaloGeometry* geometry,const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv,float w0 = 4.7);
                // return a vector v with v[0] = covIEtaIEta, v[1] = covIEtaIPhi, v[2] = covIPhiIPhi
                //this function calculates differences in eta/phi in units of crystals not global eta/phi
                //this is gives better performance in the crack regions of the calorimeter but gives otherwise identical results to covariances function
                //   except that it doesnt need an eta based correction funtion in the endcap 
                //it is multipled by an approprate crystal size to ensure it gives similar values to covariances(...)
                //
                //Warning: covIEtaIEta has been studied by egamma, but so far covIPhiIPhi hasnt been studied extensively so there could be a bug in 
                //         the covIPhiIEta or covIPhiIPhi calculations. I dont think there is but as it hasnt been heavily used, there might be one
                static std::vector<float> localCovariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, float w0 = 4.7);
static std::vector<float> localCovariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv,float w0 = 4.7);
                
                static std::vector<float> scLocalCovariances(const reco::SuperCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology, float w0 = 4.7);
static std::vector<float> scLocalCovariances(const reco::SuperCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv, float w0 = 4.7);

                // cluster second moments with respect to principal axes:
                static Cluster2ndMoments cluster2ndMoments( const reco::BasicCluster &basicCluster, const EcalRecHitCollection &recHits, double phiCorrectionFactor=0.8, double w0=4.7, bool useLogWeights=true);

                static Cluster2ndMoments cluster2ndMoments( const reco::SuperCluster &superCluster, const EcalRecHitCollection &recHits, double phiCorrectionFactor=0.8, double w0=4.7, bool useLogWeights=true);
                static Cluster2ndMoments cluster2ndMoments( const std::vector<std::pair<const EcalRecHit*, float> >& RH_ptrs_fracs, double  phiCorrectionFactor=0.8, double  w0=4.7, bool useLogWeights=true);

                static double zernike20( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0 = 6.6, bool logW = true, float w0 = 4.7 );
                static double zernike42( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0 = 6.6, bool logW = true, float w0 = 4.7 );

                // get the detId's of a matrix centered in the maximum energy crystal = (0,0)
                // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
                static std::vector<DetId> matrixDetId( const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );
static std::vector<DetId> matrixDetId( const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv  );
                // get the energy deposited in a matrix centered in the maximum energy crystal = (0,0)
                // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
                static float matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );

		//AA
		//Take into account the severities and flags
                static float matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );
		static float getFraction( const std::vector< std::pair<DetId, float> > &v_id, DetId id);
                // get the DetId and the energy of the maximum energy crystal in a vector of DetId
                static std::pair<DetId, float> getMaximum( const std::vector< std::pair<DetId, float> > &v_id, const EcalRecHitCollection *recHits);
static std::pair<DetId, float> getMaximum( const std::vector< std::pair<DetId, float> > &v_id, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);
                // get the energy of a DetId, return 0 if the DetId is not in the collection
                static float recHitEnergy(DetId id, const EcalRecHitCollection *recHits);

		//AA
		//Take into account severities and flags
                static float recHitEnergy(DetId id, const EcalRecHitCollection *recHits, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

		//Shower shape variables return vector <Roundness, Angle> of a photon
		static std::vector<float> roundnessBarrelSuperClusters( const reco::SuperCluster &superCluster ,const EcalRecHitCollection &recHits, int weightedPositionMethod = 0, float energyThreshold = 0.0);
		static std::vector<float> roundnessBarrelSuperClustersUserExtended( const reco::SuperCluster &superCluster ,const EcalRecHitCollection &recHits, int ieta_delta=0, int iphi_delta=0, float energyRHThresh=0.00000, int weightedPositionMethod=0);
		static std::vector<float> roundnessSelectedBarrelRecHits(const std::vector<std::pair<const EcalRecHit*,float> >&rhVector, int weightedPositionMethod = 0);
        private:
                struct EcalClusterEnergyDeposition
                { 
                        double deposited_energy;
                        double r;
                        double phi;
                };

                static std::vector<EcalClusterEnergyDeposition> getEnergyDepTopology( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, bool logW, float w0 );

                static math::XYZVector meanClusterPosition( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology *topology, const CaloGeometry *geometry );
static math::XYZVector meanClusterPosition( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology *topology, const CaloGeometry *geometry, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv );
                //return energy weighted mean distance from the seed crystal in number of crystals
                //<iEta,iPhi>, iPhi is not defined for endcap and is returned as zero 
                static std::pair<float,float>  mean5x5PositionInLocalCrysCoord(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology);
static std::pair<float,float>  mean5x5PositionInLocalCrysCoord(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

		static std::pair<float,float> mean5x5PositionInXY(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology);
static std::pair<float,float> mean5x5PositionInXY(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology, const std::vector<int>& flagsexcl, const std::vector<int>& severitiesexcl, const EcalSeverityLevelAlgo *sevLv);

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

                //useful functions for the localCovariances function
                static float getIEta(const DetId& id);
                static float getIPhi(const DetId& id);
		static float getNormedIX(const DetId& id);
                static float getNormedIY(const DetId& id);
                static float getDPhiEndcap(const DetId& crysId,float meanX,float meanY);
                static float getNrCrysDiffInEta(const DetId& crysId,const DetId& orginId);
                static float getNrCrysDiffInPhi(const DetId& crysId,const DetId& orginId);
		
		//useful functions for showerRoundnessBarrel function
		static int deltaIEta(int seed_ieta, int rh_ieta);
		static int deltaIPhi(int seed_iphi, int rh_iphi);
		static std::vector<int> getSeedPosition(const std::vector<std::pair<const EcalRecHit*,float> >&RH_ptrs);
		static float getSumEnergy(const std::vector<std::pair<const EcalRecHit*,float> >&RH_ptrs_fracs);
		static float computeWeight(float eRH, float energyTotal, int weightedPositionMethod);
		
};

#endif
