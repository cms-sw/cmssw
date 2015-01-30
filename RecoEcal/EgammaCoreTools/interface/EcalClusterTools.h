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

/** Note about "noZS" and the new templated version of EcalClusterTools
 *
 * "noZS" means "no zero-suppression" and means that the rechit quantity is 
 * computed without fractions and always including the full matrix of crystals
 * about the seed.
 *
 * To access the noZS variables use the "noZS::EcalClusterTools"
 * The interface at this point is the same as the standard tools, except the
 * behavior is altered to provide the output as described above.
 *
 * This was achieved by templating the EcalClusterTools.
 * Do not use EcalClusterToolsT<> directly.
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

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "CLHEP/Geometry/Transform3D.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"


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

template<bool noZS>
class EcalClusterToolsT {
        public:
                EcalClusterToolsT() {};
                ~EcalClusterToolsT() {};

                // various energies in the matrix nxn surrounding the maximum energy crystal of the input cluster
		//we use an eta/phi coordinate system rather than phi/eta
                //note e3x2 does not have a definate eta/phi geometry, it takes the maximum 3x2 block containing the 
                //seed regardless of whether that 3 in eta or phi
                static float e1x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );


                static float e3x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );


                static float e1x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );

                static float e5x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );

                static float e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );

                static float e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );

                static float e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );

                static float e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology);

                static float e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                static int   n5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );

                // energy in the 2x5 strip right of the max crystal (does not contain max crystal)
		// 2 crystals wide in eta, 5 wide in phi.
                static float e2x5Right( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                // energy in the 2x5 strip left of the max crystal (does not contain max crystal)

                static float e2x5Left( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                // energy in the 5x2 strip above the max crystal (does not contain max crystal)
		// 5 crystals wide in eta, 2 wide in phi.

                static float e2x5Top( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                // energy in the 5x2 strip below the max crystal (does not contain max crystal)                

                static float e2x5Bottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );
                // energy in a 2x5 strip containing the seed (max) crystal.
                // 2 crystals wide in eta, 5 wide in phi.
                // it is the maximum of either (1x5left + 1x5center) or (1x5right + 1x5center)
                static float e2x5Max( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology );

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

                static std::vector<float> energyBasketFractionPhi( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits);

                // return a vector v with v[0] = etaLat, v[1] = phiLat, v[2] = lat
                static std::vector<float> lat( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, bool logW = true, float w0 = 4.7 );

                // return a vector v with v[0] = covEtaEta, v[1] = covEtaPhi, v[2] = covPhiPhi

                static std::vector<float> covariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, const CaloGeometry* geometry, float w0 = 4.7);

                // return a vector v with v[0] = covIEtaIEta, v[1] = covIEtaIPhi, v[2] = covIPhiIPhi
                //this function calculates differences in eta/phi in units of crystals not global eta/phi
                //this is gives better performance in the crack regions of the calorimeter but gives otherwise identical results to covariances function
                //   except that it doesnt need an eta based correction funtion in the endcap 
                //it is multipled by an approprate crystal size to ensure it gives similar values to covariances(...)
                //
                //Warning: covIEtaIEta has been studied by egamma, but so far covIPhiIPhi hasnt been studied extensively so there could be a bug in 
                //         the covIPhiIEta or covIPhiIPhi calculations. I dont think there is but as it hasnt been heavily used, there might be one
                static std::vector<float> localCovariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, float w0 = 4.7);
                
                static std::vector<float> scLocalCovariances(const reco::SuperCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology, float w0 = 4.7);

                // cluster second moments with respect to principal axes:
                static Cluster2ndMoments cluster2ndMoments( const reco::BasicCluster &basicCluster, const EcalRecHitCollection &recHits, double phiCorrectionFactor=0.8, double w0=4.7, bool useLogWeights=true);

                static Cluster2ndMoments cluster2ndMoments( const reco::SuperCluster &superCluster, const EcalRecHitCollection &recHits, double phiCorrectionFactor=0.8, double w0=4.7, bool useLogWeights=true);
                static Cluster2ndMoments cluster2ndMoments( const std::vector<std::pair<const EcalRecHit*, float> >& RH_ptrs_fracs, double  phiCorrectionFactor=0.8, double  w0=4.7, bool useLogWeights=true);

                static double zernike20( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0 = 6.6, bool logW = true, float w0 = 4.7 );
                static double zernike42( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0 = 6.6, bool logW = true, float w0 = 4.7 );

                // get the detId's of a matrix centered in the maximum energy crystal = (0,0)
                // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
                static std::vector<DetId> matrixDetId( const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );

                // get the energy deposited in a matrix centered in the maximum energy crystal = (0,0)
                // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
                static float matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );
                static int matrixSize( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );

                static float getFraction( const std::vector< std::pair<DetId, float> > &v_id, DetId id);
                // get the DetId and the energy of the maximum energy crystal in a vector of DetId
                static std::pair<DetId, float> getMaximum( const std::vector< std::pair<DetId, float> > &v_id, const EcalRecHitCollection *recHits);

                // get the energy of a DetId, return 0 if the DetId is not in the collection
                static float recHitEnergy(DetId id, const EcalRecHitCollection *recHits);

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

                //return energy weighted mean distance from the seed crystal in number of crystals
                //<iEta,iPhi>, iPhi is not defined for endcap and is returned as zero 
                static std::pair<float,float>  mean5x5PositionInLocalCrysCoord(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology);

                static std::pair<float,float> mean5x5PositionInXY(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology);

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

// implementation
template<bool noZS> 
float EcalClusterToolsT<noZS>::getFraction( const std::vector< std::pair<DetId, float> > &v_id, DetId id
			  ){
  if(noZS) return 1.0;
  float frac = 0.0;
  for ( size_t i = 0; i < v_id.size(); ++i ) {
    if(v_id[i].first.rawId()==id.rawId()){
      frac= v_id[i].second;
      break;
    }
  }
  return frac;
}

template<bool noZS>
std::pair<DetId, float> EcalClusterToolsT<noZS>::getMaximum( const std::vector< std::pair<DetId, float> > &v_id, const EcalRecHitCollection *recHits)
{
    float max = 0;
    DetId id(0);
    for ( size_t i = 0; i < v_id.size(); ++i ) {
      float energy = recHitEnergy( v_id[i].first, recHits ) * (noZS ? 1.0 : v_id[i].second);
        if ( energy > max ) {
            max = energy;
            id = v_id[i].first;
        }
    }
    return std::pair<DetId, float>(id, max);
}

template<bool noZS>
std::pair<DetId, float> EcalClusterToolsT<noZS>::getMaximum( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits)
{
    return getMaximum( cluster.hitsAndFractions(), recHits );
}


template<bool noZS>
float EcalClusterToolsT<noZS>::recHitEnergy(DetId id, const EcalRecHitCollection *recHits)
{
  if ( id == DetId(0) ) {
    return 0;
  } else {
    EcalRecHitCollection::const_iterator it = recHits->find( id );
    if ( it != recHits->end() ) {
      if( noZS && ( it->checkFlag(EcalRecHit::kTowerRecovered) ||
		    it->checkFlag(EcalRecHit::kWeird) ||
		    (it->detid().subdetId() == EcalBarrel && 
		     it->checkFlag(EcalRecHit::kDiWeird) ) 
		    ) 
	  ) {
	return 0.0;
      } else {
	return (*it).energy();
      }
    } else {
      //throw cms::Exception("EcalRecHitNotFound") << "The recHit corresponding to the DetId" << id.rawId() << " not found in the EcalRecHitCollection";
      // the recHit is not in the collection (hopefully zero suppressed)
      return 0;
    }
  }
  return 0;
}


// Returns the energy in a rectangle of crystals
// specified in eta by ixMin and ixMax
//       and in phi by iyMin and iyMax
//
// Reference picture (X=seed crystal)
//    iy ___________
//     2 |_|_|_|_|_|
//     1 |_|_|_|_|_|
//     0 |_|_|X|_|_|
//    -1 |_|_|_|_|_|
//    -2 |_|_|_|_|_|
//      -2 -1 0 1 2 ix
template<bool noZS>
float EcalClusterToolsT<noZS>::matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
  //take into account fractions
    // fast version
    CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
    float energy = 0;
    const std::vector< std::pair<DetId, float> >& v_id = cluster.hitsAndFractions();
    for ( int i = ixMin; i <= ixMax; ++i ) {
        for ( int j = iyMin; j <= iyMax; ++j ) {
	  cursor.home();
	  cursor.offsetBy( i, j );
	  float frac=getFraction(v_id,*cursor);
	  energy += recHitEnergy( *cursor, recHits )*frac;
        }
    }
    // slow elegant version
    //float energy = 0;
    //std::vector<DetId> v_id = matrixDetId( topology, id, ixMin, ixMax, iyMin, iyMax );
    //for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {
    //        energy += recHitEnergy( *it, recHits );
    //}
    return energy;
}

template<bool noZS>
int EcalClusterToolsT<noZS>::matrixSize( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
    // fast version
    CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
    int result = 0;
    const std::vector< std::pair<DetId, float> >& v_id = cluster.hitsAndFractions();
    for ( int i = ixMin; i <= ixMax; ++i ) {
        for ( int j = iyMin; j <= iyMax; ++j ) {
            cursor.home();
            cursor.offsetBy( i, j );
            float frac=getFraction(v_id,*cursor);
            float energy = recHitEnergy( *cursor, recHits )*frac;
            if (energy > 0) result++;
        }
    }
    return result;
}


template<bool noZS>
std::vector<DetId> EcalClusterToolsT<noZS>::matrixDetId( const CaloTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
    CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
    std::vector<DetId> v;
    for ( int i = ixMin; i <= ixMax; ++i ) {
        for ( int j = iyMin; j <= iyMax; ++j ) {
            cursor.home();
            cursor.offsetBy( i, j );
            if ( *cursor != DetId(0) ) v.push_back( *cursor );
        }
    }
    return v;
}


template<bool noZS>
float EcalClusterToolsT<noZS>::e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    std::list<float> energies;
    float max_E =  matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 0 );
    max_E = std::max( max_E, matrixEnergy( cluster, recHits, topology, id, -1, 0,  0, 1 ) );
    max_E = std::max( max_E, matrixEnergy( cluster, recHits, topology, id,  0, 1,  0, 1 ) );
    max_E = std::max( max_E, matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 0 ) );
    return max_E;
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    float max_E = matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 0 );
    max_E = std::max( max_E, matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 1 ) );
    max_E = std::max( max_E, matrixEnergy( cluster, recHits, topology, id, -1, 1,  0, 1 ) );
    max_E = std::max( max_E, matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 1 ) );
    return max_E;
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 1 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;    
    float max_E = matrixEnergy( cluster, recHits, topology, id, -1, 2, -2, 1 );
    max_E = std::max( max_E, matrixEnergy( cluster, recHits, topology, id, -2, 1, -2, 1 ) );
    max_E = std::max( max_E, matrixEnergy( cluster, recHits, topology, id, -2, 1, -1, 2 ) );
    max_E = std::max( max_E, matrixEnergy( cluster, recHits, topology, id, -1, 2, -1, 2 ) );
    return max_E;
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, -2, 2 );
}

template<bool noZS>
int EcalClusterToolsT<noZS>::n5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixSize( cluster, recHits, topology, id, -2, 2, -2, 2 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
    return getMaximum( cluster.hitsAndFractions(), recHits ).second;
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
    std::vector<float> energies;
    const std::vector< std::pair<DetId, float> >& v_id = cluster.hitsAndFractions();
    energies.reserve( v_id.size() );
    if ( v_id.size() < 2 ) return 0;
    for ( size_t i = 0; i < v_id.size(); ++i ) {
      energies.push_back( recHitEnergy( v_id[i].first, recHits ) * (noZS ? 1.0 : v_id[i].second) );
    }
    std::partial_sort( energies.begin(), energies.begin()+2, energies.end(), std::greater<float>() );
    return energies[1];


}

template<bool noZS>
float EcalClusterToolsT<noZS>::e2x5Right( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 1, 2, -2, 2 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e2x5Left( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, -1, -2, 2 );
}

template<bool noZS> 
float EcalClusterToolsT<noZS>::e2x5Top( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, 1, 2 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e2x5Bottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, -2, -1 );
}

// Energy in 2x5 strip containing the max crystal.
// Adapted from code by Sam Harper
template<bool noZS>
float EcalClusterToolsT<noZS>::e2x5Max( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id =      getMaximum( cluster.hitsAndFractions(), recHits ).first;

    // 1x5 strip left of seed
    float left   = matrixEnergy( cluster, recHits, topology, id, -1, -1, -2, 2 );
    // 1x5 strip right of seed
    float right  = matrixEnergy( cluster, recHits, topology, id,  1,  1, -2, 2 );
    // 1x5 strip containing seed
    float centre = matrixEnergy( cluster, recHits, topology, id,  0,  0, -2, 2 );

    // Return the maximum of (left+center) or (right+center) strip
    return left > right ? left+centre : right+centre;
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e1x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, -2, 2 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e5x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -2, 2, 0, 0 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e1x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, -1, 1 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::e3x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -1, 1, 0, 0 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::eLeft( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, -1, -1, 0, 0 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::eRight( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 1, 1, 0, 0 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::eTop( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, 1, 1 );
}

template<bool noZS>
float EcalClusterToolsT<noZS>::eBottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology* topology )
{
    DetId id = getMaximum( cluster.hitsAndFractions(), recHits ).first;
    return matrixEnergy( cluster, recHits, topology, id, 0, 0, -1, -1 );
}

template<bool noZS>
std::vector<float> EcalClusterToolsT<noZS>::energyBasketFractionEta( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
    std::vector<float> basketFraction( 2 * EBDetId::kModulesPerSM );
    float clusterEnergy = cluster.energy();
    const std::vector< std::pair<DetId, float> >& v_id = cluster.hitsAndFractions();
    if ( v_id[0].first.subdetId() != EcalBarrel ) {
        edm::LogWarning("EcalClusterToolsT<noZS>::energyBasketFractionEta") << "Trying to get basket fraction for endcap basic-clusters. Basket fractions can be obtained ONLY for barrel basic-clusters. Returning empty vector.";
        return basketFraction;
    }
    for ( size_t i = 0; i < v_id.size(); ++i ) {
        basketFraction[ EBDetId(v_id[i].first).im()-1 + EBDetId(v_id[i].first).positiveZ()*EBDetId::kModulesPerSM ] += recHitEnergy( v_id[i].first, recHits ) * v_id[i].second / clusterEnergy;
    }
    std::sort( basketFraction.rbegin(), basketFraction.rend() );
    return basketFraction;
}

template<bool noZS>
std::vector<float> EcalClusterToolsT<noZS>::energyBasketFractionPhi( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
    std::vector<float> basketFraction( 2 * (EBDetId::MAX_IPHI / EBDetId::kCrystalsInPhi) );
    float clusterEnergy = cluster.energy();
    const std::vector< std::pair<DetId, float> >& v_id = cluster.hitsAndFractions();
    if ( v_id[0].first.subdetId() != EcalBarrel ) {
        edm::LogWarning("EcalClusterToolsT<noZS>::energyBasketFractionPhi") << "Trying to get basket fraction for endcap basic-clusters. Basket fractions can be obtained ONLY for barrel basic-clusters. Returning empty vector.";
        return basketFraction;
    }
    for ( size_t i = 0; i < v_id.size(); ++i ) {
      basketFraction[ (EBDetId(v_id[i].first).iphi()-1)/EBDetId::kCrystalsInPhi + EBDetId(v_id[i].first).positiveZ()*EBDetId::kTowersInPhi] += recHitEnergy( v_id[i].first, recHits ) * (noZS ? 1.0 : v_id[i].second) / clusterEnergy;
    }
    std::sort( basketFraction.rbegin(), basketFraction.rend() );
    return basketFraction;
}

template<bool noZS>
std::vector<typename EcalClusterToolsT<noZS>::EcalClusterEnergyDeposition> EcalClusterToolsT<noZS>::getEnergyDepTopology( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, bool logW, float w0 )
{
    std::vector<typename EcalClusterToolsT<noZS>::EcalClusterEnergyDeposition> energyDistribution;
    // init a map of the energy deposition centered on the
    // cluster centroid. This is for momenta calculation only.
    CLHEP::Hep3Vector clVect(cluster.position().x(), cluster.position().y(), cluster.position().z());
    CLHEP::Hep3Vector clDir(clVect);
    clDir*=1.0/clDir.mag();
    // in the transverse plane, axis perpendicular to clusterDir
    CLHEP::Hep3Vector theta_axis(clDir.y(),-clDir.x(),0.0);
    theta_axis *= 1.0/theta_axis.mag();
    CLHEP::Hep3Vector phi_axis = theta_axis.cross(clDir);

    const std::vector< std::pair<DetId, float> >& clusterDetIds = cluster.hitsAndFractions();

    EcalClusterEnergyDeposition clEdep;
    EcalRecHit testEcalRecHit;
    std::vector< std::pair<DetId, float> >::const_iterator posCurrent;
    // loop over crystals
    for(posCurrent=clusterDetIds.begin(); posCurrent!=clusterDetIds.end(); ++posCurrent) {
        EcalRecHitCollection::const_iterator itt = recHits->find( (*posCurrent).first );
        testEcalRecHit=*itt;

        if(( (*posCurrent).first != DetId(0)) && (recHits->find( (*posCurrent).first ) != recHits->end())) {
	  clEdep.deposited_energy = testEcalRecHit.energy() * (noZS ? 1.0 : (*posCurrent).second);
            // if logarithmic weight is requested, apply cut on minimum energy of the recHit
            if(logW) {
                //double w0 = parameterMap_.find("W0")->second;

	      double weight = std::max(0.0, w0 + log(std::abs(clEdep.deposited_energy)/cluster.energy()) );
                if(weight==0) {
                    LogDebug("ClusterShapeAlgo") << "Crystal has insufficient energy: E = " 
                        << clEdep.deposited_energy << " GeV; skipping... ";
                    continue;
                }
                else LogDebug("ClusterShapeAlgo") << "===> got crystal. Energy = " << clEdep.deposited_energy << " GeV. ";
            }
            DetId id_ = (*posCurrent).first;
            const CaloCellGeometry *this_cell = geometry->getSubdetectorGeometry(id_)->getGeometry(id_);
            GlobalPoint cellPos = this_cell->getPosition();
            CLHEP::Hep3Vector gblPos (cellPos.x(),cellPos.y(),cellPos.z()); //surface position?
            // Evaluate the distance from the cluster centroid
            CLHEP::Hep3Vector diff = gblPos - clVect;
            // Important: for the moment calculation, only the "lateral distance" is important
            // "lateral distance" r_i = distance of the digi position from the axis Origin-Cluster Center
            // ---> subtract the projection on clDir
            CLHEP::Hep3Vector DigiVect = diff - diff.dot(clDir)*clDir;
            clEdep.r = DigiVect.mag();
            LogDebug("ClusterShapeAlgo") << "E = " << clEdep.deposited_energy
                << "\tdiff = " << diff.mag()
                << "\tr = " << clEdep.r;
            clEdep.phi = DigiVect.angle(theta_axis);
            if(DigiVect.dot(phi_axis)<0) clEdep.phi = 2 * M_PI - clEdep.phi;
            energyDistribution.push_back(clEdep);
        }
    } 
    return energyDistribution;
}

template<bool noZS>
std::vector<float> EcalClusterToolsT<noZS>::lat( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, bool logW, float w0 )
{
    std::vector<EcalClusterToolsT::EcalClusterEnergyDeposition> energyDistribution = getEnergyDepTopology( cluster, recHits, geometry, logW, w0 );

    std::vector<float> lat;
    double r, redmoment=0;
    double phiRedmoment = 0 ;
    double etaRedmoment = 0 ;
    int n,n1,n2,tmp;
    int clusterSize=energyDistribution.size();
    float etaLat_, phiLat_, lat_;
    if (clusterSize<3) {
        etaLat_ = 0.0 ; 
        lat_ = 0.0;
        lat.push_back(0.);
        lat.push_back(0.);
        lat.push_back(0.);
        return lat; 
    }

    n1=0; n2=1;
    if (energyDistribution[1].deposited_energy > 
            energyDistribution[0].deposited_energy) 
    {
        tmp=n2; n2=n1; n1=tmp;
    }
    for (int i=2; i<clusterSize; i++) {
        n=i;
        if (energyDistribution[i].deposited_energy > 
                energyDistribution[n1].deposited_energy) 
        {
            tmp = n2;
            n2 = n1; n1 = i; n=tmp;
        } else {
            if (energyDistribution[i].deposited_energy > 
                    energyDistribution[n2].deposited_energy) 
            {
                tmp=n2; n2=i; n=tmp;
            }
        }

        r = energyDistribution[n].r;
        redmoment += r*r* energyDistribution[n].deposited_energy;
        double rphi = r * cos (energyDistribution[n].phi) ;
        phiRedmoment += rphi * rphi * energyDistribution[n].deposited_energy;
        double reta = r * sin (energyDistribution[n].phi) ;
        etaRedmoment += reta * reta * energyDistribution[n].deposited_energy;
    } 
    double e1 = energyDistribution[n1].deposited_energy;
    double e2 = energyDistribution[n2].deposited_energy;

    lat_ = redmoment/(redmoment+2.19*2.19*(e1+e2));
    phiLat_ = phiRedmoment/(phiRedmoment+2.19*2.19*(e1+e2));
    etaLat_ = etaRedmoment/(etaRedmoment+2.19*2.19*(e1+e2));

    lat.push_back(etaLat_);
    lat.push_back(phiLat_);
    lat.push_back(lat_);
    return lat;
}

template<bool noZS>
math::XYZVector EcalClusterToolsT<noZS>::meanClusterPosition( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloTopology *topology, const CaloGeometry *geometry )
{
    // find mean energy position of a 5x5 cluster around the maximum
    math::XYZVector meanPosition(0.0, 0.0, 0.0);
    const std::vector<std::pair<DetId,float> >& hsAndFs = cluster.hitsAndFractions();
    std::vector<DetId> v_id = matrixDetId( topology, getMaximum( cluster, recHits ).first, -2, 2, -2, 2 );
    for( const std::pair<DetId,float>& hitAndFrac : hsAndFs ) {
      for( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {
	if( hitAndFrac.first != *it && !noZS) continue;
	GlobalPoint positionGP = geometry->getSubdetectorGeometry( *it )->getGeometry( *it )->getPosition();
	math::XYZVector position(positionGP.x(),positionGP.y(),positionGP.z());
	meanPosition = meanPosition + recHitEnergy( *it, recHits ) * position * hitAndFrac.second;
      }
      if(noZS) break;
    }
    return meanPosition / e5x5( cluster, recHits, topology );
}

//returns mean energy weighted eta/phi in crystals from the seed
//iPhi is not defined for endcap and is returned as zero
//return <eta,phi>
//we have an issue in working out what to do for negative energies
//I (Sam Harper) think it makes sense to ignore crystals with E<0 in the calculation as they are ignored
//in the sigmaIEtaIEta calculation (well they arent fully ignored, they do still contribute to the e5x5 sum
//in the sigmaIEtaIEta calculation but not here)
template<bool noZS>
std::pair<float,float>  EcalClusterToolsT<noZS>::mean5x5PositionInLocalCrysCoord(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology)
{
    DetId seedId =  getMaximum( cluster, recHits ).first;
    float meanDEta=0.;
    float meanDPhi=0.;
    float energySum=0.;

    const std::vector<std::pair<DetId,float> >& hsAndFs = cluster.hitsAndFractions();
    std::vector<DetId> v_id = matrixDetId( topology,seedId, -2, 2, -2, 2 );
    for( const std::pair<DetId,float>& hAndF : hsAndFs ) {
      for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {  
	if( hAndF.first != *it && !noZS ) continue;
        float energy = recHitEnergy(*it,recHits) * hAndF.second;
        if(energy<0.) continue;//skipping negative energy crystals
        meanDEta += energy * getNrCrysDiffInEta(*it,seedId);
        meanDPhi += energy * getNrCrysDiffInPhi(*it,seedId);	
        energySum +=energy;
      }
      if(noZS) break;
    }
    meanDEta /=energySum;
    meanDPhi /=energySum;
    return std::pair<float,float>(meanDEta,meanDPhi);
}

//returns mean energy weighted x/y in normalised crystal coordinates
//only valid for endcap, returns 0,0 for barrel
//we have an issue in working out what to do for negative energies
//I (Sam Harper) think it makes sense to ignore crystals with E<0 in the calculation as they are ignored
//in the sigmaIEtaIEta calculation (well they arent fully ignored, they do still contribute to the e5x5 sum
//in the sigmaIEtaIEta calculation but not here)
template<bool noZS>
std::pair<float,float> EcalClusterToolsT<noZS>::mean5x5PositionInXY(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology)
{
    DetId seedId =  getMaximum( cluster, recHits ).first;

    std::pair<float,float> meanXY(0.,0.);
    if(seedId.subdetId()==EcalBarrel) return meanXY;

    float energySum=0.;

    const std::vector<std::pair<DetId,float> >& hsAndFs = cluster.hitsAndFractions();
    std::vector<DetId> v_id = matrixDetId( topology,seedId, -2, 2, -2, 2 );
    for( const std::pair<DetId,float>& hAndF : hsAndFs ) {
      for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {  
	if( hAndF.first != *it && !noZS) continue;
        float energy = recHitEnergy(*it,recHits) * hAndF.second;
        if(energy<0.) continue;//skipping negative energy crystals
        meanXY.first += energy * getNormedIX(*it);
        meanXY.second += energy * getNormedIY(*it);
        energySum +=energy;
      }
      if(noZS) break;
    }
    meanXY.first/=energySum;
    meanXY.second/=energySum;
    return meanXY;
}

template<bool noZS>
std::vector<float> EcalClusterToolsT<noZS>::covariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloTopology *topology, const CaloGeometry* geometry, float w0)
{
    float e_5x5 = e5x5( cluster, recHits, topology );
    float covEtaEta, covEtaPhi, covPhiPhi;
    if (e_5x5 >= 0.) {
        //double w0_ = parameterMap_.find("W0")->second;
        const std::vector< std::pair<DetId, float>>& v_id =cluster.hitsAndFractions();
        math::XYZVector meanPosition = meanClusterPosition( cluster, recHits, topology, geometry );

        // now we can calculate the covariances
        double numeratorEtaEta = 0;
        double numeratorEtaPhi = 0;
        double numeratorPhiPhi = 0;
        double denominator     = 0;

        DetId id = getMaximum( v_id, recHits ).first;
        CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
        for ( int i = -2; i <= 2; ++i ) {
            for ( int j = -2; j <= 2; ++j ) {
                cursor.home();
                cursor.offsetBy( i, j );
                float frac=getFraction(v_id,*cursor);
                float energy = recHitEnergy( *cursor, recHits )*frac;

                if ( energy <= 0 ) continue;

                GlobalPoint position = geometry->getSubdetectorGeometry(*cursor)->getGeometry(*cursor)->getPosition();

                double dPhi = position.phi() - meanPosition.phi();
                if (dPhi > + Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }
                if (dPhi < - Geom::pi()) { dPhi = Geom::twoPi() + dPhi; }

                double dEta = position.eta() - meanPosition.eta();
                double w = 0.;
                w = std::max(0.0f, w0 + std::log( energy / e_5x5 ));

                denominator += w;
                numeratorEtaEta += w * dEta * dEta;
                numeratorEtaPhi += w * dEta * dPhi;
                numeratorPhiPhi += w * dPhi * dPhi;
            }
        }

        if (denominator != 0.0) {
            covEtaEta =  numeratorEtaEta / denominator;
            covEtaPhi =  numeratorEtaPhi / denominator;
            covPhiPhi =  numeratorPhiPhi / denominator;
        } else {
            covEtaEta = 999.9;
            covEtaPhi = 999.9;
            covPhiPhi = 999.9;
        }

    } else {
        // Warn the user if there was no energy in the cells and return zeroes.
        //       std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
        covEtaEta = 0;
        covEtaPhi = 0;
        covPhiPhi = 0;
    }
    std::vector<float> v;
    v.push_back( covEtaEta );
    v.push_back( covEtaPhi );
    v.push_back( covPhiPhi );
    return v;
}

//for covIEtaIEta,covIEtaIPhi and covIPhiIPhi are defined but only covIEtaIEta has been actively studied
//instead of using absolute eta/phi it counts crystals normalised so that it gives identical results to normal covariances except near the cracks where of course its better 
//it also does not require any eta correction function in the endcap
//it is multipled by an approprate crystal size to ensure it gives similar values to covariances(...)
template<bool noZS>
std::vector<float> EcalClusterToolsT<noZS>::localCovariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology,float w0)
{

    float e_5x5 = e5x5( cluster, recHits, topology );
    float covEtaEta, covEtaPhi, covPhiPhi;

    if (e_5x5 >= 0.) {
        //double w0_ = parameterMap_.find("W0")->second;
        const std::vector< std::pair<DetId, float> >& v_id = cluster.hitsAndFractions();
        std::pair<float,float> mean5x5PosInNrCrysFromSeed =  mean5x5PositionInLocalCrysCoord( cluster, recHits, topology );
        std::pair<float,float> mean5x5XYPos =  mean5x5PositionInXY(cluster,recHits,topology);

        // now we can calculate the covariances
        double numeratorEtaEta = 0;
        double numeratorEtaPhi = 0;
        double numeratorPhiPhi = 0;
        double denominator     = 0;

        //these allow us to scale the localCov by the crystal size 
        //so that the localCovs have the same average value as the normal covs
        const double barrelCrysSize = 0.01745; //approximate size of crystal in eta,phi in barrel
        const double endcapCrysSize = 0.0447; //the approximate crystal size sigmaEtaEta was corrected to in the endcap

        DetId seedId = getMaximum( v_id, recHits ).first;

        bool isBarrel=seedId.subdetId()==EcalBarrel;
        const double crysSize = isBarrel ? barrelCrysSize : endcapCrysSize;

        CaloNavigator<DetId> cursor = CaloNavigator<DetId>( seedId, topology->getSubdetectorTopology( seedId ) );

        for ( int eastNr = -2; eastNr <= 2; ++eastNr ) { //east is eta in barrel
            for ( int northNr = -2; northNr <= 2; ++northNr ) { //north is phi in barrel
                cursor.home();
                cursor.offsetBy( eastNr, northNr);
                float frac = getFraction(v_id,*cursor);
                float energy = recHitEnergy( *cursor, recHits )*frac;
                if ( energy <= 0 ) continue;

                float dEta = getNrCrysDiffInEta(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.first;
                float dPhi = 0;

                if(isBarrel)  dPhi = getNrCrysDiffInPhi(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.second;
                else dPhi = getDPhiEndcap(*cursor,mean5x5XYPos.first,mean5x5XYPos.second);


                double w = std::max(0.0f,w0 + std::log( energy / e_5x5 ));

                denominator += w;
                numeratorEtaEta += w * dEta * dEta;
                numeratorEtaPhi += w * dEta * dPhi;
                numeratorPhiPhi += w * dPhi * dPhi;
            } //end east loop
        }//end north loop


        //multiplying by crysSize to make the values compariable to normal covariances
        if (denominator != 0.0) {
            covEtaEta =  crysSize*crysSize* numeratorEtaEta / denominator;
            covEtaPhi =  crysSize*crysSize* numeratorEtaPhi / denominator;
            covPhiPhi =  crysSize*crysSize* numeratorPhiPhi / denominator;
        } else {
            covEtaEta = 999.9;
            covEtaPhi = 999.9;
            covPhiPhi = 999.9;
        }


    } else {
        // Warn the user if there was no energy in the cells and return zeroes.
        //       std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
        covEtaEta = 0;
        covEtaPhi = 0;
        covPhiPhi = 0;
    }
    std::vector<float> v;
    v.push_back( covEtaEta );
    v.push_back( covEtaPhi );
    v.push_back( covPhiPhi );
    return v;
}

template<bool noZS>
double EcalClusterToolsT<noZS>::zernike20( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0, bool logW, float w0 )
{
    return absZernikeMoment( cluster, recHits, geometry, 2, 0, R0, logW, w0 );
}

template<bool noZS>
double EcalClusterToolsT<noZS>::zernike42( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, double R0, bool logW, float w0 )
{
    return absZernikeMoment( cluster, recHits, geometry, 4, 2, R0, logW, w0 );
}

template<bool noZS>
double EcalClusterToolsT<noZS>::absZernikeMoment( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, int n, int m, double R0, bool logW, float w0 )
{
    // 1. Check if n,m are correctly
    if ((m>n) || ((n-m)%2 != 0) || (n<0) || (m<0)) return -1;

    // 2. Check if n,R0 are within validity Range :
    // n>20 or R0<2.19cm  just makes no sense !
    if ((n>20) || (R0<=2.19)) return -1;
    if (n<=5) return fast_AbsZernikeMoment(cluster, recHits, geometry, n, m, R0, logW, w0 );
    else return calc_AbsZernikeMoment(cluster, recHits, geometry, n, m, R0, logW, w0 );
}

template<bool noZS>
double EcalClusterToolsT<noZS>::fast_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, int n, int m, double R0, bool logW, float w0 )
{
    double r,ph,e,Re=0,Im=0;
    double TotalEnergy = cluster.energy();
    int index = (n/2)*(n/2)+(n/2)+m;
    std::vector<EcalClusterEnergyDeposition> energyDistribution = getEnergyDepTopology( cluster, recHits, geometry, logW, w0 );
    int clusterSize = energyDistribution.size();
    if(clusterSize < 3) return 0.0;

    for (int i=0; i<clusterSize; i++)
    { 
        r = energyDistribution[i].r / R0;
        if (r<1) {
            std::vector<double> pol;
            pol.push_back( f00(r) );
            pol.push_back( f11(r) );
            pol.push_back( f20(r) );
            pol.push_back( f22(r) );
            pol.push_back( f31(r) );
            pol.push_back( f33(r) );
            pol.push_back( f40(r) );
            pol.push_back( f42(r) );
            pol.push_back( f44(r) );
            pol.push_back( f51(r) );
            pol.push_back( f53(r) );
            pol.push_back( f55(r) );
            ph = (energyDistribution[i]).phi;
            e = energyDistribution[i].deposited_energy;
            Re = Re + e/TotalEnergy * pol[index] * cos( (double) m * ph);
            Im = Im - e/TotalEnergy * pol[index] * sin( (double) m * ph);
        }
    }
    return sqrt(Re*Re+Im*Im);
}

template<bool noZS>
double EcalClusterToolsT<noZS>::calc_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloGeometry *geometry, int n, int m, double R0, bool logW, float w0 )
{
    double r, ph, e, Re=0, Im=0, f_nm;
    double TotalEnergy = cluster.energy();
    std::vector<EcalClusterEnergyDeposition> energyDistribution = getEnergyDepTopology( cluster, recHits, geometry, logW, w0 );
    int clusterSize=energyDistribution.size();
    if(clusterSize<3) return 0.0;

    for (int i = 0; i < clusterSize; ++i)
    { 
        r = energyDistribution[i].r / R0;
        if (r < 1) {
            ph = energyDistribution[i].phi;
            e = energyDistribution[i].deposited_energy;
            f_nm = 0;
            for (int s=0; s<=(n-m)/2; s++) {
                if (s%2==0) { 
                    f_nm = f_nm + factorial(n-s)*pow(r,(double) (n-2*s))/(factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
                } else {
                    f_nm = f_nm - factorial(n-s)*pow(r,(double) (n-2*s))/(factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
                }
            }
            Re = Re + e/TotalEnergy * f_nm * cos( (double) m*ph);
            Im = Im - e/TotalEnergy * f_nm * sin( (double) m*ph);
        }
    }
    return sqrt(Re*Re+Im*Im);
}

//returns the crystal 'eta' from the det id
//it is defined as the number of crystals from the centre in the eta direction
//for the barrel with its eta/phi geometry it is always integer
//for the endcap it is fractional due to the x/y geometry
template<bool noZS>
float  EcalClusterToolsT<noZS>::getIEta(const DetId& id)
{
    if(id.det()==DetId::Ecal){
        if(id.subdetId()==EcalBarrel){
            EBDetId ebId(id);
            return ebId.ieta();
        }else if(id.subdetId()==EcalEndcap){
            float iXNorm = getNormedIX(id);
            float iYNorm = getNormedIY(id);

            return std::sqrt(iXNorm*iXNorm+iYNorm*iYNorm);
        }
    }
    return 0.;    
}


//returns the crystal 'phi' from the det id
//it is defined as the number of crystals from the centre in the phi direction
//for the barrel with its eta/phi geometry it is always integer
//for the endcap it is not defined 
template<bool noZS>
float  EcalClusterToolsT<noZS>::getIPhi(const DetId& id)
{
    if(id.det()==DetId::Ecal){
        if(id.subdetId()==EcalBarrel){
            EBDetId ebId(id);
            return ebId.iphi();
        }
    }
    return 0.;    
}

//want to map 1=-50,50=-1,51=1 and 100 to 50 so sub off one if zero or neg
template<bool noZS>
float EcalClusterToolsT<noZS>::getNormedIX(const DetId& id)
{
    if(id.det()==DetId::Ecal && id.subdetId()==EcalEndcap){
        EEDetId eeId(id);      
        int iXNorm  = eeId.ix()-50;
        if(iXNorm<=0) iXNorm--;
        return iXNorm;
    }
    return 0;
}

//want to map 1=-50,50=-1,51=1 and 100 to 50 so sub off one if zero or neg
template<bool noZS>
float EcalClusterToolsT<noZS>::getNormedIY(const DetId& id)
{
    if(id.det()==DetId::Ecal && id.subdetId()==EcalEndcap){
        EEDetId eeId(id);      
        int iYNorm  = eeId.iy()-50;
        if(iYNorm<=0) iYNorm--;
        return iYNorm;
    }
    return 0;
}

//nr crystals crysId is away from orgin id in eta
template<bool noZS>
float EcalClusterToolsT<noZS>::getNrCrysDiffInEta(const DetId& crysId,const DetId& orginId)
{
    float crysIEta = getIEta(crysId);
    float orginIEta = getIEta(orginId);
    bool isBarrel = orginId.subdetId()==EcalBarrel;

    float nrCrysDiff = crysIEta-orginIEta;

    //no iEta=0 in barrel, so if go from positive to negative
    //need to reduce abs(detEta) by 1
    if(isBarrel){ 
        if(crysIEta*orginIEta<0){ // -1 to 1 transition
            if(crysIEta>0) nrCrysDiff--;
            else nrCrysDiff++;
        }
    }
    return nrCrysDiff;
}

//nr crystals crysId is away from orgin id in phi
template<bool noZS>
float EcalClusterToolsT<noZS>::getNrCrysDiffInPhi(const DetId& crysId,const DetId& orginId)
{
    float crysIPhi = getIPhi(crysId);
    float orginIPhi = getIPhi(orginId);
    bool isBarrel = orginId.subdetId()==EcalBarrel;

    float nrCrysDiff = crysIPhi-orginIPhi;

    if(isBarrel){ //if barrel, need to map into 0-180 
        if (nrCrysDiff > + 180) { nrCrysDiff = nrCrysDiff - 360; }
        if (nrCrysDiff < - 180) { nrCrysDiff = nrCrysDiff + 360; }
    }
    return nrCrysDiff;
}

//nr crystals crysId is away from mean phi in 5x5 in phi
template<bool noZS>
float EcalClusterToolsT<noZS>::getDPhiEndcap(const DetId& crysId,float meanX,float meanY)
{
    float iXNorm  = getNormedIX(crysId);
    float iYNorm  = getNormedIY(crysId);

    float hitLocalR2 = (iXNorm-meanX)*(iXNorm-meanX)+(iYNorm-meanY)*(iYNorm-meanY);
    float hitR2 = iXNorm*iXNorm+iYNorm*iYNorm;
    float meanR2 = meanX*meanX+meanY*meanY;
    float hitR = sqrt(hitR2);
    float meanR = sqrt(meanR2);

    float tmp = (hitR2+meanR2-hitLocalR2)/(2*hitR*meanR);
    if (tmp<-1) tmp =-1;
    if (tmp>1)  tmp=1;
    float phi = acos(tmp);
    float dPhi = hitR*phi;

    return dPhi;
}

template<bool noZS>
std::vector<float> EcalClusterToolsT<noZS>::scLocalCovariances(const reco::SuperCluster &cluster, const EcalRecHitCollection* recHits,const CaloTopology *topology, float w0)
{
    const reco::BasicCluster bcluster = *(cluster.seed());

    float e_5x5 = e5x5(bcluster, recHits, topology);
    float covEtaEta, covEtaPhi, covPhiPhi;

    if (e_5x5 >= 0.) {
        const std::vector<std::pair<DetId, float> >& v_id = cluster.hitsAndFractions();
        std::pair<float,float> mean5x5PosInNrCrysFromSeed =  mean5x5PositionInLocalCrysCoord(bcluster, recHits, topology);
        std::pair<float,float> mean5x5XYPos =  mean5x5PositionInXY(cluster,recHits,topology);
        // now we can calculate the covariances
        double numeratorEtaEta = 0;
        double numeratorEtaPhi = 0;
        double numeratorPhiPhi = 0;
        double denominator     = 0;

        const double barrelCrysSize = 0.01745; //approximate size of crystal in eta,phi in barrel
        const double endcapCrysSize = 0.0447; //the approximate crystal size sigmaEtaEta was corrected to in the endcap

        DetId seedId = getMaximum(v_id, recHits).first;  
        bool isBarrel=seedId.subdetId()==EcalBarrel;

        const double crysSize = isBarrel ? barrelCrysSize : endcapCrysSize;

        for (size_t i = 0; i < v_id.size(); ++i) {
            CaloNavigator<DetId> cursor = CaloNavigator<DetId>(v_id[i].first, topology->getSubdetectorTopology(v_id[i].first));
            float frac = getFraction(v_id,*cursor);
            float energy = recHitEnergy(*cursor, recHits)*frac;

            if (energy <= 0) continue;

            float dEta = getNrCrysDiffInEta(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.first;
            float dPhi = 0;
            if(isBarrel)  dPhi = getNrCrysDiffInPhi(*cursor,seedId) - mean5x5PosInNrCrysFromSeed.second;
            else dPhi = getDPhiEndcap(*cursor,mean5x5XYPos.first,mean5x5XYPos.second);



            double w = 0.;
            w = std::max(0.0f, w0 + std::log( energy / e_5x5 ));

            denominator += w;
            numeratorEtaEta += w * dEta * dEta;
            numeratorEtaPhi += w * dEta * dPhi;
            numeratorPhiPhi += w * dPhi * dPhi;
        }

        //multiplying by crysSize to make the values compariable to normal covariances
        if (denominator != 0.0) {
            covEtaEta =  crysSize*crysSize* numeratorEtaEta / denominator;
            covEtaPhi =  crysSize*crysSize* numeratorEtaPhi / denominator;
            covPhiPhi =  crysSize*crysSize* numeratorPhiPhi / denominator;
        } else {
            covEtaEta = 999.9;
            covEtaPhi = 999.9;
            covPhiPhi = 999.9;
        }

    } else {
        // Warn the user if there was no energy in the cells and return zeroes.
        // std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
        covEtaEta = 0;
        covEtaPhi = 0;
        covPhiPhi = 0;
    }

    std::vector<float> v;
    v.push_back( covEtaEta );
    v.push_back( covEtaPhi );
    v.push_back( covPhiPhi );

    return v;
}


// compute cluster second moments with respect to principal axes (eigenvectors of sEtaEta, sPhiPhi, sEtaPhi matrix)
// store also angle alpha between major axis and phi.
// takes into account shower elongation in phi direction due to magnetic field effect: 
// default value of 0.8 ensures sMaj = sMin for unconverted photons 
// (if phiCorrectionFactor=1 sMaj > sMin and alpha=0 also for unconverted photons)
template<bool noZS>
Cluster2ndMoments EcalClusterToolsT<noZS>::cluster2ndMoments( const reco::BasicCluster &basicCluster, const EcalRecHitCollection &recHits, double phiCorrectionFactor, double w0, bool useLogWeights) {

  // for now implemented only for EB:
  //  if( fabs( basicCluster.eta() ) < 1.479 ) { 

  std::vector<std::pair<const EcalRecHit*, float> > RH_ptrs_fracs;
  
  const std::vector< std::pair<DetId, float> >& myHitsPair = basicCluster.hitsAndFractions();
  
  for(unsigned int i=0; i<myHitsPair.size(); i++){
    //get pointer to recHit object
    EcalRecHitCollection::const_iterator myRH = recHits.find(myHitsPair[i].first);
    RH_ptrs_fracs.push_back(  std::make_pair(&(*myRH) , myHitsPair[i].second)  );
  }
  
  return EcalClusterToolsT<noZS>::cluster2ndMoments(RH_ptrs_fracs, phiCorrectionFactor, w0, useLogWeights);
}

template<bool noZS>
Cluster2ndMoments EcalClusterToolsT<noZS>::cluster2ndMoments( const reco::SuperCluster &superCluster, const EcalRecHitCollection &recHits, double phiCorrectionFactor, double w0, bool useLogWeights) {

  // for now returns second moments of supercluster seed cluster:
  Cluster2ndMoments returnMoments;
  returnMoments.sMaj = -1.;
  returnMoments.sMin = -1.;
  returnMoments.alpha = 0.;

  // for now implemented only for EB:
  //  if( fabs( superCluster.eta() ) < 1.479 ) { 
    returnMoments = EcalClusterToolsT<noZS>::cluster2ndMoments( *(superCluster.seed()), recHits, phiCorrectionFactor, w0, useLogWeights);
    //  }

  return returnMoments;

}

template<bool noZS>
Cluster2ndMoments EcalClusterToolsT<noZS>::cluster2ndMoments( const std::vector<std::pair<const EcalRecHit*, float> >& RH_ptrs_fracs, double phiCorrectionFactor, double w0, bool useLogWeights) {

  double mid_eta(0),mid_phi(0),mid_x(0),mid_y(0);
  
  double Etot = EcalClusterToolsT<noZS>::getSumEnergy(  RH_ptrs_fracs  );
 
  double max_phi=-10.;
  double min_phi=100.;
  
  
  std::vector<double> etaDetId;
  std::vector<double> phiDetId;
  std::vector<double> xDetId;
  std::vector<double> yDetId;
  std::vector<double> wiDetId;
 
  unsigned int nCry=0;
  double denominator=0.;
  bool isBarrel(1);

  // loop over rechits and compute weights:
  for(std::vector<std::pair<const EcalRecHit*, float> >::const_iterator rhf_ptr = RH_ptrs_fracs.begin(); rhf_ptr != RH_ptrs_fracs.end(); rhf_ptr++){
    const EcalRecHit* rh_ptr = rhf_ptr->first;


    //get iEta, iPhi
    double temp_eta(0),temp_phi(0),temp_x(0),temp_y(0);
    isBarrel = rh_ptr->detid().subdetId()==EcalBarrel;
    
    if(isBarrel) {
      temp_eta = (getIEta(rh_ptr->detid()) > 0. ? getIEta(rh_ptr->detid()) + 84.5 : getIEta(rh_ptr->detid()) + 85.5);
      temp_phi= getIPhi(rh_ptr->detid()) - 0.5;
    }
    else {
      temp_eta = getIEta(rh_ptr->detid());  
      temp_x =  getNormedIX(rh_ptr->detid());
      temp_y =  getNormedIY(rh_ptr->detid());
    }	  

    double temp_ene=rh_ptr->energy() * (noZS ? 1.0 : rhf_ptr->second);
    
    double temp_wi=((useLogWeights) ?
                    std::max(0.0, w0 + std::log( std::abs(temp_ene)/Etot ))
                    :  temp_ene);


    if(temp_phi>max_phi) max_phi=temp_phi;
    if(temp_phi<min_phi) min_phi=temp_phi;
    etaDetId.push_back(temp_eta);
    phiDetId.push_back(temp_phi);
    xDetId.push_back(temp_x);
    yDetId.push_back(temp_y);
    wiDetId.push_back(temp_wi);
    denominator+=temp_wi;
    nCry++;
  }

  if(isBarrel){
    // correct phi wrap-around:
    if(max_phi==359.5 && min_phi==0.5){ 
      for(unsigned int i=0; i<nCry; i++){
	if(phiDetId[i] - 179. > 0.) phiDetId[i]-=360.; 
	mid_phi+=phiDetId[i]*wiDetId[i];
	mid_eta+=etaDetId[i]*wiDetId[i];
      }
    } else{
      for(unsigned int i=0; i<nCry; i++){
	mid_phi+=phiDetId[i]*wiDetId[i];
	mid_eta+=etaDetId[i]*wiDetId[i];
      }
    }
  }else{
    for(unsigned int i=0; i<nCry; i++){
      mid_eta+=etaDetId[i]*wiDetId[i];      
      mid_x+=xDetId[i]*wiDetId[i];
      mid_y+=yDetId[i]*wiDetId[i];
    }
  }
  
  mid_eta/=denominator;
  mid_phi/=denominator;
  mid_x/=denominator;
  mid_y/=denominator;


  // See = sigma eta eta
  // Spp = (B field corrected) sigma phi phi
  // See = (B field corrected) sigma eta phi
  double See=0.;
  double Spp=0.;
  double Sep=0.;
  double deta(0),dphi(0);
  // compute (phi-corrected) covariance matrix:
  for(unsigned int i=0; i<nCry; i++) {
    if(isBarrel) {
      deta = etaDetId[i]-mid_eta;
      dphi = phiDetId[i]-mid_phi;
    } else {
      deta = etaDetId[i]-mid_eta;
      float hitLocalR2 = (xDetId[i]-mid_x)*(xDetId[i]-mid_x)+(yDetId[i]-mid_y)*(yDetId[i]-mid_y);
      float hitR2 = xDetId[i]*xDetId[i]+yDetId[i]*yDetId[i];
      float meanR2 = mid_x*mid_x+mid_y*mid_y;
      float hitR = sqrt(hitR2);
      float meanR = sqrt(meanR2);
      float phi = acos((hitR2+meanR2-hitLocalR2)/(2*hitR*meanR));
      dphi = hitR*phi;

    }
    See += (wiDetId[i]* deta * deta) / denominator;
    Spp += phiCorrectionFactor*(wiDetId[i]* dphi * dphi) / denominator;
    Sep += sqrt(phiCorrectionFactor)*(wiDetId[i]*deta*dphi) / denominator;
  }

  Cluster2ndMoments returnMoments;

  // compute matrix eigenvalues:
  returnMoments.sMaj = ((See + Spp) + sqrt((See - Spp)*(See - Spp) + 4.*Sep*Sep)) / 2.;
  returnMoments.sMin = ((See + Spp) - sqrt((See - Spp)*(See - Spp) + 4.*Sep*Sep)) / 2.;

  returnMoments.alpha = atan( (See - Spp + sqrt( (Spp - See)*(Spp - See) + 4.*Sep*Sep )) / (2.*Sep));

  return returnMoments;

}

//compute shower shapes: roundness and angle in a vector. Roundness is 0th element, Angle is 1st element.
//description: uses classical mechanics inertia tensor.
//             roundness is smaller_eValue/larger_eValue
//             angle is the angle from the iEta axis to the smallest eVector (a.k.a. the shower's elongated axis)
// this function uses only recHits belonging to a SC above energyThreshold (default 0)
// you can select linear weighting = energy_recHit/total_energy         (weightedPositionMethod=0) default
// or log weighting = max( 0.0, 4.2 + log(energy_recHit/total_energy) ) (weightedPositionMethod=1)
template<bool noZS>
std::vector<float> EcalClusterToolsT<noZS>::roundnessBarrelSuperClusters( const reco::SuperCluster &superCluster ,const EcalRecHitCollection &recHits, int weightedPositionMethod, float energyThreshold){//int positionWeightingMethod=0){
  std::vector<std::pair<const EcalRecHit*, float> > RH_ptrs_fracs;
    const std::vector< std::pair<DetId, float> >& myHitsPair = superCluster.hitsAndFractions();    
    for(unsigned int i=0; i< myHitsPair.size(); ++i){
        //get pointer to recHit object
        EcalRecHitCollection::const_iterator myRH = recHits.find(myHitsPair[i].first);
        if( myRH != recHits.end() && myRH->energy()*(noZS ? 1.0 : myHitsPair[i].second) > energyThreshold){ 
	  //require rec hit to have positive energy
	  RH_ptrs_fracs.push_back(  std::make_pair(&(*myRH) , myHitsPair[i].second)  );
        }
    }
    std::vector<float> temp = EcalClusterToolsT<noZS>::roundnessSelectedBarrelRecHits(RH_ptrs_fracs,weightedPositionMethod); 
    return temp;
}

// this function uses all recHits within specified window ( with default values ieta_delta=2, iphi_delta=5) around SC's highest recHit.
// recHits must pass an energy threshold "energyRHThresh" (default 0)
// you can select linear weighting = energy_recHit/total_energy         (weightedPositionMethod=0)
// or log weighting = max( 0.0, 4.2 + log(energy_recHit/total_energy) ) (weightedPositionMethod=1)
template<bool noZS>
std::vector<float> EcalClusterToolsT<noZS>::roundnessBarrelSuperClustersUserExtended( const reco::SuperCluster &superCluster ,const EcalRecHitCollection &recHits, int ieta_delta, int iphi_delta, float energyRHThresh, int weightedPositionMethod){

  std::vector<std::pair<const EcalRecHit*, float> > RH_ptrs_fracs;
    const std::vector< std::pair<DetId, float> >& myHitsPair = superCluster.hitsAndFractions();
    for(unsigned int i=0; i<myHitsPair.size(); ++i){
        //get pointer to recHit object
        EcalRecHitCollection::const_iterator myRH = recHits.find(myHitsPair[i].first);
        if(myRH != recHits.end() && myRH->energy()*(noZS ? 1.0 : myHitsPair[i].second) > energyRHThresh)
	  RH_ptrs_fracs.push_back(  std::make_pair(&(*myRH) , myHitsPair[i].second) );
    }


    std::vector<int> seedPosition = EcalClusterToolsT<noZS>::getSeedPosition(  RH_ptrs_fracs  );

    for(EcalRecHitCollection::const_iterator rh = recHits.begin(); rh != recHits.end(); rh++){
        EBDetId EBdetIdi( rh->detid() );
	float the_fraction = 0;
        //if(rh != recHits.end())
        bool inEtaWindow = (   abs(  deltaIEta(seedPosition[0],EBdetIdi.ieta())  ) <= ieta_delta   );
        bool inPhiWindow = (   abs(  deltaIPhi(seedPosition[1],EBdetIdi.iphi())  ) <= iphi_delta   );
        bool passEThresh = (  rh->energy() > energyRHThresh  );
        bool alreadyCounted = false;

        // figure out if the rechit considered now is already inside the SC
        bool is_SCrh_inside_recHits = false;
        for(unsigned int i=0; i<myHitsPair.size(); i++){
            EcalRecHitCollection::const_iterator SCrh = recHits.find(myHitsPair[i].first);
            if(SCrh != recHits.end()){
	      the_fraction = myHitsPair[i].second;
                is_SCrh_inside_recHits = true;
                if( rh->detid() == SCrh->detid()  ) alreadyCounted = true;
            }
        }//for loop over SC's recHits

        if( is_SCrh_inside_recHits && !alreadyCounted && passEThresh && inEtaWindow && inPhiWindow){
	  RH_ptrs_fracs.push_back( std::make_pair(&(*rh),the_fraction) );
        }

    }//for loop over rh
    return EcalClusterToolsT<noZS>::roundnessSelectedBarrelRecHits(RH_ptrs_fracs,weightedPositionMethod);
}

// this function computes the roundness and angle variables for vector of pointers to recHits you pass it
// you can select linear weighting = energy_recHit/total_energy         (weightedPositionMethod=0)
// or log weighting = max( 0.0, 4.2 + log(energy_recHit/total_energy) ) (weightedPositionMethod=1)
template<bool noZS>
std::vector<float> EcalClusterToolsT<noZS>::roundnessSelectedBarrelRecHits( const std::vector<std::pair<const EcalRecHit*,float> >& RH_ptrs_fracs, int weightedPositionMethod){//int weightedPositionMethod = 0){
    //positionWeightingMethod = 0 linear weighting, 1 log energy weighting

    std::vector<float> shapes; // this is the returning vector

    //make sure photon has more than one crystal; else roundness and angle suck
    if(RH_ptrs_fracs.size()<2){
        shapes.push_back( -3 );
        shapes.push_back( -3 );
        return shapes;
    }

    //Find highest E RH (Seed) and save info, compute sum total energy used
    std::vector<int> seedPosition = EcalClusterToolsT<noZS>::getSeedPosition(  RH_ptrs_fracs  );// *recHits);
    int tempInt = seedPosition[0];
    if(tempInt <0) tempInt++;
    float energyTotal = EcalClusterToolsT<noZS>::getSumEnergy(  RH_ptrs_fracs  );

    //1st loop over rechits: compute new weighted center position in coordinates centered on seed
    float centerIEta = 0.;
    float centerIPhi = 0.;
    float denominator = 0.;

    for(std::vector<std::pair<const EcalRecHit*,float> >::const_iterator rhf_ptr = RH_ptrs_fracs.begin(); rhf_ptr != RH_ptrs_fracs.end(); rhf_ptr++){
      const EcalRecHit* rh_ptr = rhf_ptr->first;
        //get iEta, iPhi
        EBDetId EBdetIdi( rh_ptr->detid() );
        if(fabs(energyTotal) < 0.0001){
            // don't /0, bad!
            shapes.push_back( -2 );
            shapes.push_back( -2 );
            return shapes;
        }
	float rh_energy = rh_ptr->energy() * (noZS ? 1.0 : rhf_ptr->second);
        float weight = 0;
        if(fabs(weightedPositionMethod)<0.0001){ //linear
            weight = rh_energy/energyTotal;
        }else{ //logrithmic
            weight = std::max(0.0, 4.2 + log(rh_energy/energyTotal));
        }
        denominator += weight;
        centerIEta += weight*deltaIEta(seedPosition[0],EBdetIdi.ieta()); 
        centerIPhi += weight*deltaIPhi(seedPosition[1],EBdetIdi.iphi());
    }
    if(fabs(denominator) < 0.0001){
        // don't /0, bad!
        shapes.push_back( -2 );
        shapes.push_back( -2 );
        return shapes;
    }
    centerIEta = centerIEta / denominator;
    centerIPhi = centerIPhi / denominator;


    //2nd loop over rechits: compute inertia tensor
    TMatrixDSym inertia(2); //initialize 2d inertia tensor
    double inertia00 = 0.;
    double inertia01 = 0.;// = inertia10 b/c matrix should be symmetric
    double inertia11 = 0.;
    int i = 0;
    for(std::vector<std::pair<const EcalRecHit*,float> >::const_iterator rhf_ptr = RH_ptrs_fracs.begin(); rhf_ptr != RH_ptrs_fracs.end(); rhf_ptr++){
      const EcalRecHit* rh_ptr = rhf_ptr->first;
        //get iEta, iPhi
        EBDetId EBdetIdi( rh_ptr->detid() );

        if(fabs(energyTotal) < 0.0001){
            // don't /0, bad!
            shapes.push_back( -2 );
            shapes.push_back( -2 );
            return shapes;
        }
	float rh_energy = rh_ptr->energy() * (noZS ? 1.0 : rhf_ptr->second);
        float weight = 0;
        if(fabs(weightedPositionMethod) < 0.0001){ //linear
            weight = rh_energy/energyTotal;
        }else{ //logrithmic
            weight = std::max(0.0, 4.2 + log(rh_energy/energyTotal));
        }

        float ieta_rh_to_center = deltaIEta(seedPosition[0],EBdetIdi.ieta()) - centerIEta;
        float iphi_rh_to_center = deltaIPhi(seedPosition[1],EBdetIdi.iphi()) - centerIPhi;

        inertia00 += weight*iphi_rh_to_center*iphi_rh_to_center;
        inertia01 -= weight*iphi_rh_to_center*ieta_rh_to_center;
        inertia11 += weight*ieta_rh_to_center*ieta_rh_to_center;
        i++;
    }

    inertia[0][0] = inertia00;
    inertia[0][1] = inertia01; // use same number here
    inertia[1][0] = inertia01; // and here to insure symmetry
    inertia[1][1] = inertia11;


    //step 1 find principal axes of inertia
    TMatrixD eVectors(2,2);
    TVectorD eValues(2);
    //std::cout<<"EcalClusterToolsT<noZS>::showerRoundness- about to compute eVectors"<<std::endl;
    eVectors=inertia.EigenVectors(eValues); //ordered highest eV to lowest eV (I checked!)
    //and eVectors are in columns of matrix! I checked!
    //and they are normalized to 1



    //step 2 select eta component of smaller eVal's eVector
    TVectorD smallerAxis(2);//easiest to spin SC on this axis (smallest eVal)
    smallerAxis[0]=eVectors[0][1];//row,col  //eta component
    smallerAxis[1]=eVectors[1][1];           //phi component

    //step 3 compute interesting quatities
    Double_t temp = fabs(smallerAxis[0]);// closer to 1 ->beamhalo, closer to 0 something else
    if(fabs(eValues[0]) < 0.0001){
        // don't /0, bad!
        shapes.push_back( -2 );
        shapes.push_back( -2 );
        return shapes;
    }

    float Roundness = eValues[1]/eValues[0];
    float Angle=acos(temp);

    if( -0.00001 < Roundness && Roundness < 0) Roundness = 0.;
    if( -0.00001 < Angle && Angle < 0 ) Angle = 0.;

    shapes.push_back( Roundness );
    shapes.push_back( Angle );
    return shapes;

}
//private functions useful for roundnessBarrelSuperClusters etc.
//compute delta iphi between a seed and a particular recHit
//iphi [1,360]
//safe gaurds are put in to ensure the difference is between [-180,180]
template<bool noZS>
int EcalClusterToolsT<noZS>::deltaIPhi(int seed_iphi, int rh_iphi){
    int rel_iphi = rh_iphi - seed_iphi;
    // take care of cyclic variable iphi [1,360]
    if(rel_iphi >  180) rel_iphi = rel_iphi - 360;
    if(rel_iphi < -180) rel_iphi = rel_iphi + 360;
    return rel_iphi;
}

//compute delta ieta between a seed and a particular recHit
//ieta [-85,-1] and [1,85]
//safe gaurds are put in to shift the negative ieta +1 to make an ieta=0 so differences are computed correctly
template<bool noZS>
int EcalClusterToolsT<noZS>::deltaIEta(int seed_ieta, int rh_ieta){
    // get rid of the fact that there is no ieta=0
    if(seed_ieta < 0) seed_ieta++;
    if(rh_ieta   < 0) rh_ieta++;
    int rel_ieta = rh_ieta - seed_ieta;
    return rel_ieta;
}

//return (ieta,iphi) of highest energy recHit of the recHits passed to this function
template<bool noZS>
std::vector<int> EcalClusterToolsT<noZS>::getSeedPosition(const std::vector<std::pair<const EcalRecHit*, float> >& RH_ptrs_fracs){
    std::vector<int> seedPosition;
    float eSeedRH = 0;
    int iEtaSeedRH = 0;
    int iPhiSeedRH = 0;

    for(std::vector<std::pair<const EcalRecHit*,float> >::const_iterator rhf_ptr = RH_ptrs_fracs.begin(); rhf_ptr != RH_ptrs_fracs.end(); rhf_ptr++){
      const EcalRecHit* rh_ptr = rhf_ptr->first;
        //get iEta, iPhi
        EBDetId EBdetIdi( rh_ptr->detid() );
	float rh_energy = rh_ptr->energy() * (noZS ? 1.0 : rhf_ptr->second);

        if(eSeedRH < rh_energy){
            eSeedRH = rh_energy;
            iEtaSeedRH = EBdetIdi.ieta();
            iPhiSeedRH = EBdetIdi.iphi();
        }

    }// for loop

    seedPosition.push_back(iEtaSeedRH);
    seedPosition.push_back(iPhiSeedRH);
    return seedPosition;
}

// return the total energy of the recHits passed to this function
template<bool noZS>
float EcalClusterToolsT<noZS>::getSumEnergy(const std::vector<std::pair<const EcalRecHit*, float> >& RH_ptrs_fracs){
    float sumE = 0.;
    for( const auto& hAndF : RH_ptrs_fracs ) {
      sumE += hAndF.first->energy() * (noZS ? 1.0 : hAndF.second);
    }    
    return sumE;
}

typedef EcalClusterToolsT<false> EcalClusterTools;

namespace noZS {
  typedef EcalClusterToolsT<true> EcalClusterTools;
}

#endif
