#ifndef RecoEcal_EgammaCoreTools_EcalClusterLazyTools_h
#define RecoEcal_EgammaCoreTools_EcalClusterLazyTools_h

/** \class EcalClusterLazyTools
 *  
 * various cluster tools (e.g. cluster shapes)
 *
 * \author Federico Ferri
 * 
 * \version $Id: 
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


class CaloTopology;
class CaloGeometry;

class EcalClusterLazyTools {
        public:
                EcalClusterLazyTools( const edm::Event &ev, const edm::EventSetup &es, edm::InputTag redEBRecHits, edm::InputTag redEERecHits );
                ~EcalClusterLazyTools();

                // various energies in the matrix nxn surrounding the maximum energy crystal of the input cluster
                float e1x3( const reco::BasicCluster &cluster );
                float e3x1( const reco::BasicCluster &cluster );
                float e1x5( const reco::BasicCluster &cluster );
                float e5x1( const reco::BasicCluster &cluster );
                float e2x2( const reco::BasicCluster &cluster );
                float e3x2( const reco::BasicCluster &cluster );
                float e3x3( const reco::BasicCluster &cluster );
                float e4x4( const reco::BasicCluster &cluster );
                float e5x5( const reco::BasicCluster &cluster );
                // energy in the 2x5 strip right of the max crystal (does not contain max crystal)
                // 2 crystals wide in eta, 5 wide in phi.
                float e2x5Right( const reco::BasicCluster &cluster );
                // energy in the 2x5 strip left of the max crystal (does not contain max crystal)
                float e2x5Left( const reco::BasicCluster &cluster );
                // energy in the 5x2 strip above the max crystal (does not contain max crystal)
                // 5 crystals wide in eta, 2 wide in phi.
                float e2x5Top( const reco::BasicCluster &cluster );
                // energy in the 5x2 strip below the max crystal (does not contain max crystal)
                float e2x5Bottom( const reco::BasicCluster &cluster );
                // energy in a 2x5 strip containing the seed (max) crystal.
                // 2 crystals wide in eta, 5 wide in phi.
                // it is the maximum of either (1x5left + 1x5center) or (1x5right + 1x5center)
                float e2x5Max( const reco::BasicCluster &cluster );
                // energies in the crystal left, right, top, bottom w.r.t. to the most energetic crystal
                float eLeft( const reco::BasicCluster &cluster );
                float eRight( const reco::BasicCluster &cluster );
                float eTop( const reco::BasicCluster &cluster );
                float eBottom( const reco::BasicCluster &cluster );
                // the energy of the most energetic crystal in the cluster
                float eMax( const reco::BasicCluster &cluster );
                // the energy of the second most energetic crystal in the cluster
                float e2nd( const reco::BasicCluster &cluster );
                // get the DetId and the energy of the maximum energy crystal of the input cluster
                std::pair<DetId, float> getMaximum( const reco::BasicCluster &cluster );
                std::vector<float> energyBasketFractionEta( const reco::BasicCluster &cluster );
                std::vector<float> energyBasketFractionPhi( const reco::BasicCluster &cluster );
                // return a vector v with v[0] = etaLat, v[1] = phiLat, v[2] = lat
                std::vector<float> lat( const reco::BasicCluster &cluster, bool logW = true, float w0 = 4.7 );
                // return a vector v with v[0] = covEtaEta, v[1] = covEtaPhi, v[2] = covPhiPhi
                std::vector<float> covariances(const reco::BasicCluster &cluster, float w0 = 4.7 );
                // return a vector v with v[0] = covIEtaIEta, v[1] = covIEtaIPhi, v[2] = covIPhiIPhi
                //this function calculates differences in eta/phi in units of crystals not global eta/phi
                //this is gives better performance in the crack regions of the calorimeter but gives otherwise identical results to covariances function
                //this is only defined for the barrel, it returns covariances when the cluster is in the endcap
                //Warning: covIEtaIEta has been studied by egamma, but so far covIPhiIPhi hasnt been studied extensively so there could be a bug in 
                //         the covIPhiIEta or covIPhiIPhi calculations. I dont think there is but as it hasnt been heavily used, there might be one
                std::vector<float> localCovariances(const reco::BasicCluster &cluster, float w0 = 4.7);
                double zernike20( const reco::BasicCluster &cluster, double R0 = 6.6, bool logW = true, float w0 = 4.7 );
                double zernike42( const reco::BasicCluster &cluster, double R0 = 6.6, bool logW = true, float w0 = 4.7 );

                // get the detId's of a matrix centered in the maximum energy crystal = (0,0)
                // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
                std::vector<DetId> matrixDetId( DetId id, int ixMin, int ixMax, int iyMin, int iyMax );
                // get the energy deposited in a matrix centered in the maximum energy crystal = (0,0)
                // the size is specified by ixMin, ixMax, iyMin, iyMax in unit of crystals
                float matrixEnergy( const reco::BasicCluster &cluster, DetId id, int ixMin, int ixMax, int iyMin, int iyMax );

        private:
                void getGeometry( const edm::EventSetup &es );
                void getTopology( const edm::EventSetup &es );
                void getEBRecHits( const edm::Event &ev, edm::InputTag redEBRecHits );
                void getEERecHits( const edm::Event &ev, edm::InputTag redEERecHits );
                const EcalRecHitCollection * getEcalRecHitCollection( const reco::BasicCluster &cluster );

                const CaloGeometry *geometry_;
                const CaloTopology *topology_;
                const EcalRecHitCollection *ebRecHits_;
                const EcalRecHitCollection *eeRecHits_;
};

#endif
