// -*- C++ -*-
//
// Package:    testEcalClusterTools
// Class:      testEcalClusterTools
// 
/**\class testEcalClusterTools testEcalClusterTools.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  "Federico Ferri federi
//         Created:  Mon Apr  7 14:11:00 CEST 2008
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// to access recHits and BasicClusters
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

// to use the cluster tools
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"



class testEcalClusterTools : public edm::EDAnalyzer {
        public:
                explicit testEcalClusterTools(const edm::ParameterSet&);
                ~testEcalClusterTools();

                edm::InputTag barrelClusterCollection_;
                edm::InputTag endcapClusterCollection_;
                edm::InputTag reducedBarrelRecHitCollection_;
                edm::InputTag reducedEndcapRecHitCollection_;

        private:
                virtual void analyze(const edm::Event&, const edm::EventSetup&);
};



testEcalClusterTools::testEcalClusterTools(const edm::ParameterSet& ps)
{
        barrelClusterCollection_ = ps.getParameter<edm::InputTag>("barrelClusterCollection");
        endcapClusterCollection_ = ps.getParameter<edm::InputTag>("endcapClusterCollection");
        reducedBarrelRecHitCollection_ = ps.getParameter<edm::InputTag>("reducedBarrelRecHitCollection");
        reducedEndcapRecHitCollection_ = ps.getParameter<edm::InputTag>("reducedEndcapRecHitCollection");
}



testEcalClusterTools::~testEcalClusterTools()
{
}



void testEcalClusterTools::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
        edm::Handle< reco::BasicClusterCollection > pEBClusters;
        ev.getByLabel( barrelClusterCollection_, pEBClusters );
        const reco::BasicClusterCollection *ebClusters = pEBClusters.product();

        edm::Handle< reco::BasicClusterCollection > pEEClusters;
        ev.getByLabel( endcapClusterCollection_, pEEClusters );
        const reco::BasicClusterCollection *eeClusters = pEEClusters.product();

        edm::Handle< EcalRecHitCollection > pEBRecHits;
        ev.getByLabel( reducedBarrelRecHitCollection_, pEBRecHits );
        const EcalRecHitCollection *ebRecHits = pEBRecHits.product();

        edm::Handle< EcalRecHitCollection > pEERecHits;
        ev.getByLabel( reducedEndcapRecHitCollection_, pEERecHits );
        const EcalRecHitCollection *eeRecHits = pEERecHits.product();

        edm::ESHandle<CaloGeometry> pGeometry;
        es.get<CaloGeometryRecord>().get(pGeometry);
        const CaloGeometry *geometry = pGeometry.product();

        edm::ESHandle<CaloTopology> pTopology;
        es.get<CaloTopologyRecord>().get(pTopology);
        const CaloTopology *topology = pTopology.product();

        std::cout << "========== BARREL ==========" << std::endl;
        for (const auto & ebCluster : *ebClusters) {
                std::cout << "----- new cluster -----" << std::endl;
                std::cout << "----------------- size: " << ebCluster.size() << " energy: " << ebCluster.energy() << std::endl;

                std::cout << "e1x3..................... " << EcalClusterTools::e1x3( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "e3x1..................... " << EcalClusterTools::e3x1( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "e1x5..................... " << EcalClusterTools::e1x5( ebCluster, ebRecHits, topology ) << std::endl;
                //std::cout << "e5x1..................... " << EcalClusterTools::e5x1( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e2x2..................... " << EcalClusterTools::e2x2( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "e3x3..................... " << EcalClusterTools::e3x3( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "e4x4..................... " << EcalClusterTools::e4x4( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "e5x5..................... " << EcalClusterTools::e5x5( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "e2x5Right................ " << EcalClusterTools::e2x5Right( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "e2x5Left................. " << EcalClusterTools::e2x5Left( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "e2x5Top.................. " << EcalClusterTools::e2x5Top( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "e2x5Bottom............... " << EcalClusterTools::e2x5Bottom( ebCluster, ebRecHits, topology ) << std::endl;
		std::cout << "e2x5Max.................. " << EcalClusterTools::e2x5Max( ebCluster, ebRecHits, topology ) << std::endl;
                std::cout << "eMax..................... " << EcalClusterTools::eMax( ebCluster, ebRecHits ) << std::endl;
                std::cout << "e2nd..................... " << EcalClusterTools::e2nd( ebCluster, ebRecHits ) << std::endl;
                std::vector<float> vEta = EcalClusterTools::energyBasketFractionEta( ebCluster, ebRecHits );
                std::cout << "energyBasketFractionEta..";
                for (float i : vEta) {
                        std::cout << " " << i;
                }
                std::cout << std::endl;
                std::vector<float> vPhi = EcalClusterTools::energyBasketFractionPhi( ebCluster, ebRecHits );
                std::cout << "energyBasketFractionPhi..";
                for (float i : vPhi) {
                        std::cout << " " << i;
                }
                std::cout << std::endl;
                std::vector<float> vLat = EcalClusterTools::lat( ebCluster, ebRecHits, geometry );
                std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
                std::vector<float> vCov = EcalClusterTools::covariances( ebCluster, ebRecHits, topology, geometry );
                std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;
		std::vector<float> vLocCov = EcalClusterTools::localCovariances( ebCluster, ebRecHits, topology );
                std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
                std::cout << "zernike20................ " << EcalClusterTools::zernike20( ebCluster, ebRecHits, geometry ) << std::endl;
                std::cout << "zernike42................ " << EcalClusterTools::zernike42( ebCluster, ebRecHits, geometry ) << std::endl;
        }

        std::cout << "========== ENDCAPS ==========" << std::endl;
        for (const auto & eeCluster : *eeClusters) {
                std::cout << "----- new cluster -----" << std::endl;
                std::cout << "----------------- size: " << eeCluster.size() << " energy: " << eeCluster.energy() << std::endl;
                
                std::cout << "e1x3..................... " << EcalClusterTools::e1x3( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "e3x1..................... " << EcalClusterTools::e3x1( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "e1x5..................... " << EcalClusterTools::e1x5( eeCluster, eeRecHits, topology ) << std::endl;
                //std::cout << "e5x1..................... " << EcalClusterTools::e5x1( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e2x2..................... " << EcalClusterTools::e2x2( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "e3x3..................... " << EcalClusterTools::e3x3( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "e4x4..................... " << EcalClusterTools::e4x4( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "e5x5..................... " << EcalClusterTools::e5x5( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "e2x5Right................ " << EcalClusterTools::e2x5Right( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "e2x5Left................. " << EcalClusterTools::e2x5Left( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "e2x5Top.................. " << EcalClusterTools::e2x5Top( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "e2x5Bottom............... " << EcalClusterTools::e2x5Bottom( eeCluster, eeRecHits, topology ) << std::endl;
                std::cout << "eMax..................... " << EcalClusterTools::eMax( eeCluster, eeRecHits ) << std::endl;
                std::cout << "e2nd..................... " << EcalClusterTools::e2nd( eeCluster, eeRecHits ) << std::endl;
                std::vector<float> vLat = EcalClusterTools::lat( eeCluster, eeRecHits, geometry );
                std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
                std::vector<float> vCov = EcalClusterTools::covariances( eeCluster, eeRecHits, topology, geometry );
                std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;
		std::vector<float> vLocCov = EcalClusterTools::localCovariances( eeCluster, eeRecHits, topology );
                std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
                std::cout << "zernike20................ " << EcalClusterTools::zernike20( eeCluster, eeRecHits, geometry ) << std::endl;
                std::cout << "zernike42................ " << EcalClusterTools::zernike42( eeCluster, eeRecHits, geometry ) << std::endl;
        }
}

//define this as a plug-in
DEFINE_FWK_MODULE(testEcalClusterTools);
