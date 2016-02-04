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
// $Id: testEcalClusterTools.cc,v 1.11 2010/01/04 15:08:33 ferriff Exp $
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
        for (reco::BasicClusterCollection::const_iterator it = ebClusters->begin(); it != ebClusters->end(); ++it ) {
                std::cout << "----- new cluster -----" << std::endl;
                std::cout << "----------------- size: " << (*it).size() << " energy: " << (*it).energy() << std::endl;

                std::cout << "e1x3..................... " << EcalClusterTools::e1x3( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e3x1..................... " << EcalClusterTools::e3x1( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e1x5..................... " << EcalClusterTools::e1x5( *it, ebRecHits, topology ) << std::endl;
                //std::cout << "e5x1..................... " << EcalClusterTools::e5x1( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e2x2..................... " << EcalClusterTools::e2x2( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e3x3..................... " << EcalClusterTools::e3x3( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e4x4..................... " << EcalClusterTools::e4x4( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e5x5..................... " << EcalClusterTools::e5x5( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e2x5Right................ " << EcalClusterTools::e2x5Right( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e2x5Left................. " << EcalClusterTools::e2x5Left( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e2x5Top.................. " << EcalClusterTools::e2x5Top( *it, ebRecHits, topology ) << std::endl;
                std::cout << "e2x5Bottom............... " << EcalClusterTools::e2x5Bottom( *it, ebRecHits, topology ) << std::endl;
		std::cout << "e2x5Max.................. " << EcalClusterTools::e2x5Max( *it, ebRecHits, topology ) << std::endl;
                std::cout << "eMax..................... " << EcalClusterTools::eMax( *it, ebRecHits ) << std::endl;
                std::cout << "e2nd..................... " << EcalClusterTools::e2nd( *it, ebRecHits ) << std::endl;
                std::vector<float> vEta = EcalClusterTools::energyBasketFractionEta( *it, ebRecHits );
                std::cout << "energyBasketFractionEta..";
                for (size_t i = 0; i < vEta.size(); ++i ) {
                        std::cout << " " << vEta[i];
                }
                std::cout << std::endl;
                std::vector<float> vPhi = EcalClusterTools::energyBasketFractionPhi( *it, ebRecHits );
                std::cout << "energyBasketFractionPhi..";
                for (size_t i = 0; i < vPhi.size(); ++i ) {
                        std::cout << " " << vPhi[i];
                }
                std::cout << std::endl;
                std::vector<float> vLat = EcalClusterTools::lat( *it, ebRecHits, geometry );
                std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
                std::vector<float> vCov = EcalClusterTools::covariances( *it, ebRecHits, topology, geometry );
                std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;
		std::vector<float> vLocCov = EcalClusterTools::localCovariances( *it, ebRecHits, topology );
                std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
                std::cout << "zernike20................ " << EcalClusterTools::zernike20( *it, ebRecHits, geometry ) << std::endl;
                std::cout << "zernike42................ " << EcalClusterTools::zernike42( *it, ebRecHits, geometry ) << std::endl;
        }

        std::cout << "========== ENDCAPS ==========" << std::endl;
        for (reco::BasicClusterCollection::const_iterator it = eeClusters->begin(); it != eeClusters->end(); ++it ) {
                std::cout << "----- new cluster -----" << std::endl;
                std::cout << "----------------- size: " << (*it).size() << " energy: " << (*it).energy() << std::endl;
                
                std::cout << "e1x3..................... " << EcalClusterTools::e1x3( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e3x1..................... " << EcalClusterTools::e3x1( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e1x5..................... " << EcalClusterTools::e1x5( *it, eeRecHits, topology ) << std::endl;
                //std::cout << "e5x1..................... " << EcalClusterTools::e5x1( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e2x2..................... " << EcalClusterTools::e2x2( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e3x3..................... " << EcalClusterTools::e3x3( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e4x4..................... " << EcalClusterTools::e4x4( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e5x5..................... " << EcalClusterTools::e5x5( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e2x5Right................ " << EcalClusterTools::e2x5Right( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e2x5Left................. " << EcalClusterTools::e2x5Left( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e2x5Top.................. " << EcalClusterTools::e2x5Top( *it, eeRecHits, topology ) << std::endl;
                std::cout << "e2x5Bottom............... " << EcalClusterTools::e2x5Bottom( *it, eeRecHits, topology ) << std::endl;
                std::cout << "eMax..................... " << EcalClusterTools::eMax( *it, eeRecHits ) << std::endl;
                std::cout << "e2nd..................... " << EcalClusterTools::e2nd( *it, eeRecHits ) << std::endl;
                std::vector<float> vLat = EcalClusterTools::lat( *it, eeRecHits, geometry );
                std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
                std::vector<float> vCov = EcalClusterTools::covariances( *it, eeRecHits, topology, geometry );
                std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;
		std::vector<float> vLocCov = EcalClusterTools::localCovariances( *it, eeRecHits, topology );
                std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
                std::cout << "zernike20................ " << EcalClusterTools::zernike20( *it, eeRecHits, geometry ) << std::endl;
                std::cout << "zernike42................ " << EcalClusterTools::zernike42( *it, eeRecHits, geometry ) << std::endl;
        }
}

//define this as a plug-in
DEFINE_FWK_MODULE(testEcalClusterTools);
