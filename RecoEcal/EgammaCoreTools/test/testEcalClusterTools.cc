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

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <memory>

class testEcalClusterTools : public edm::one::EDAnalyzer<> {
public:
  explicit testEcalClusterTools(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  using ClusterTools = noZS::EcalClusterTools;  // alternatively just EcalClusterTools

  const edm::EDGetToken barrelClusterToken_;
  const edm::EDGetToken endcapClusterToken_;
  const edm::EDGetToken barrelRecHitToken_;
  const edm::EDGetToken endcapRecHitToken_;
};

testEcalClusterTools::testEcalClusterTools(const edm::ParameterSet& ps)
    : barrelClusterToken_(
          consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("barrelClusterCollection"))),
      endcapClusterToken_(
          consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("endcapClusterCollection"))),
      barrelRecHitToken_(consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("barrelRecHitCollection"))),
      endcapRecHitToken_(consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("endcapRecHitCollection"))) {}

void testEcalClusterTools::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  edm::Handle<EcalRecHitCollection> pEBRecHits;
  ev.getByToken(barrelRecHitToken_, pEBRecHits);
  const EcalRecHitCollection* ebRecHits = pEBRecHits.product();

  edm::Handle<EcalRecHitCollection> pEERecHits;
  ev.getByToken(endcapRecHitToken_, pEERecHits);
  const EcalRecHitCollection* eeRecHits = pEERecHits.product();

  edm::Handle<reco::BasicClusterCollection> pEBClusters;
  ev.getByToken(barrelClusterToken_, pEBClusters);
  const reco::BasicClusterCollection* ebClusters = pEBClusters.product();

  edm::Handle<reco::BasicClusterCollection> pEEClusters;
  ev.getByToken(endcapClusterToken_, pEEClusters);
  const reco::BasicClusterCollection* eeClusters = pEEClusters.product();

  edm::ESHandle<CaloGeometry> pGeometry;
  es.get<CaloGeometryRecord>().get(pGeometry);
  const CaloGeometry* geometry = pGeometry.product();

  edm::ESHandle<CaloTopology> pTopology;
  es.get<CaloTopologyRecord>().get(pTopology);
  const CaloTopology* topology = pTopology.product();

  std::cout << "========== BARREL ==========" << std::endl;
  for (auto const& clus : *ebClusters) {
    DetId maxId = ClusterTools::getMaximum(clus, eeRecHits).first;

    std::cout << "----- new cluster -----" << std::endl;
    std::cout << "----------------- size: " << clus.size() << " energy: " << clus.energy() << std::endl;

    std::cout << "e1x3..................... " << ClusterTools::e1x3(clus, ebRecHits, topology) << std::endl;
    std::cout << "e3x1..................... " << ClusterTools::e3x1(clus, ebRecHits, topology) << std::endl;
    std::cout << "e1x5..................... " << ClusterTools::e1x5(clus, ebRecHits, topology) << std::endl;
    std::cout << "e5x1..................... " << ClusterTools::e5x1(clus, ebRecHits, topology) << std::endl;
    std::cout << "e2x2..................... " << ClusterTools::e2x2(clus, ebRecHits, topology) << std::endl;
    std::cout << "e3x3..................... " << ClusterTools::e3x3(clus, ebRecHits, topology) << std::endl;
    std::cout << "e4x4..................... " << ClusterTools::e4x4(clus, ebRecHits, topology) << std::endl;
    std::cout << "e5x5..................... " << ClusterTools::e5x5(clus, ebRecHits, topology) << std::endl;
    std::cout << "n5x5..................... " << ClusterTools::n5x5(clus, eeRecHits, topology) << std::endl;
    std::cout << "e2x5Right................ " << ClusterTools::e2x5Right(clus, ebRecHits, topology) << std::endl;
    std::cout << "e2x5Left................. " << ClusterTools::e2x5Left(clus, ebRecHits, topology) << std::endl;
    std::cout << "e2x5Top.................. " << ClusterTools::e2x5Top(clus, ebRecHits, topology) << std::endl;
    std::cout << "e2x5Bottom............... " << ClusterTools::e2x5Bottom(clus, ebRecHits, topology) << std::endl;
    std::cout << "e2x5Max.................. " << ClusterTools::e2x5Max(clus, ebRecHits, topology) << std::endl;
    std::cout << "eMax..................... " << ClusterTools::eMax(clus, ebRecHits) << std::endl;
    std::cout << "e2nd..................... " << ClusterTools::e2nd(clus, ebRecHits) << std::endl;
    std::vector<float> vEta = ClusterTools::energyBasketFractionEta(clus, ebRecHits);
    std::cout << "energyBasketFractionEta..";
    for (auto const& eta : vEta)
      std::cout << " " << eta;
    std::cout << std::endl;
    std::vector<float> vPhi = ClusterTools::energyBasketFractionPhi(clus, ebRecHits);
    std::cout << "energyBasketFractionPhi..";
    for (auto const& phi : vPhi)
      std::cout << " " << phi;
    std::cout << std::endl;
    std::vector<float> vLat = ClusterTools::lat(clus, ebRecHits, geometry);
    std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
    std::vector<float> vCov = ClusterTools::covariances(clus, ebRecHits, topology, geometry);
    std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;
    std::vector<float> vLocCov = ClusterTools::localCovariances(clus, ebRecHits, topology);
    std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
    std::cout << "zernike20................ " << ClusterTools::zernike20(clus, ebRecHits, geometry) << std::endl;
    std::cout << "zernike42................ " << ClusterTools::zernike42(clus, ebRecHits, geometry) << std::endl;
    std::cout << "nrSaturatedCrysIn5x5..... " << ClusterTools::nrSaturatedCrysIn5x5(maxId, ebRecHits, topology)
              << std::endl;
  }

  std::cout << "========== ENDCAPS ==========" << std::endl;
  for (auto const& clus : *eeClusters) {
    DetId maxId = ClusterTools::getMaximum(clus, eeRecHits).first;

    std::cout << "----- new cluster -----" << std::endl;
    std::cout << "----------------- size: " << clus.size() << " energy: " << clus.energy() << std::endl;

    std::cout << "e1x3..................... " << ClusterTools::e1x3(clus, eeRecHits, topology) << std::endl;
    std::cout << "e3x1..................... " << ClusterTools::e3x1(clus, eeRecHits, topology) << std::endl;
    std::cout << "e1x5..................... " << ClusterTools::e1x5(clus, eeRecHits, topology) << std::endl;
    std::cout << "e5x1..................... " << ClusterTools::e5x1(clus, eeRecHits, topology) << std::endl;
    std::cout << "e2x2..................... " << ClusterTools::e2x2(clus, eeRecHits, topology) << std::endl;
    std::cout << "e3x3..................... " << ClusterTools::e3x3(clus, eeRecHits, topology) << std::endl;
    std::cout << "e4x4..................... " << ClusterTools::e4x4(clus, eeRecHits, topology) << std::endl;
    std::cout << "e5x5..................... " << ClusterTools::e5x5(clus, eeRecHits, topology) << std::endl;
    std::cout << "n5x5..................... " << ClusterTools::n5x5(clus, eeRecHits, topology) << std::endl;
    std::cout << "e2x5Right................ " << ClusterTools::e2x5Right(clus, eeRecHits, topology) << std::endl;
    std::cout << "e2x5Left................. " << ClusterTools::e2x5Left(clus, eeRecHits, topology) << std::endl;
    std::cout << "e2x5Top.................. " << ClusterTools::e2x5Top(clus, eeRecHits, topology) << std::endl;
    std::cout << "e2x5Bottom............... " << ClusterTools::e2x5Bottom(clus, eeRecHits, topology) << std::endl;
    std::cout << "eMax..................... " << ClusterTools::eMax(clus, eeRecHits) << std::endl;
    std::cout << "e2nd..................... " << ClusterTools::e2nd(clus, eeRecHits) << std::endl;
    std::vector<float> vLat = ClusterTools::lat(clus, eeRecHits, geometry);
    std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
    std::vector<float> vCov = ClusterTools::covariances(clus, eeRecHits, topology, geometry);
    std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;
    std::vector<float> vLocCov = ClusterTools::localCovariances(clus, eeRecHits, topology);
    std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
    std::cout << "zernike20................ " << ClusterTools::zernike20(clus, eeRecHits, geometry) << std::endl;
    std::cout << "zernike42................ " << ClusterTools::zernike42(clus, eeRecHits, geometry) << std::endl;
    std::cout << "nrSaturatedCrysIn5x5..... " << ClusterTools::nrSaturatedCrysIn5x5(maxId, eeRecHits, topology)
              << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(testEcalClusterTools);
