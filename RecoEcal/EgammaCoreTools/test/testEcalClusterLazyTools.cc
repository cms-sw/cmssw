// -*- C++ -*-
//
// Package:    testEcalClusterLazyTools
// Class:      testEcalClusterLazyTools
//
/**\class testEcalClusterLazyTools testEcalClusterLazyTools.cc

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
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include <memory>

class testEcalClusterLazyTools : public edm::one::EDAnalyzer<> {
public:
  explicit testEcalClusterLazyTools(const edm::ParameterSet&);

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  using LazyTools = noZS::EcalClusterLazyTools;  // alternatively just EcalClusterLazyTools

  const edm::EDGetTokenT<reco::BasicClusterCollection> barrelClusterToken_;
  const edm::EDGetTokenT<reco::BasicClusterCollection> endcapClusterToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> barrelRecHitToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> endcapRecHitToken_;

  LazyTools::ESGetTokens esGetTokens_;
};

testEcalClusterLazyTools::testEcalClusterLazyTools(const edm::ParameterSet& ps)
    : barrelClusterToken_(consumes(ps.getParameter<edm::InputTag>("barrelClusterCollection"))),
      endcapClusterToken_(consumes(ps.getParameter<edm::InputTag>("endcapClusterCollection"))),
      barrelRecHitToken_(consumes(ps.getParameter<edm::InputTag>("barrelRecHitCollection"))),
      endcapRecHitToken_(consumes(ps.getParameter<edm::InputTag>("endcapRecHitCollection"))),
      esGetTokens_{consumesCollector()} {}

void testEcalClusterLazyTools::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  edm::Handle<reco::BasicClusterCollection> pEBClusters;
  ev.getByToken(barrelClusterToken_, pEBClusters);
  const reco::BasicClusterCollection* ebClusters = pEBClusters.product();

  edm::Handle<reco::BasicClusterCollection> pEEClusters;
  ev.getByToken(endcapClusterToken_, pEEClusters);
  const reco::BasicClusterCollection* eeClusters = pEEClusters.product();

  LazyTools lazyTools(ev, esGetTokens_.get(es), barrelRecHitToken_, endcapRecHitToken_);

  std::cout << "========== BARREL ==========" << std::endl;
  for (auto const& clus : *ebClusters) {
    std::cout << "----- new cluster -----" << std::endl;
    std::cout << "----------------- size: " << (clus).size() << " energy: " << (clus).energy() << std::endl;

    std::cout << "e1x3..................... " << lazyTools.e1x3(clus) << std::endl;
    std::cout << "e3x1..................... " << lazyTools.e3x1(clus) << std::endl;
    std::cout << "e1x5..................... " << lazyTools.e1x5(clus) << std::endl;
    std::cout << "e5x1..................... " << lazyTools.e5x1(clus) << std::endl;
    std::cout << "e2x2..................... " << lazyTools.e2x2(clus) << std::endl;
    std::cout << "e3x3..................... " << lazyTools.e3x3(clus) << std::endl;
    std::cout << "e4x4..................... " << lazyTools.e4x4(clus) << std::endl;
    std::cout << "e5x5..................... " << lazyTools.e5x5(clus) << std::endl;
    std::cout << "n5x5..................... " << lazyTools.n5x5(clus) << std::endl;
    std::cout << "e2x5Right................ " << lazyTools.e2x5Right(clus) << std::endl;
    std::cout << "e2x5Left................. " << lazyTools.e2x5Left(clus) << std::endl;
    std::cout << "e2x5Top.................. " << lazyTools.e2x5Top(clus) << std::endl;
    std::cout << "e2x5Bottom............... " << lazyTools.e2x5Bottom(clus) << std::endl;
    std::cout << "e2x5Max.................. " << lazyTools.e2x5Max(clus) << std::endl;
    std::cout << "eMax..................... " << lazyTools.eMax(clus) << std::endl;
    std::cout << "e2nd..................... " << lazyTools.e2nd(clus) << std::endl;
    std::vector<float> vEta = lazyTools.energyBasketFractionEta(clus);
    std::cout << "energyBasketFractionEta..";
    for (size_t i = 0; i < vEta.size(); ++i) {
      std::cout << " " << vEta[i];
    }
    std::cout << std::endl;
    std::vector<float> vPhi = lazyTools.energyBasketFractionPhi(clus);
    std::cout << "energyBasketFractionPhi..";
    for (size_t i = 0; i < vPhi.size(); ++i) {
      std::cout << " " << vPhi[i];
    }
    std::cout << std::endl;
    std::vector<float> vLat = lazyTools.lat(clus);
    std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
    std::vector<float> vCov = lazyTools.covariances(clus);
    std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;
    std::vector<float> vLocCov = lazyTools.localCovariances(clus);
    std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
    std::cout << "zernike20................ " << lazyTools.zernike20(clus) << std::endl;
    std::cout << "zernike42................ " << lazyTools.zernike42(clus) << std::endl;
  }

  std::cout << "========== ENDCAPS ==========" << std::endl;
  for (auto const& clus : *eeClusters) {
    std::cout << "----- new cluster -----" << std::endl;
    std::cout << "----------------- size: " << (clus).size() << " energy: " << (clus).energy() << std::endl;

    std::cout << "e1x3..................... " << lazyTools.e1x3(clus) << std::endl;
    std::cout << "e3x1..................... " << lazyTools.e3x1(clus) << std::endl;
    std::cout << "e1x5..................... " << lazyTools.e1x5(clus) << std::endl;
    std::cout << "e5x1..................... " << lazyTools.e5x1(clus) << std::endl;
    std::cout << "e2x2..................... " << lazyTools.e2x2(clus) << std::endl;
    std::cout << "e3x3..................... " << lazyTools.e3x3(clus) << std::endl;
    std::cout << "e4x4..................... " << lazyTools.e4x4(clus) << std::endl;
    std::cout << "e5x5..................... " << lazyTools.e5x5(clus) << std::endl;
    std::cout << "n5x5..................... " << lazyTools.n5x5(clus) << std::endl;
    std::cout << "e2x5Right................ " << lazyTools.e2x5Right(clus) << std::endl;
    std::cout << "e2x5Left................. " << lazyTools.e2x5Left(clus) << std::endl;
    std::cout << "e2x5Top.................. " << lazyTools.e2x5Top(clus) << std::endl;
    std::cout << "e2x5Bottom............... " << lazyTools.e2x5Bottom(clus) << std::endl;
    std::cout << "eMax..................... " << lazyTools.eMax(clus) << std::endl;
    std::cout << "e2nd..................... " << lazyTools.e2nd(clus) << std::endl;
    std::vector<float> vLat = lazyTools.lat(clus);
    std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
    std::vector<float> vCov = lazyTools.covariances(clus);
    std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;
    std::vector<float> vLocCov = lazyTools.localCovariances(clus);
    std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
    std::cout << "zernike20................ " << lazyTools.zernike20(clus) << std::endl;
    std::cout << "zernike42................ " << lazyTools.zernike42(clus) << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(testEcalClusterLazyTools);
