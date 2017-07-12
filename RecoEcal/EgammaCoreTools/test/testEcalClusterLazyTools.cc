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

//#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"



class testEcalClusterLazyTools : public edm::EDAnalyzer {
public:
  explicit testEcalClusterLazyTools(const edm::ParameterSet&);
  ~testEcalClusterLazyTools();
  
  edm::EDGetTokenT<reco::BasicClusterCollection> barrelClusterToken_;
  edm::EDGetTokenT<reco::BasicClusterCollection> endcapClusterToken_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitToken_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedEndcapRecHitToken_;
  
private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
};



testEcalClusterLazyTools::testEcalClusterLazyTools(const edm::ParameterSet& ps)
{
  barrelClusterToken_ =       consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("barrelClusterCollection"));
  endcapClusterToken_ =       consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("endcapClusterCollection"));
  reducedBarrelRecHitToken_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("reducedBarrelRecHitCollection"));
  reducedEndcapRecHitToken_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("reducedEndcapRecHitCollection"));
}



testEcalClusterLazyTools::~testEcalClusterLazyTools()
{}



void testEcalClusterLazyTools::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle< reco::BasicClusterCollection > pEBClusters;
  ev.getByToken( barrelClusterToken_, pEBClusters );
  const reco::BasicClusterCollection *ebClusters = pEBClusters.product();
  
  edm::Handle< reco::BasicClusterCollection > pEEClusters;
  ev.getByToken( endcapClusterToken_, pEEClusters );
  const reco::BasicClusterCollection *eeClusters = pEEClusters.product();

  EcalClusterLazyTools lazyTools( ev, es, reducedBarrelRecHitToken_, reducedEndcapRecHitToken_ );
  
  std::cout << "========== BARREL ==========" << std::endl;
  for (const auto & ebCluster : *ebClusters) {
    std::cout << "----- new cluster -----" << std::endl;
    std::cout << "----------------- size: " << ebCluster.size() << " energy: " << ebCluster.energy() << std::endl;
    
    std::cout << "e1x3..................... " << lazyTools.e1x3( ebCluster ) << std::endl;
    std::cout << "e3x1..................... " << lazyTools.e3x1( ebCluster ) << std::endl;
    std::cout << "e1x5..................... " << lazyTools.e1x5( ebCluster ) << std::endl;
    //std::cout << "e5x1..................... " << lazyTools.e5x1( *it ) << std::endl;
    std::cout << "e2x2..................... " << lazyTools.e2x2( ebCluster ) << std::endl;
    std::cout << "e3x3..................... " << lazyTools.e5x5( ebCluster ) << std::endl;
    std::cout << "e4x4..................... " << lazyTools.e4x4( ebCluster ) << std::endl;
    std::cout << "e5x5..................... " << lazyTools.e3x3( ebCluster ) << std::endl;
    std::cout << "e2x5Right................ " << lazyTools.e2x5Right( ebCluster ) << std::endl;
    std::cout << "e2x5Left................. " << lazyTools.e2x5Left( ebCluster ) << std::endl;
    std::cout << "e2x5Top.................. " << lazyTools.e2x5Top( ebCluster ) << std::endl;
    std::cout << "e2x5Bottom............... " << lazyTools.e2x5Bottom( ebCluster ) << std::endl;
    std::cout << "e2x5Max.................. " << lazyTools.e2x5Max( ebCluster ) << std::endl;
    std::cout << "eMax..................... " << lazyTools.eMax( ebCluster ) << std::endl;
    std::cout << "e2nd..................... " << lazyTools.e2nd( ebCluster ) << std::endl;
    std::vector<float> vEta = lazyTools.energyBasketFractionEta( ebCluster );
    std::cout << "energyBasketFractionEta..";
    for (float i : vEta) {
      std::cout << " " << i;
    }
    std::cout << std::endl;
    std::vector<float> vPhi = lazyTools.energyBasketFractionPhi( ebCluster );
    std::cout << "energyBasketFractionPhi..";
    for (float i : vPhi) {
      std::cout << " " << i;
    }
    std::cout << std::endl;
    std::vector<float> vLat = lazyTools.lat( ebCluster );
    std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
    std::vector<float> vCov = lazyTools.covariances( ebCluster );
    std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;  
    std::vector<float> vLocCov = lazyTools.localCovariances( ebCluster );
    std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
    std::cout << "zernike20................ " << lazyTools.zernike20( ebCluster ) << std::endl;
    std::cout << "zernike42................ " << lazyTools.zernike42( ebCluster ) << std::endl;
  }
  
  std::cout << "========== ENDCAPS ==========" << std::endl;
  for (const auto & eeCluster : *eeClusters) {
    std::cout << "----- new cluster -----" << std::endl;
    std::cout << "----------------- size: " << eeCluster.size() << " energy: " << eeCluster.energy() << std::endl;
    
    std::cout << "e1x3..................... " << lazyTools.e1x3( eeCluster ) << std::endl;
    std::cout << "e3x1..................... " << lazyTools.e3x1( eeCluster ) << std::endl;
    std::cout << "e1x5..................... " << lazyTools.e1x5( eeCluster ) << std::endl;
    //std::cout << "e5x1..................... " << lazyTools.e5x1( *it ) << std::endl;
    std::cout << "e2x2..................... " << lazyTools.e2x2( eeCluster ) << std::endl;
    std::cout << "e3x3..................... " << lazyTools.e5x5( eeCluster ) << std::endl;
    std::cout << "e4x4..................... " << lazyTools.e4x4( eeCluster ) << std::endl;
    std::cout << "e5x5..................... " << lazyTools.e3x3( eeCluster ) << std::endl;
    std::cout << "e2x5Right................ " << lazyTools.e2x5Right( eeCluster ) << std::endl;
    std::cout << "e2x5Left................. " << lazyTools.e2x5Left( eeCluster ) << std::endl;
    std::cout << "e2x5Top.................. " << lazyTools.e2x5Top( eeCluster ) << std::endl;
    std::cout << "e2x5Bottom............... " << lazyTools.e2x5Bottom( eeCluster ) << std::endl;
    std::cout << "eMax..................... " << lazyTools.eMax( eeCluster ) << std::endl;
    std::cout << "e2nd..................... " << lazyTools.e2nd( eeCluster ) << std::endl;
    std::vector<float> vLat = lazyTools.lat( eeCluster );
    std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
    std::vector<float> vCov = lazyTools.covariances( eeCluster );
    std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl; 
    std::vector<float> vLocCov = lazyTools.localCovariances( eeCluster );
    std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
    std::cout << "zernike20................ " << lazyTools.zernike20( eeCluster ) << std::endl;
    std::cout << "zernike42................ " << lazyTools.zernike42( eeCluster ) << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(testEcalClusterLazyTools);
