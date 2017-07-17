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
  for (reco::BasicClusterCollection::const_iterator it = ebClusters->begin(); it != ebClusters->end(); ++it ) {
    std::cout << "----- new cluster -----" << std::endl;
    std::cout << "----------------- size: " << (*it).size() << " energy: " << (*it).energy() << std::endl;
    
    std::cout << "e1x3..................... " << lazyTools.e1x3( *it ) << std::endl;
    std::cout << "e3x1..................... " << lazyTools.e3x1( *it ) << std::endl;
    std::cout << "e1x5..................... " << lazyTools.e1x5( *it ) << std::endl;
    //std::cout << "e5x1..................... " << lazyTools.e5x1( *it ) << std::endl;
    std::cout << "e2x2..................... " << lazyTools.e2x2( *it ) << std::endl;
    std::cout << "e3x3..................... " << lazyTools.e5x5( *it ) << std::endl;
    std::cout << "e4x4..................... " << lazyTools.e4x4( *it ) << std::endl;
    std::cout << "e5x5..................... " << lazyTools.e3x3( *it ) << std::endl;
    std::cout << "e2x5Right................ " << lazyTools.e2x5Right( *it ) << std::endl;
    std::cout << "e2x5Left................. " << lazyTools.e2x5Left( *it ) << std::endl;
    std::cout << "e2x5Top.................. " << lazyTools.e2x5Top( *it ) << std::endl;
    std::cout << "e2x5Bottom............... " << lazyTools.e2x5Bottom( *it ) << std::endl;
    std::cout << "e2x5Max.................. " << lazyTools.e2x5Max( *it ) << std::endl;
    std::cout << "eMax..................... " << lazyTools.eMax( *it ) << std::endl;
    std::cout << "e2nd..................... " << lazyTools.e2nd( *it ) << std::endl;
    std::vector<float> vEta = lazyTools.energyBasketFractionEta( *it );
    std::cout << "energyBasketFractionEta..";
    for (size_t i = 0; i < vEta.size(); ++i ) {
      std::cout << " " << vEta[i];
    }
    std::cout << std::endl;
    std::vector<float> vPhi = lazyTools.energyBasketFractionPhi( *it );
    std::cout << "energyBasketFractionPhi..";
    for (size_t i = 0; i < vPhi.size(); ++i ) {
      std::cout << " " << vPhi[i];
    }
    std::cout << std::endl;
    std::vector<float> vLat = lazyTools.lat( *it );
    std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
    std::vector<float> vCov = lazyTools.covariances( *it );
    std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl;  
    std::vector<float> vLocCov = lazyTools.localCovariances( *it );
    std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
    std::cout << "zernike20................ " << lazyTools.zernike20( *it ) << std::endl;
    std::cout << "zernike42................ " << lazyTools.zernike42( *it ) << std::endl;
  }
  
  std::cout << "========== ENDCAPS ==========" << std::endl;
  for (reco::BasicClusterCollection::const_iterator it = eeClusters->begin(); it != eeClusters->end(); ++it ) {
    std::cout << "----- new cluster -----" << std::endl;
    std::cout << "----------------- size: " << (*it).size() << " energy: " << (*it).energy() << std::endl;
    
    std::cout << "e1x3..................... " << lazyTools.e1x3( *it ) << std::endl;
    std::cout << "e3x1..................... " << lazyTools.e3x1( *it ) << std::endl;
    std::cout << "e1x5..................... " << lazyTools.e1x5( *it ) << std::endl;
    //std::cout << "e5x1..................... " << lazyTools.e5x1( *it ) << std::endl;
    std::cout << "e2x2..................... " << lazyTools.e2x2( *it ) << std::endl;
    std::cout << "e3x3..................... " << lazyTools.e5x5( *it ) << std::endl;
    std::cout << "e4x4..................... " << lazyTools.e4x4( *it ) << std::endl;
    std::cout << "e5x5..................... " << lazyTools.e3x3( *it ) << std::endl;
    std::cout << "e2x5Right................ " << lazyTools.e2x5Right( *it ) << std::endl;
    std::cout << "e2x5Left................. " << lazyTools.e2x5Left( *it ) << std::endl;
    std::cout << "e2x5Top.................. " << lazyTools.e2x5Top( *it ) << std::endl;
    std::cout << "e2x5Bottom............... " << lazyTools.e2x5Bottom( *it ) << std::endl;
    std::cout << "eMax..................... " << lazyTools.eMax( *it ) << std::endl;
    std::cout << "e2nd..................... " << lazyTools.e2nd( *it ) << std::endl;
    std::vector<float> vLat = lazyTools.lat( *it );
    std::cout << "lat...................... " << vLat[0] << " " << vLat[1] << " " << vLat[2] << std::endl;
    std::vector<float> vCov = lazyTools.covariances( *it );
    std::cout << "covariances.............. " << vCov[0] << " " << vCov[1] << " " << vCov[2] << std::endl; 
    std::vector<float> vLocCov = lazyTools.localCovariances( *it );
    std::cout << "local covariances........ " << vLocCov[0] << " " << vLocCov[1] << " " << vLocCov[2] << std::endl;
    std::cout << "zernike20................ " << lazyTools.zernike20( *it ) << std::endl;
    std::cout << "zernike42................ " << lazyTools.zernike42( *it ) << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(testEcalClusterLazyTools);
