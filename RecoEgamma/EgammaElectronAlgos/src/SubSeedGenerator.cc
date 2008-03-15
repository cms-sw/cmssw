// Package:    EgammaElectronAlgos
// Class:      SubSeedGenerator.

#include "RecoEgamma/EgammaElectronAlgos/interface/SubSeedGenerator.h" 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>

using namespace std;

SubSeedGenerator::SubSeedGenerator(const edm::ParameterSet& conf) {
  initialSeeds_               = conf.getParameter<edm::InputTag>("initialSeeds");
  dr_                  = conf.getParameter<double>("seedDr");
  dphi_                = conf.getParameter<double>("seedDPhi");
  deta_                = conf.getParameter<double>("seedDEta");
  pt_                  = conf.getParameter<double>("seedPt");
}

SubSeedGenerator::~SubSeedGenerator() {
  
}


void SubSeedGenerator::setupES(const edm::EventSetup& setup) {
}

void  SubSeedGenerator::run(edm::Event& e, const edm::EventSetup& setup, const edm::Handle<reco::SuperClusterCollection> &superClusters, reco::ElectronPixelSeedCollection & out){
  
  edm::ESHandle<TrackerGeometry> tracker;
  setup.get<TrackerDigiGeometryRecord>().get(tracker);
  
  // get initial TrajectorySeeds
  edm::Handle<TrajectorySeedCollection> theInitialSeedColl;
  e.getByLabel(initialSeeds_, theInitialSeedColl);
  
  //seeds selection
  for(unsigned int i=0; i< superClusters->size(); ++i) {
    reco::SuperCluster theClus = (*superClusters)[i];   
    
    std::vector<TrajectorySeed>::const_iterator seed_iter;
    for(seed_iter = theInitialSeedColl->begin(); seed_iter != theInitialSeedColl->end(); ++seed_iter) {
      
      GlobalPoint  gp = tracker->idToDet( DetId(seed_iter->startingState().detId()))->surface().toGlobal( seed_iter->startingState().parameters().position());
      GlobalVector gv = tracker->idToDet( DetId(seed_iter->startingState().detId()))->surface().toGlobal( seed_iter->startingState().parameters().momentum());
      
      math::XYZVector seedGlobalDir(gv.x(),gv.y(),gv.z());   
      math::XYZVector clusterGlobalDir(theClus.x() - gp.x(), theClus.y() - gp.y(), theClus.z() - gp.z());
      
      double clusEt = theClus.energy()*sin(clusterGlobalDir.theta());
      double clusEstimatedCurvature = clusEt/0.3/4*100;  //4 tesla (temporary solution)
      double DphiBending = theClus.position().rho()/2./clusEstimatedCurvature; //ecal radius
      
      //cout << "=== et,curvature, phiBending: " 
      //   << clusEt << " , "
      //   << clusEstimatedCurvature << " , "
      //   << DphiBending << endl;
      

      double tmpDr = ROOT::Math::VectorUtil::DeltaR(clusterGlobalDir, seedGlobalDir);
      float dEta = fabs(clusterGlobalDir.Eta() - seedGlobalDir.Eta());
      float dPhi = fabs(acos(cos(clusterGlobalDir.Phi() - seedGlobalDir.Phi())));     

      float dPhi1 = fabs(dPhi - DphiBending);
      float dPhi2 = fabs(dPhi + DphiBending);
      
      if (dEta <= deta_) {
        if (dPhi1 <= dphi_|| dPhi2 <= dphi_ ) {
          if (gv.perp() > pt_) {
            //if(tmpDr <= dr_) {  
              edm::Ref<reco::SuperClusterCollection> sclRef=edm::Ref<reco::SuperClusterCollection> (superClusters,i);
              out.push_back(reco::ElectronPixelSeed(sclRef,*seed_iter)); 
	      //}
          }
        }
      }
    }
    
  }//end loop over cluster
  
  
  edm::LogVerbatim("myElectronProd") << "========== SubSeedsCollectionProducer Info ==========";
  edm::LogVerbatim("myElectronProd") << "number of initial seeds: " << theInitialSeedColl->size();
  edm::LogVerbatim("myElectronProd") << "number of filtered seeds: " << out.size();
  edm::LogVerbatim("myElectronProd") << "=================================================";
  
}

