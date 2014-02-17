#ifndef TESTTRACKHITS_H
#define TESTTRACKHITS_H
// -*- C++ -*-
//
// Package:    TestTrackHits
// Class:      TestTrackHits
// 
/**\class TestTrackHits TestTrackHits.cc RecoTracker/TestTrackHits/src/TestTrackHits.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giuseppe Cerati
//         Created:  Tue Feb 13 17:29:10 CET 2007
// $Id: TestTrackHits.h,v 1.8 2010/02/11 00:14:37 wmtan Exp $
//
//


#include <memory>
#include <vector>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
//#include "PhysicsTools/RecoAlgos/interface/RecoTrackSelector.h"
#include <sstream>

class TestTrackHits : public edm::EDAnalyzer {
public:
  explicit TestTrackHits(const edm::ParameterSet&);
  ~TestTrackHits();

private:
  virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  std::pair<LocalPoint,LocalVector> projectHit(const PSimHit&, const StripGeomDetUnit*, const BoundPlane&);

  template<unsigned int D>
    double computeChi2Increment(MeasurementExtractor, TransientTrackingRecHit::ConstRecHitPointer);
  double computeChi2Increment(MeasurementExtractor me, TransientTrackingRecHit::ConstRecHitPointer hit) {
    switch (hit->dimension()) {
    case 1: return computeChi2Increment<1>(me,hit);
    case 2: return computeChi2Increment<2>(me,hit);
    case 3: return computeChi2Increment<3>(me,hit);
    case 4: return computeChi2Increment<4>(me,hit);
    case 5: return computeChi2Increment<5>(me,hit);
    }
    throw cms::Exception("CkfDebugger error: rechit of dimension not 1,2,3,4,5");
  }

  const edm::ParameterSet conf_;
  TrackerHitAssociator * hitAssociator;

  std::string propagatorName;
  std::string builderName;
  std::string srcName;
  std::string tpName;
  std::string updatorName;
  std::string out;
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  edm::ESHandle<TrajectoryStateUpdator> theUpdator;
  edm::ESHandle<TrackAssociatorBase> trackAssociator;
  edm::Handle<std::vector<Trajectory> > trajCollectionHandle;
  edm::Handle<edm::View<reco::Track> > trackCollectionHandle;
  edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationCollectionHandle;
  edm::Handle<TrackingParticleCollection> trackingParticleCollectionHandle;
  //RecoTrackSelector selectRecoTracks;

  TFile* file;
  std::stringstream title;
  std::map<std::string,TH1F*> hPullGP_X_ts;
  std::map<std::string,TH1F*> hPullGP_Y_ts;
  std::map<std::string,TH1F*> hPullGP_Z_ts;
  std::map<std::string,TH1F*> hPullGM_X_ts;
  std::map<std::string,TH1F*> hPullGM_Y_ts;
  std::map<std::string,TH1F*> hPullGM_Z_ts;
  std::map<std::string,TH1F*> hPullGP_X_rs;
  std::map<std::string,TH1F*> hPullGP_Y_rs;
  std::map<std::string,TH1F*> hPullGP_Z_rs;
  std::map<std::string,TH1F*> hPullGP_X_tr;
  std::map<std::string,TH1F*> hPullGP_Y_tr;
  std::map<std::string,TH1F*> hPullGP_Z_tr;
  std::map<std::string,TH1F*> hChi2Increment;  
  std::map<std::string,TH1F*> hChi2Increment_mono;
  std::map<std::string,TH1F*> hChi2Increment_stereo;
  std::map<std::string,TH2F*> hChi2IncrementVsEta;  
  std::map<std::string,TH1F*> hChi2GoodHit;  
  std::map<std::string,TH1F*> hChi2BadHit;
  std::map<std::string,TH1F*> hChi2DeltaHit;
  std::map<std::string,TH1F*> hChi2NSharedHit;
  std::map<std::string,TH1F*> hChi2SharedHit;
  TH1F *hTotChi2Increment, *hTotChi2GoodHit, *hTotChi2BadHit, *hTotChi2DeltaHit, *hTotChi2NSharedHit, *hTotChi2SharedHit;
  TH2F *hProcess_vs_Chi2, *hClsize_vs_Chi2, *hGoodHit_vs_Chi2;
  TH2F *hPixClsize_vs_Chi2, *hPrjClsize_vs_Chi2, *hSt1Clsize_vs_Chi2, *hSt2Clsize_vs_Chi2;
  TH1F *hClusterSize, *hPixClusterSize, *hPrjClusterSize, *hSt1ClusterSize, *hSt2ClusterSize;
  TH1F *hSimHitVecSize, *hPixSimHitVecSize, *hPrjSimHitVecSize, *hSt1SimHitVecSize, *hSt2SimHitVecSize;
  TH1F *goodbadmerged,*energyLossRatio, *mergedPull;
  TH1F *probXgood,*probXbad,*probXdelta,*probXshared,*probXnoshare;
  TH1F *probYgood,*probYbad,*probYdelta,*probYshared,*probYnoshare;

  std::map<std::string,TH1F*> hPullGP_X_ts_mono;
  std::map<std::string,TH1F*> hPullGP_Y_ts_mono;
  std::map<std::string,TH1F*> hPullGP_Z_ts_mono;
  std::map<std::string,TH1F*> hPullGM_X_ts_mono;
  std::map<std::string,TH1F*> hPullGM_Y_ts_mono;
  std::map<std::string,TH1F*> hPullGM_Z_ts_mono;
  std::map<std::string,TH1F*> hPullGP_X_rs_mono;
  std::map<std::string,TH1F*> hPullGP_Y_rs_mono;
  std::map<std::string,TH1F*> hPullGP_Z_rs_mono;
  std::map<std::string,TH1F*> hPullGP_X_tr_mono;
  std::map<std::string,TH1F*> hPullGP_Y_tr_mono;
  std::map<std::string,TH1F*> hPullGP_Z_tr_mono;

  std::map<std::string,TH1F*> hPullGP_X_ts_stereo;
  std::map<std::string,TH1F*> hPullGP_Y_ts_stereo;
  std::map<std::string,TH1F*> hPullGP_Z_ts_stereo;
  std::map<std::string,TH1F*> hPullGM_X_ts_stereo;
  std::map<std::string,TH1F*> hPullGM_Y_ts_stereo;
  std::map<std::string,TH1F*> hPullGM_Z_ts_stereo;
  std::map<std::string,TH1F*> hPullGP_X_rs_stereo;
  std::map<std::string,TH1F*> hPullGP_Y_rs_stereo;
  std::map<std::string,TH1F*> hPullGP_Z_rs_stereo;
  std::map<std::string,TH1F*> hPullGP_X_tr_stereo;
  std::map<std::string,TH1F*> hPullGP_Y_tr_stereo;
  std::map<std::string,TH1F*> hPullGP_Z_tr_stereo;
};

#endif
