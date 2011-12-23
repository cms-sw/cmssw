// -*- C++ -*-
//
// Package:    EgammaHLTAlgos
// Class:      EgammaHLTPixelMatchElectronAlgo.
// 
/**\class EgammaHLTPixelMatchElectronAlgo EgammaHLTAlgos/EgammaHLTPixelMatchElectronAlgo

 Description: top algorithm producing TrackCandidate and Electron objects from supercluster
              driven pixel seeded Ckf tracking for HLT
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
// $Id: EgammaHLTPixelMatchElectronAlgo.cc,v 1.14 2009/10/14 14:18:31 covarell Exp $
//
//
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTPixelMatchElectronAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"

using namespace edm;
using namespace std;
using namespace reco;
//using namespace math; // conflicts with DataFormat/Math/interface/Point3D.h!!!!

//EgammaHLTPixelMatchElectronAlgo::EgammaHLTPixelMatchElectronAlgo():  
//  theCkfTrajectoryBuilder(0), theTrajectoryCleaner(0),
//  theInitialStateEstimator(0), theNavigationSchool(0) {}

EgammaHLTPixelMatchElectronAlgo::EgammaHLTPixelMatchElectronAlgo(const edm::ParameterSet &conf) :
  trackProducer_( conf.getParameter<edm::InputTag>("TrackProducer") ),
  BSProducer_(    conf.getParameter<edm::InputTag>("BSProducer") )
{}

EgammaHLTPixelMatchElectronAlgo::~EgammaHLTPixelMatchElectronAlgo() {

  // delete theInitialStateEstimator;
  //delete theNavigationSchool;
  //delete theTrajectoryCleaner; 
    
}

void EgammaHLTPixelMatchElectronAlgo::setupES(const edm::EventSetup& es) {
  //services
  es.get<TrackerRecoGeometryRecord>().get(theGeomSearchTracker);
  es.get<IdealMagneticFieldRecord>().get(theMagField);
}

void  EgammaHLTPixelMatchElectronAlgo::run(Event& e, ElectronCollection & outEle) {

  // get the input 
  edm::Handle<TrackCollection> tracksH;
  //  e.getByLabel(trackLabel_,trackInstanceName_,tracksH);
 e.getByLabel(trackProducer_,tracksH);

  //Get the Beam Spot position
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  // iEvent.getByType(recoBeamSpotHandle);
  e.getByLabel(BSProducer_,recoBeamSpotHandle);
  // gets its position
  const BeamSpot::Point& BSPosition = recoBeamSpotHandle->position(); 
  Global3DPoint bs(BSPosition.x(),BSPosition.y(),0);
  process(tracksH,outEle,bs);

  return;
}

void EgammaHLTPixelMatchElectronAlgo::process(edm::Handle<TrackCollection> tracksH, ElectronCollection & outEle, Global3DPoint & bs) {
  const TrackCollection *tracks=tracksH.product();
  for (unsigned int i=0;i<tracks->size();++i) {
    const Track & t=(*tracks)[i];

    const TrackRef trackRef = edm::Ref<TrackCollection>(tracksH,i);
    edm::RefToBase<TrajectorySeed> seed = trackRef->extra()->seedRef();
    ElectronSeedRef elseed=seed.castTo<ElectronSeedRef>();

    edm::RefToBase<CaloCluster> caloCluster = elseed->caloCluster() ;
    SuperClusterRef scRef = caloCluster.castTo<SuperClusterRef>() ;

    //const SuperClusterRef & scRef=elseed->superCluster();
    
        // Get the momentum at vertex (not at the innermost layer)
    TSCPBuilderNoMaterial tscpBuilder;
    
    FreeTrajectoryState fts = trajectoryStateTransform::innerFreeState(t,theMagField.product());
    TrajectoryStateClosestToPoint tscp = tscpBuilder(fts, bs );
    
    float scale = scRef->energy()/tscp.momentum().mag();
  
    const math::XYZTLorentzVector momentum(tscp.momentum().x()*scale,
 					   tscp.momentum().y()*scale,
 					   tscp.momentum().z()*scale,
					   scRef->energy());

    
    Electron ele(t.charge(),momentum, t.vertex() );
    ele.setSuperCluster(scRef);
    edm::Ref<TrackCollection> myRef(tracksH,i);
    ele.setTrack(myRef);
    outEle.push_back(ele);

  }  // loop over tracks
}

