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
// $Id: EgammaHLTPixelMatchElectronAlgo.cc,v 1.3 2007/03/07 09:07:54 monicava Exp $
//
//
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTPixelMatchElectronAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include <TMath.h>
#include <sstream>

using namespace edm;
using namespace std;
using namespace reco;
//using namespace math; // conflicts with DataFormat/Math/interface/Point3D.h!!!!

EgammaHLTPixelMatchElectronAlgo::EgammaHLTPixelMatchElectronAlgo():  
 theCkfTrajectoryBuilder(0), theTrajectoryCleaner(0),
 theInitialStateEstimator(0), theNavigationSchool(0) {}

EgammaHLTPixelMatchElectronAlgo::~EgammaHLTPixelMatchElectronAlgo() {

  delete theInitialStateEstimator;
  delete theNavigationSchool;
  delete theTrajectoryCleaner; 
    
}

void EgammaHLTPixelMatchElectronAlgo::setupES(const edm::EventSetup& es, const edm::ParameterSet &conf) {

  //services
  es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
  es.get<IdealMagneticFieldRecord>().get(theMagField);

  // get nested parameter set for the TransientInitialStateEstimator
  ParameterSet tise_params = conf.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;
  theInitialStateEstimator       = new TransientInitialStateEstimator( es,tise_params);

  theNavigationSchool   = new SimpleNavigationSchool(&(*theGeomSearchTracker),&(*theMagField));

  // set the correct navigation
  NavigationSetter setter( *theNavigationSchool);

  //  theCkfTrajectoryBuilder = new CkfTrajectoryBuilder(conf,es,theMeasurementTracker);
  theTrajectoryCleaner = new TrajectoryCleanerBySharedHits();    
  std::string trajectoryBuilderName = conf.getParameter<std::string>("TrajectoryBuilder");
  edm::ESHandle<TrackerTrajectoryBuilder> theTrajectoryBuilderHandle;
  es.get<CkfComponentsRecord>().get(trajectoryBuilderName,theTrajectoryBuilderHandle);
  theCkfTrajectoryBuilder = theTrajectoryBuilderHandle.product();    

  trackBarrelLabel_ = conf.getParameter<string>("TrackBarrelLabel");
  trackBarrelInstanceName_ = conf.getParameter<string>("TrackBarrelProducer");
  trackEndcapLabel_ = conf.getParameter<string>("TrackEndcapLabel");
  trackEndcapInstanceName_ = conf.getParameter<string>("TrackEndcapProducer");
  assBarrelLabel_ = conf.getParameter<string>("SCLBarrelLabel");
  assBarrelInstanceName_ = conf.getParameter<string>("SCLBarrelProducer");
  assEndcapLabel_ = conf.getParameter<string>("SCLEndcapLabel");
  assEndcapInstanceName_ = conf.getParameter<string>("SCLEndcapProducer");
}

void  EgammaHLTPixelMatchElectronAlgo::run(Event& e, ElectronCollection & outEle) {

  // get the input 
  edm::Handle<TrackCollection> tracksBarrelH;
  edm::Handle<TrackCollection> tracksEndcapH;
  e.getByLabel(trackBarrelLabel_,trackBarrelInstanceName_,tracksBarrelH);
  e.getByLabel(trackEndcapLabel_,trackEndcapInstanceName_,tracksEndcapH);
  edm::Handle<SeedSuperClusterAssociationCollection> barrelH;
  edm::Handle<SeedSuperClusterAssociationCollection> endcapH;
  e.getByLabel(assBarrelLabel_,assBarrelInstanceName_,barrelH);
  e.getByLabel(assEndcapLabel_,assEndcapInstanceName_,endcapH);
  
  // create electrons from tracks in 2 steps: barrel + endcap
  const SeedSuperClusterAssociationCollection  *sclAss=&(*barrelH);
  process(tracksBarrelH,sclAss,outEle);
  sclAss=&(*endcapH);
  process(tracksEndcapH,sclAss,outEle);

  return;
}

void EgammaHLTPixelMatchElectronAlgo::process(edm::Handle<TrackCollection> tracksH,const SeedSuperClusterAssociationCollection *sclAss,ElectronCollection & outEle) {
  const TrackCollection *tracks=tracksH.product();
  for (unsigned int i=0;i<tracks->size();++i) {
    const Track & t=(*tracks)[i];
    // look for corresponding seed
    //temporary as long as there is no way to have a pointer to the seed from the track
    edm::Ref<TrajectorySeedCollection> seed;
    bool found = false;
    for( SeedSuperClusterAssociationCollection::const_iterator it= sclAss->begin(); it != sclAss->end(); ++it) {
      seed=(*it).key;
      if (equal(seed,t)) {
	found=true;
	break;
      }
    }
    
    // for the time being take the momentum from the track 
    const SuperCluster theClus=*((*sclAss)[seed]);
    TSCPBuilderNoMaterial tscpBuilder;
    TrajectoryStateTransform tsTransform;
    FreeTrajectoryState fts = tsTransform.innerFreeState(t,theMagField.product());
    TrajectoryStateClosestToPoint tscp = tscpBuilder(fts, Global3DPoint(0,0,0) );
    
    float scale = (*sclAss)[seed]->energy()/tscp.momentum().mag();
    const math::XYZTLorentzVector momentum(tscp.momentum().x()*scale,
					   tscp.momentum().y()*scale,
					   tscp.momentum().z()*scale,
					   (*sclAss)[seed]->energy());
    
    Electron ele(t.charge(),momentum, t.vertex());
    ele.setSuperCluster((*sclAss)[seed]);
    edm::Ref<TrackCollection> myRef(tracksH,i);
    ele.setTrack(myRef);
    outEle.push_back(ele);

  }  // loop over tracks
}

//**************************************************************************
// all the following  is temporary, to be replaced by a method Track::seed()
//**************************************************************************
bool EgammaHLTPixelMatchElectronAlgo::equal(edm::Ref<TrajectorySeedCollection> ts, const Track& t) {
  // we have 2 valid rechits from the seed
  // which we have to find in the track
  // curiously, they are not the first ones...
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  typedef recHitContainer::const_iterator const_iterator;
  typedef std::pair<const_iterator,const_iterator> range;
  range r=ts->recHits();
  int foundHits=0;
  for (TrackingRecHitCollection::const_iterator rhits=r.first; rhits!=r.second; rhits++) {
    if ((*rhits).isValid()) {
      for (unsigned int j=0;j<t.recHitsSize();j++) {
	TrackingRecHitRef rh =t.recHit(j);
	if (rh->isValid()) {
	  if (compareHits((*rhits),(*rh))) {
	    foundHits++;
	    break;
	  }
	}
      }
    }
  }
  if (foundHits==2) return true;

  return false;
}

bool EgammaHLTPixelMatchElectronAlgo::compareHits(const TrackingRecHit& rh1, const TrackingRecHit & rh2) const {
       const float eps=.001;
       return ((TMath::Abs(rh1.localPosition().x()-rh2.localPosition().x())<eps)
		&& (TMath::Abs(rh1.localPosition().y()-rh2.localPosition().y())<eps)
	       &&(TMath::Abs(rh1.localPosition().z()-rh2.localPosition().z())<eps));
     }
  
