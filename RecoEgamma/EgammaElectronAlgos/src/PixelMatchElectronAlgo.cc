// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      PixelMatchElectronAlgo.
// 
/**\class PixelMatchElectronAlgo EgammaElectronAlgos/PixelMatchElectronAlgo

 Description: top algorithm producing TrackCandidate and Electron objects from supercluster
              driven pixel seeded Ckf tracking

*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Thu july 6 13:22:06 CEST 2006
// $Id: PixelMatchElectronAlgo.cc,v 1.5 2006/08/24 08:26:53 charlot Exp $
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchElectronAlgo.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "Geometry/Vector/interface/GlobalPoint.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Units/PhysicalConstants.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace math;

PixelMatchElectronAlgo::PixelMatchElectronAlgo(double maxEOverP, double maxHOverE, 
                                               double maxDeltaEta, double maxDeltaPhi):  
 maxEOverP_(maxEOverP), maxHOverE_(maxHOverE), maxDeltaEta_(maxDeltaEta), 
 maxDeltaPhi_(maxDeltaPhi), theCkfTrajectoryBuilder(0), theTrajectoryCleaner(0),
 theInitialStateEstimator(0), theMeasurementTracker(0), theNavigationSchool(0) {}

PixelMatchElectronAlgo::~PixelMatchElectronAlgo() {

  delete theInitialStateEstimator;
  delete theMeasurementTracker;
  delete theNavigationSchool;
  delete theCkfTrajectoryBuilder;
  delete theTrajectoryCleaner; 
    
}

void PixelMatchElectronAlgo::setupES(const edm::EventSetup& es, const edm::ParameterSet &conf) {

  //services
  es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
  es.get<IdealMagneticFieldRecord>().get(theMagField);

  // get nested parameter set for the TransientInitialStateEstimator
  ParameterSet tise_params = conf.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;
  theInitialStateEstimator       = new TransientInitialStateEstimator( es,tise_params);

  // get nested parameter set for the MeasurementTracker
  ParameterSet mt_params = conf.getParameter<ParameterSet>("MeasurementTrackerParameters") ;
  theMeasurementTracker = new MeasurementTracker(es, mt_params);

  theNavigationSchool   = new SimpleNavigationSchool(&(*theGeomSearchTracker),&(*theMagField));

  // set the correct navigation
  NavigationSetter setter( *theNavigationSchool);

  theCkfTrajectoryBuilder = new CkfTrajectoryBuilder(conf,es,theMeasurementTracker);
  theTrajectoryCleaner = new TrajectoryCleanerBySharedHits();    

  inputDataModuleLabel_=conf.getParameter<string>("SeedProducer");
  //inputDataInstanceName_=conf.getParameter<string>("seedLabel");
  
}

//void  PixelMatchElectronAlgo::run(const edm::Event& e, reco::ElectronTrackCollection & outTk, 
void  PixelMatchElectronAlgo::run(const Event& e, TrackCandidateCollection & outTk, 
 ElectronCollection & outEle) {

  theMeasurementTracker->update(e);

  Handle<ElectronPixelSeedCollection> collseed;
  LogDebug("") << 
   "PixelMatchElectronAlgo::run, getting input seeds : " << inputDataModuleLabel_ ;
  std::cout << "PixelMatchElectronAlgo::run, getting input seeds : " <<
   inputDataModuleLabel_ <<std::endl;
  e.getByLabel(inputDataModuleLabel_, collseed);
  ElectronPixelSeedCollection theSeedColl = *collseed;
  LogDebug("") << 
   "PixelMatchElectronAlgo::run, got " << (*collseed).size()<< " input seeds ";
  std::cout << "PixelMatchElectronAlgo::run, got " << (*collseed).size()<< " input seeds " << std::endl;

  // this is needed because lack of polymorphism
  map<const ElectronPixelSeed*, const Trajectory*> seedMap;
  map<const ElectronPixelSeed*, const Trajectory*>::iterator itmap;
  
  // first build electron tracks
  if ((*collseed).size()>0){
    vector<Trajectory> theFinalTrajectories;
    ElectronPixelSeedCollection::const_iterator iseed;
      
    vector<Trajectory> rawResult;

    cout << "Starting loop over seeds." << endl;
    LogDebug("") << "Starting loop over seeds ";
    for(iseed=theSeedColl.begin();iseed!=theSeedColl.end();iseed++){
      LogDebug("") << "new seed ";
      vector<Trajectory> theTmpTrajectories;
      theTmpTrajectories = theCkfTrajectoryBuilder->trajectories(*iseed);
      LogDebug("") << "CkfTrajectoryBuilder returned " << theTmpTrajectories.size()
			     << " trajectories for this seed";
      theTrajectoryCleaner->clean(theTmpTrajectories);
      for(vector<Trajectory>::const_iterator it=theTmpTrajectories.begin();
	  it!=theTmpTrajectories.end(); it++){
	if( it->isValid() ) {
	  rawResult.push_back(*it);
	  // this is uggly
	  seedMap[&(*iseed)] = &(*it);
	}
      }
      LogDebug("PixelMatchElectronAlgoCkfPattern") << "Number of trajectories after cleaning " << rawResult.size();
    }
    LogDebug("") << "End loop over seeds";
    cout << "End loop over seeds." << endl;

    vector<Trajectory> unsmoothedResult;
    LogDebug("") << "Starting second cleaning..." << std::endl;
    theTrajectoryCleaner->clean(rawResult);

    for (vector<Trajectory>::const_iterator itraw = rawResult.begin();
	 itraw != rawResult.end(); itraw++) {
      if((*itraw).isValid()) unsmoothedResult.push_back( *itraw);
    }
    cout << "Number of trajectories after second cleaning " << rawResult.size() <<endl;
    LogDebug("PixelMatchElectronAlgoCkfPattern") << "Number of trajectories after second cleaning " << rawResult.size();
    //analyseCleanedTrajectories(unsmoothedResult);

    for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin();
	 it != unsmoothedResult.end(); it++) {

      OwnVector<TrackingRecHit> recHits;
      float ndof=0;
      //RC OwnVector<const TransientTrackingRecHit> thits = it->recHits();
      TransientTrackingRecHit::RecHitContainer thits = it->recHits();
      for (TransientTrackingRecHit::RecHitContainer::const_iterator hitIt = thits.begin(); 
	   hitIt != thits.end(); hitIt++) {
	recHits.push_back( (*hitIt)->hit()->clone());
	if (hitIt->get()->isValid()) ndof = ndof + ((**hitIt).dimension())*((**hitIt).weight());
      }
      ndof = ndof - 5;
      
      //PTrajectoryStateOnDet state = *(it->seed().startingState().clone());
      std::pair<TrajectoryStateOnSurface, const GeomDet*> initState = 
	theInitialStateEstimator->innerState( *it);

      // temporary protection againt invalid initial states
      if (! initState.first.isValid() || initState.second == 0) {
        //cout << "invalid innerState, will not make TrackCandidate" << endl;
        continue;
      }

      PTrajectoryStateOnDet* state = TrajectoryStateTransform().persistentState( initState.first,
										 initState.second->geographicalId().rawId());
      //	FitTester fitTester(es);
      //	fitTester.fit( *it);

      TrackCandidate aTrackCandidate(recHits,*(it->seed().clone()),*state);
      cout << "New track candidate created" << std::endl;
      LogDebug("") << "New track candidate created";
      LogDebug("") << "n valid and invalid hit, chi2 : " 
	 << it->foundHits() << " , " << it->lostHits() <<" , " <<it->chiSquared();
      outTk.push_back(aTrackCandidate);
      
      // construct Track from TrackCandidate 
      // for the time being TrackCandidate are promoted to Track without refitting forward and backward
      // this is to be done by the TrackProducer algos when we'll be able to retrieve our Tracks from the framework

      TSCPBuilderNoMaterial tscpBuilder;
    
      TrajectoryStateClosestToPoint tscp = tscpBuilder(*(initState.first.freeState()), Global3DPoint(0,0,0) );

      PerigeeTrajectoryParameters::ParameterVector param = tscp.perigeeParameters();
      PerigeeTrajectoryError::CovarianceMatrix covar = tscp.perigeeError();

      Track aTrack(it->chiSquared(),
       int(ndof), //FIXME fix weight() in TrackingRecHit
       //it->foundHits(),
       //0, //FIXME no corresponding method in trajectory.h
       //it->lostHits(),//FIXME to be fixed in Trajectory.h
       param, tscp.pt(), covar);
       
      cout << "New track created" << std::endl;
      LogDebug("") << "New track created";
      cout << "n valid and invalid hit, chi2 : " 
	 << it->foundHits() << " , " << it->lostHits() <<" , " <<it->chiSquared() << endl;
      LogDebug("") << "n valid and invalid hit, chi2 : " 
	 << it->foundHits() << " , " << it->lostHits() <<" , " <<it->chiSquared();

      // now build electrons
 
      //This one is not polymorphic, access by value!!
      //Uggly code to retreive the supercluster pointer     
      //const ElectronPixelSeed* epseed = dynamic_cast<ElectronPixelSeed *>((it->seed().clone()));
      const ElectronPixelSeed* epseed=0;
      for (itmap=seedMap.begin(); itmap!=seedMap.end(); itmap++) {
	epseed = itmap->first;
	if (itmap->second == &(*it)) break;
      }

      if (preSelection(*(epseed->superCluster()),aTrack)) {      
	cout << "Creating new electron " << std::endl;
	
	// for the time being take the momentum from the track 
	const XYZTLorentzVector momentum(tscp.momentum().x(),
	                                 tscp.momentum().y(),
	                                 tscp.momentum().z(),
	                                 sqrt(tscp.momentum().mag2() + electron_mass_c2*electron_mass_c2*1.e-6) );
	
	Electron ele(aTrack.charge(),momentum,XYZPoint( 0, 0, 0 ));
	cout << "electron energy " << epseed->superCluster()->energy() << endl;
        ele.setSuperCluster(epseed->superCluster());

	cout << "New electron created " << std::endl;
	outEle.push_back(ele);
        LogDebug("PixelMatchElectronAlgoCkfPattern") << "New electron created";
      }

    }
  
  }
    
  cout << "========== PixelMatchElectronAlgo Info ==========" << endl;
  LogInfo("PixelMatchElectronAlgo") << "========== PixelMatchElectronAlgo Info ==========";
  LogInfo("PixelMatchElectronAlgo") << "Event " << e.id();
  LogInfo("PixelMatchElectronAlgo") << "Number of seeds: " << theSeedColl.size();
  cout << "Number of final electron tracks: " << outTk.size() << endl;
  LogInfo("PixelMatchElectronAlgo") << "Number of final electron tracks: " << outTk.size();
  cout << "Number of final electrons: " << outEle.size() << endl;
  LogInfo("PixelMatchElectronAlgo") << "Number of final electrons: " << outEle.size();
  for (vector<Electron>::const_iterator it = outEle.begin(); it != outEle.end(); it++) {
    cout << "New electron with charge, pt, eta, phi " 
	 << it->charge() << " , " << it->pt() << " , " << it->eta() << " , " << it->phi() << endl;
    LogInfo("PixelMatchElectronAlgo") << "New electron with charge, pt, eta, phi : " 
	 << it->charge() << " , " << it->pt() << " , " << it->eta() << " , " << it->phi();
  }
  cout << "=================================================" << endl;
  LogInfo("PixelMatchElectronAlgo") << "=================================================";
     
}

bool PixelMatchElectronAlgo::preSelection(const SuperCluster& clus, const Track& track) 
{
  // to be implemented
  return true;  
}  

