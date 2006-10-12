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
// $Id: PixelMatchElectronAlgo.cc,v 1.12 2006/10/04 10:47:10 rahatlou Exp $
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchElectronAlgo.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"

#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Units/PhysicalConstants.h"

#include <sstream>

using namespace edm;
using namespace std;
using namespace reco;
//using namespace math; // conflicts with DataFormat/Math/interface/Point3D.h!!!!

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

void  PixelMatchElectronAlgo::run(Event& e, ElectronCollection & outEle) {

//   ==============================   preparations ==============================
  reco::TrackExtraRefProd rTrackExtras = e.getRefBeforePut<reco::TrackExtraCollection>();
  TrackingRecHitRefProd rHits = e.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackRefProd refprod = e.getRefBeforePut<reco::TrackCollection>();

  std::auto_ptr<TrackCollection> outTracks(new TrackCollection);
  std::auto_ptr<TrackCandidateCollection> outTrackCandidates(new TrackCandidateCollection);
  std::auto_ptr<TrackingRecHitCollection> selHits(new TrackingRecHitCollection);
  std::auto_ptr<TrackExtraCollection> outTrackExtras(new TrackExtraCollection);
  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;


  theMeasurementTracker->update(e);

  Handle<ElectronPixelSeedCollection> collseed;
  LogDebug("") << 
    "PixelMatchElectronAlgo::run, getting input seeds : " << inputDataModuleLabel_ ;
  e.getByLabel(inputDataModuleLabel_, collseed);
  ElectronPixelSeedCollection theSeedColl = *collseed;

  LogDebug("") << 
    "PixelMatchElectronAlgo::run, got " << (*collseed).size()<< " input seeds ";
  // this is needed because lack of polymorphism
  map<const ElectronPixelSeed*, const Trajectory*> seedMap;
  map<const ElectronPixelSeed*, const Trajectory*>::iterator itmap;

  //  std::vector<const ElectronPixelSeed *> seedPtrs;
  
  if ((*collseed).size()>0){
    vector<Trajectory> theFinalTrajectories;
    ElectronPixelSeedCollection::const_iterator iseed;
      
    vector<Trajectory> rawResult;
    vector<const Trajectory*> rawResultPtr; // ShR: ptr to original tracks used as keys in the map below

    LogDebug("") << "Starting loop over seeds ";

    // ShR: use trajectory as key to retrieve the correct seed
    map<const Trajectory*, const ElectronPixelSeed*> trajToSeedMap;

//   ==============================   loop over seeds ==============================
    //std::cout << "fill the maps seeds <--> trajectory" << std::endl;
    unsigned ik=0;
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
	    rawResult.push_back( *it ); // ShR: problem -- this is already a copy which is the beginning of the problem.
                                      //    elements of rawResult have a diferente address than the original
                                      //    trajectories used as keys to the map above

          const Trajectory* iTraj = &(rawResult.back());
          rawResultPtr.push_back( iTraj ); // ShR: store ptr to original trajectories
          //seedMap[&(*iseed)] = &(*it); // ShR: Problem -- using ref of temporary object
          seedMap[&(*iseed)] = iTraj; // iTraj is address of element of a vector that exists out of this scope
          trajToSeedMap[iTraj] = &(*iseed); // ShR: trajectory -> seed
          //std::cout << "traj: " << &(rawResult.back()) << "   seed: " << &(*iseed)
          //          << " &(rawResult[ik]): " << &(rawResult[ik])
          //          << " rawResultPtr[ik]: " << rawResultPtr[ik] << std::endl;
          ik++;
	  }
      }
      LogDebug("PixelMatchElectronAlgoCkfPattern") << "Number of trajectories after cleaning " << rawResult.size();
    }
    LogDebug("") << "End loop over seeds";

    // ShR: the loop above was over theTmpTrajectories and pointers were used to fill the map
    // the real problms starts below where a COPY of the tarjectory is made instead of their address
    // this is the usual problem of copying objects but using the ptr to the original instance

//   ==============================   loop over trajectories ==============================
    vector<Trajectory> unsmoothedResult;
    vector<const Trajectory*> unsmoothedResultPtrs; // ShR: vector of addresses not copies!
    unsigned int unInd = 0; // ShR: needed to find the correct pointer to use with the map after 2nd cleaning
                            // unInd runs over all elements in unsmoothedResult and is needed to get back to the
                            // ptr to the original trajectories used as keys in the map

    // ShR: address of first couple of elements are ALWYS different at this point! compare
    //      to printout from above inside the loop
    //      This is the reason we still need the vector of pointers rawResultPtr until we understand the real issue here
    //for(unsigned int ij=0; ij< rawResult.size(); ++ij) {
    //   std::cout << "ij: " << ij << " &(rawResult[ij]): " << &(rawResult[ij])
    //                             << " rawResultPtr[ij]): " << rawResultPtr[ij] << std::endl;
    //}

    LogDebug("") << "Starting second cleaning..." << std::endl;
    theTrajectoryCleaner->clean(rawResult);
    for (vector<Trajectory>::const_iterator itraw = rawResult.begin();
        itraw != rawResult.end(); itraw++) {

    if((*itraw).isValid())  {

        unsmoothedResult.push_back( *itraw);
        unsmoothedResultPtrs.push_back( rawResultPtr[unInd] ); // ShR: store the original ptr used in the map
        //unsmoothedResultPtrs.push_back( &(*itraw) ); // ShR: Problem -- although the natural thing to do this does not work
                                                       //       because the address of first elements of rawResult changes!
        //std::cout << "found valid trajectory  rawResultPtr[unInd]: " << rawResultPtr[unInd] << "  &(*itraw): " << &(*itraw) << std::endl;
      }
      unInd++;
    }
    LogDebug("PixelMatchElectronAlgoCkfPattern") << "Number of trajectories after second cleaning " << rawResult.size();
    //analyseCleanedTrajectories(unsmoothedResult);

//   ==============================   loop over cleaned trajectories  and create tracks ==============================
   unsigned int ind=0;
   unsigned int trajInd = 0; // ShR: index needed to find the correct pointer in unsmoothedResultPtrs
   for (vector<Trajectory>::iterator it = unsmoothedResult.begin();
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

      //      TrackCandidate aTrackCandidate(recHits,*(it->seed().clone()),*state);
      TrackCandidate * aTrackCandidatePtr= new TrackCandidate(recHits,*(it->seed().clone()),*state);
      LogDebug("") << "New track candidate created";
      LogDebug("") << "n valid and invalid hit, chi2 : " 
		   << it->foundHits() << " , " << it->lostHits() <<" , " <<it->chiSquared();
      outTrackCandidates->push_back(*aTrackCandidatePtr);

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
      outTracks->push_back(aTrack);

      // fill in track Extras
      Trajectory *theTraj = &(*it);
      //sets the outermost and innermost TSOSs
      TrajectoryStateOnSurface outertsos;
      TrajectoryStateOnSurface innertsos;
      unsigned int innerId, outerId;
      if (theTraj->direction() == alongMomentum) {
	outertsos = theTraj->lastMeasurement().updatedState();
	innertsos = theTraj->firstMeasurement().updatedState();
	outerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
	innerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
      } else { 
	outertsos = theTraj->firstMeasurement().updatedState();
	innertsos = theTraj->lastMeasurement().updatedState();
	outerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
	innerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
      }
      //build the TrackExtra
      GlobalPoint v = outertsos.globalParameters().position();
      GlobalVector p = outertsos.globalParameters().momentum();
      math::XYZVector outmom( p.x(), p.y(), p.z() );
      math::XYZPoint  outpos( v.x(), v.y(), v.z() );
      v = innertsos.globalParameters().position();
      p = innertsos.globalParameters().momentum();
      math::XYZVector inmom( p.x(), p.y(), p.z() );
      math::XYZPoint  inpos( v.x(), v.y(), v.z() );

      reco::TrackExtraRef teref= reco::TrackExtraRef ( rTrackExtras, idx ++ );
      reco::Track & track =outTracks->back();
      track.setExtra( teref );
      outTrackExtras->push_back( reco::TrackExtra (outpos, outmom, true, inpos, inmom, true,
						   outertsos.curvilinearError(), outerId,
						   innertsos.curvilinearError(), innerId));

      const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits();
      reco::TrackExtra & tx = outTrackExtras->back();
      size_t i = 0;
      for( TrajectoryFitter::RecHitContainer::const_iterator j = transHits.begin();
	   j != transHits.end(); j ++ ) {
	TrackingRecHit * hit = (**j).hit()->clone();
	selHits->push_back( hit );
	track.setHitPattern( * hit, i ++ );
	tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
      }


      LogDebug("") << "New track created " << it->foundHits() << " , " << it->lostHits() <<" , " <<it->chiSquared() << "\n"
		   << "n valid and invalid hit, chi2 : " << it->foundHits() << " , " << it->lostHits() <<" , "
		   <<it->chiSquared();

      //std::cout << "&(*it): " << &(*it) << "  trajInd: " << trajInd
      //           << "  unsmoothedResultPtrs[trajInd]: " << unsmoothedResultPtrs[trajInd]
      //           << std::endl;

      //This one is not polymorphic, access by value!!
      //Ugly code to retrieve the supercluster pointer
      //const ElectronPixelSeed* epseed = dynamic_cast<ElectronPixelSeed *>((it->seed().clone()));
      const ElectronPixelSeed* epseed=0; // ShR: original seed
      const ElectronPixelSeed* epseedNew=0; // ShR: needed to show the problem with bad pointers
      for (itmap=seedMap.begin(); itmap!=seedMap.end(); itmap++) {
        epseed = itmap->first;
        if (itmap->second == &(*it)) break;
      }

      // ShR: trajInd is used to find the ptr to the original track stored in unsmoothedResultPtrs
      // ShR  unsmoothedResultPtrs[trajInd] is the correct key to use to find the seed
      // ShR  no loop is necessary since the new trajToSeedMap map uses trajectories as the key and seed as value
      epseedNew = trajToSeedMap[ unsmoothedResultPtrs[trajInd] ]; // remember: address of elements in rawResult used as key
      trajInd++; // ShR: increase the index to stay in synch with the iterator

      // ShR: compare the two seeds
      //std::cout << "epseed: " << epseed << "  epseedNew: " << epseedNew << std::endl;

      //if(seedMap.find(epseed)==seedMap.end()) {
        //std::cout << "seedMap has no entry with key " << epseed << std::endl;
      //}

      // ShR: set epseed = epseedNew to avoid making changes in the code below
      epseed = epseedNew;

      if (preSelection(*(epseed->superCluster()),aTrack)) {
	// for the time being take the momentum from the track 
	const math::XYZTLorentzVector momentum(tscp.momentum().x(),
					       tscp.momentum().y(),
					       tscp.momentum().z(),
					       sqrt(tscp.momentum().mag2() + electron_mass_c2*electron_mass_c2*1.e-6) );
	
	Electron ele(tscp.charge(),momentum,math::XYZPoint( 0, 0, 0 ));
	LogDebug("") << "electron energy " << epseed->superCluster()->energy();
      ele.setSuperCluster(epseed->superCluster());
	edm::Ref<TrackCollection> myRef(refprod,ind++);
	ele.setTrack(myRef);
	outEle.push_back(ele);
	LogDebug("PixelMatchElectronAlgoCkfPattern") << "New electron created";
      }
   }
  }

  std::ostringstream str;

  str << "========== PixelMatchElectronAlgo Info ==========";
  str << "Event " << e.id();
  str << "Number of seeds: " << theSeedColl.size();
  //  str << "Number of final electron tracks: " << outTk.size();
  str << "Number of final electrons: " << outEle.size();
  str << "Number of final electron tracks: " << outTracks->size();
  //cout<<" Number of final electrons:"<<outEle.size()<<std::endl;
  for (vector<Electron>::const_iterator it = outEle.begin(); it != outEle.end(); it++) {
    str << "New electron with charge, pt, eta, phi : "  << it->charge() << " , " 
        << it->pt() << " , " << it->eta() << " , " << it->phi();
    //cout << "New electron with charge, pt, eta, phi : "  << it->charge() << " , "  << it->pt() << " , " << it->eta() << " , " << it->phi()<<std::endl;
  }
  e.put(outTracks);
  e.put(outTrackCandidates);
  e.put(outTrackExtras);
  e.put(selHits);
  str << "=================================================";
  LogDebug("PixelMatchElectronAlgo") << str.str();
}

bool PixelMatchElectronAlgo::preSelection(const SuperCluster& clus, const Track& track) 
{
  // to be implemented
  return true;  
}  


