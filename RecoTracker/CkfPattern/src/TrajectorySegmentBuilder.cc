
#include "TrajectorySegmentBuilder.h"

#include "RecoTracker/CkfPattern/src/RecHitIsInvalid.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"
#include "TrackingTools/DetLayers/interface/DetGroup.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/CkfPattern/src/TrajectoryLessByFoundHits.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/DetLayers/interface/GeomDetCompatibilityChecker.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm> 

// #define DBG_TSB

namespace {
#ifdef STAT_TSB
  struct StatCount {
    long long totGroup;
    long long totSeg;
    long long totLockHits;
    long long totInvCand;
    long long trunc;
    void zero() {
      totGroup=totSeg=totLockHits=totInvCand=trunc=0;
    }
    void incr(long long g, long long s, long long l) {
      totGroup+=g;
      totSeg+=s;
      totLockHits+=l;
     }
    void truncated() { ++trunc;}
    void invalid() { ++totInvCand;}
    void print() const {
      std::cout << "TrajectorySegmentBuilder stat\nGroup/Seg/Lock/Inv/Trunc "
    		<< totGroup<<'/'<<totSeg<<'/'<<totLockHits<<'/'<<totInvCand<<'/'<<trunc
		<< std::endl;
    }
    StatCount() { zero();}
    ~StatCount() { print();}
  };

#else
  struct StatCount {
    void incr(long long, long long, long long){}
    void truncated() {}
    void invalid() {}
  };
#endif

  StatCount statCount;

}


using namespace std;

TrajectorySegmentBuilder::TempTrajectoryContainer
TrajectorySegmentBuilder::segments (const TSOS startingState)
{
  //
  // create empty trajectory
  //
  theLockedHits.clear();
  TempTrajectory startingTrajectory(theFullPropagator.propagationDirection());
  //
  // get measurement groups
  //
#ifdef TSB_TRUNCATE
  vector<TMG> measGroups = 
#else  
  vector<TMG> const & measGroups = 
#endif
    theLayerMeasurements->groupedMeasurements(theLayer,startingState,theFullPropagator,theEstimator);
    //B.M. theLayer.groupedMeasurements(startingState,theFullPropagator,theEstimator);

#ifdef DBG_TSB
  cout << "TSB: number of measurement groups = " << measGroups.size() << endl;
  //  theDbgFlg = measGroups.size()>1;
  theDbgFlg = true;
#else
  theDbgFlg = false;
#endif


#ifdef TSB_TRUNCATE
  //  V.I. to me makes things slower...

  //
  // check number of combinations
  //
  constexpr long long  MAXCOMB = 100000000;
  long long ncomb(1);
  int ngrp(0);
  bool truncate(false);
  for (auto const & gr : measGroups) {
    ++ngrp;
    int nhit(0);
    for ( auto const & m : gr.measurements()) if likely( m.recHitR().isValid() )  nhit++;
    
    if ( nhit>1 )  ncomb *= nhit;
    if unlikely( ncomb>MAXCOMB ) {
	edm::LogInfo("TrajectorySegmentBuilder") << " found " << measGroups.size() 
						 << " groups and more than " << static_cast<unsigned int>(MAXCOMB)
						 << " combinations - limiting to "
						 << (ngrp-1) << " groups";
	truncate = true;
	
	statCount.truncated();
	
	break;
      }
  }  
  //   cout << "Groups / combinations = " << measGroups.size() << " " << ncomb << endl;
  if unlikely( truncate && ngrp>0 )  measGroups.resize(ngrp-1);
  

#endif

#ifdef DBG_TSB
  if ( theDbgFlg ) {
    int ntot(1);
    for (vector<TMG>::const_iterator ig=measGroups.begin();
	 ig!=measGroups.end(); ++ig) {
      int ngrp(0);
      const vector<TM>& measurements = ig->measurements();
      for ( vector<TM>::const_iterator im=measurements.begin();
	    im!=measurements.end(); ++im ) {
	if ( im->recHit()->isValid() )  ngrp++;
      }
      cout << " " << ngrp;
      if ( ngrp>0 )  ntot *= ngrp;
    }  
    cout << endl;
    cout << "TrajectorySegmentBuilder::partialTrajectories:: det ids & hit types / group" << endl;
    for (vector<TMG>::const_iterator ig=measGroups.begin();
	 ig!=measGroups.end(); ++ig) {
      const vector<TM>& measurements = ig->measurements();
      for ( vector<TM>::const_iterator im=measurements.begin();
	    im!=measurements.end(); ++im ) {
	if ( im!=measurements.begin() )  cout << " / ";
	if ( im->recHit()->det() )
	  cout << im->recHit()->det()->geographicalId().rawId() << " "
	       << im->recHit()->getType();
	else
	  cout << "no det";
      }
      cout << endl;
    }  
  

//   if ( measGroups.size()>4 ) {
    cout << typeid(theLayer).name() << endl;
    cout << startingState.localError().matrix() << endl;
//     for (vector<TMG>::const_iterator ig=measGroups.begin();
// 	 ig!=measGroups.end(); ig++) {
//       cout << "Nr. of measurements = " << ig->measurements().size() << endl;
//       const DetGroup& dg = ig->detGroup();
//       for ( DetGroup::const_iterator id=dg.begin();
// 	    id!=dg.end(); id++ ) {
// 	GlobalPoint p(id->det()->position());
// 	GlobalVector v(id->det()->toGlobal(LocalVector(0.,0.,1.)));
// 	cout << p.perp() << " " << p.phi() << " " << p.z() << " ; "
// 	     << v.phi() << " " << v.z() << endl;
//       }
//     }
//   }
  }
#endif

  TempTrajectoryContainer candidates = 
    addGroup(startingTrajectory,measGroups.begin(),measGroups.end());

  if unlikely(theDbgFlg) cout << "TSB: back with " << candidates.size() << " candidates" << endl;
  // clean
  //
  //
  // add invalid hit - try to get first detector hit by the extrapolation
  //

  updateWithInvalidHit(startingTrajectory,measGroups,candidates);

  if unlikely(theDbgFlg) cout << "TSB: " << candidates.size() << " candidates after invalid hit" << endl;

  statCount.incr(measGroups.size(), candidates.size(), theLockedHits.size());


  theLockedHits.clear();

  return candidates;
}

void TrajectorySegmentBuilder::updateTrajectory (TempTrajectory& traj,
						 const TM& tm) const
{
  TSOS predictedState = tm.predictedState();
  ConstRecHitPointer hit = tm.recHit();
 
  if ( hit->isValid()) {
    traj.emplace(predictedState, theUpdator.update( predictedState, *hit),
		   hit, tm.estimate(), tm.layer());

//     TrajectoryMeasurement tm(traj.lastMeasurement());
//     if ( tm.updatedState().isValid() ) {
//       if ( !hit.det().surface().bounds().inside(tm.updatedState().localPosition(),
// 						tm.updatedState().localError().positionError(),3.) ) {
// 	cout << "Incompatibility after update for det at " << hit.det().position() << ":" << endl;
// 	cout << tm.predictedState().localPosition() << " " 
// 	     << tm.predictedState().localError().positionError() << endl;
// 	cout << hit.localPosition() << " " << hit.localPositionError() << endl;
// 	cout << tm.updatedState().localPosition() << " "
// 	     << tm.updatedState().localError().positionError() << endl;
//       }
//     }
  }
  else {
    traj.emplace(predictedState, hit,0, tm.layer());
  }
}


TrajectorySegmentBuilder::TempTrajectoryContainer
TrajectorySegmentBuilder::addGroup (TempTrajectory const & traj,
				    vector<TMG>::const_iterator begin,
				    vector<TMG>::const_iterator end)
{
  vector<TempTrajectory> ret;
  if ( begin==end ) {
    //std::cout << "TrajectorySegmentBuilder::addGroup" << " traj.empty()=" << traj.empty() << "EMPTY" << std::endl;
    if unlikely(theDbgFlg) cout << "TSB::addGroup : no groups left" << endl;
    if ( !traj.empty() )
      ret.push_back(traj);
    return ret;
  }
  
  if unlikely(theDbgFlg) cout << "TSB::addGroup : traj.size() = " << traj.measurements().size()
			      << " first group at " << &(*begin)
	       //        << " nr. of candidates = " << candidates.size() 
			      << endl;
  
  
  TempTrajectoryContainer updatedTrajectories; updatedTrajectories.reserve(2);
  if ( traj.measurements().empty() ) {
    vector<TM> const & firstMeasurements = unlockedMeasurements(begin->measurements());
    if ( theBestHitOnly )
      updateCandidatesWithBestHit(traj,firstMeasurements,updatedTrajectories);
    else
      updateCandidates(traj,begin->measurements(),updatedTrajectories);
    if unlikely(theDbgFlg) cout << "TSB::addGroup : updating with first group - "
				<< updatedTrajectories.size() << " trajectories" << endl;
  }
  else {
    if ( theBestHitOnly )
      updateCandidatesWithBestHit(traj,redoMeasurements(traj,begin->detGroup()),
				  updatedTrajectories);
    else
      updateCandidates(traj,redoMeasurements(traj,begin->detGroup()),
		       updatedTrajectories);
    if unlikely(theDbgFlg) cout << "TSB::addGroup : updating"
				<< updatedTrajectories.size() << " trajectories" << endl;
  }
  
  if (begin+1 != end) {
    ret.reserve(4); // a good upper bound
    for (auto const & ut : updatedTrajectories) {
      if unlikely(theDbgFlg) cout << "TSB::addGroup : trying to extend candidate at "
				  << &ut << " size " << ut.measurements().size() << endl;
      vector<TempTrajectory> finalTrajectories = addGroup(ut,begin+1,end);
      if unlikely(theDbgFlg) cout << "TSB::addGroup : " << finalTrajectories.size()
				  << " finalised candidates before cleaning" << endl;
      //B.M. to be ported later
      // V.I. only mark invalidate
      cleanCandidates(finalTrajectories);
      
      if unlikely(theDbgFlg) {
	  int ntf=0; for ( auto const & t : finalTrajectories) if (t.isValid()) ++ntf;
	  cout << "TSB::addGroup : got " << ntf
	       << " finalised candidates" << endl;
	}
      
      for ( auto & t : finalTrajectories) 
	if (t.isValid()) ret.push_back(std::move(t));
      
      //        ret.insert(ret.end(),make_move_iterator(finalTrajectories.begin()),
      //	   make_move_iterator(finalTrajectories.end()));
    }
  } else {
    ret.reserve(updatedTrajectories.size());
    for (auto & t : updatedTrajectories)
      if (!t.empty()) ret.push_back(std::move(t));
  }
  
  //std::cout << "TrajectorySegmentBuilder::addGroup" << 
  //             " traj.empty()=" << traj.empty() << 
  //             " end-begin=" << (end-begin)  <<
  //             " #updated=" << updatedTrajectories.size() << 
  //             " #result=" << ret.size() << std::endl;
  return ret;
}

void
TrajectorySegmentBuilder::updateCandidates (TempTrajectory const & traj,
					    const vector<TM>& measurements,
					    TempTrajectoryContainer& candidates)
{
  //
  // generate updated candidates with all valid hits
  //
  for ( vector<TM>::const_iterator im=measurements.begin();
	im!=measurements.end(); ++im ) {
    if ( im->recHit()->isValid() ) {
      candidates.push_back(traj);
      updateTrajectory(candidates.back(),*im);
      if ( theLockHits )  lockMeasurement(*im);
    }
  }
  //
  // keep old trajectory
  //
  candidates.push_back(traj);
}

void
TrajectorySegmentBuilder::updateCandidatesWithBestHit (TempTrajectory const& traj,
						       const vector<TM>& measurements,
						       TempTrajectoryContainer& candidates)
{
  vector<TM>::const_iterator ibest = measurements.begin();
  // get first
  while(ibest!=measurements.end() && !ibest->recHit()->isValid()) ++ibest;
  if ( ibest!=measurements.end() ) {
    // find real best;
    for ( vector<TM>::const_iterator im=ibest+1;
	  im!=measurements.end(); ++im ) {
      if ( im->recHitR().isValid() &&
	   im->estimate()<ibest->estimate()
	   )
	ibest = im;
    } 


    if ( theLockHits )  lockMeasurement(*ibest);
    candidates.push_back(traj);
    updateTrajectory(candidates.back(),*ibest);

#ifdef DBG_TSB
    if unlikely( theDbgFlg )
      cout << "TSB: found best measurement at " 
	   << ibest->recHit()->globalPosition().perp() << " "
	   << ibest->recHit()->globalPosition().phi() << " "
	   << ibest->recHit()->globalPosition().z() << endl;
#endif    
  }

  //
  // keep old trajectorTempy
  //
  candidates.push_back(traj);
}

vector<TrajectoryMeasurement>
TrajectorySegmentBuilder::redoMeasurements (const TempTrajectory& traj,
					    const DetGroup& detGroup) const
{
  vector<TM> result;
  //
  // loop over all dets
  //
  if unlikely(theDbgFlg) cout << "TSB::redoMeasurements : nr. of measurements / group =";

  tracking::TempMeasurements tmps;

  for (auto const & det : detGroup) {

    pair<bool, TrajectoryStateOnSurface> compat = 
      GeomDetCompatibilityChecker().isCompatible(det.det(),
						 traj.lastMeasurement().updatedState(),
						 theGeomPropagator,theEstimator);
    
    if unlikely(theDbgFlg && !compat.first) std::cout << " 0";

    if(!compat.first) continue;

    const MeasurementDet* mdet = theMeasurementTracker->idToDet(det.det()->geographicalId());
    // verify also first not be inactive..
    if (mdet->measurements(det.trajectoryState(), theEstimator,tmps) && tmps.hits[0]->isValid() )
      for (std::size_t i=0; i!=tmps.size(); ++i)
	result.emplace_back(det.trajectoryState(),std::move(tmps.hits[i]),tmps.distances[i],&theLayer);

    if unlikely(theDbgFlg) std::cout << " " << tmps.size();
    tmps.clear();
    
  }

  if unlikely(theDbgFlg) cout << endl;  

  std::sort( result.begin(), result.end(), TrajMeasLessEstim());

  return result;
}

void 
TrajectorySegmentBuilder::updateWithInvalidHit (TempTrajectory& traj,
						const vector<TMG>& groups,
						TempTrajectoryContainer& candidates) const
{
  //
  // first try to find an inactive hit with dets crossed by the prediction,
  //   then take any inactive hit
  //
  // loop over groups
  for ( int iteration=0; iteration<2; iteration++ ) {
    for ( vector<TMG>::const_iterator ig=groups.begin(); ig!=groups.end(); ++ig) {
      // loop over measurements
      const vector<TM>& measurements = ig->measurements();
      for ( vector<TM>::const_reverse_iterator im=measurements.rbegin();
	    im!=measurements.rend(); ++im ) {
	auto const & hit = im->recHitR();
	if ( hit.getType()==TrackingRecHit::valid ||
	     hit.getType()==TrackingRecHit::missing )  continue;
	//
	// check, if the extrapolation traverses the Det or
	// if 2nd iteration
	//
	if ( hit.det() ) {
	  auto const & predState = im->predictedState();
	  if ( iteration>0 || 
	       (predState.isValid() &&
		hit.det()->surface().bounds().inside(predState.localPosition())) ) {
	    // add the hit
	    /*TempTrajectory newTraj(traj);
	    updateTrajectory(newTraj,*im);
	    candidates.push_back(newTraj);  // FIXME: avoid useless copy */
            candidates.push_back(traj); 
            updateTrajectory(candidates.back(), *im);
	    if unlikely( theDbgFlg ) cout << "TrajectorySegmentBuilder::updateWithInvalidHit "
				  << "added inactive hit" << endl;
	    return;
	  }
	}
      }
    }
  }
  //
  // No suitable inactive hit: add a missing one
  //
  bool found(false);
  for ( int iteration=0; iteration<2; iteration++ ) {
    //
    // loop over groups
    //
    for ( vector<TMG>::const_iterator ig=groups.begin();
	  ig!=groups.end(); ++ig) {
      const vector<TM>& measurements = ig->measurements();
      for ( vector<TM>::const_reverse_iterator im=measurements.rbegin();
	    im!=measurements.rend(); ++im ) {
	//
	// only use invalid hits
	//
	auto const & hit = im->recHitR();
	if likely( hit.isValid() )  continue;

	//
	// check, if the extrapolation traverses the Det
	//
	auto const & predState = im->predictedState();
	if(hit.det()){	
	  if ( iteration>0 || (predState.isValid() &&
			       hit.det()->surface().bounds().inside(predState.localPosition())) ) {
	    // add invalid hit
	    /*TempTrajectory newTraj(traj);
	    updateTrajectory(newTraj,*im);
	    candidates.push_back(newTraj);  // FIXME: avoid useless copy */
            candidates.push_back(traj); 
            updateTrajectory(candidates.back(), *im);
	    found = true;
	    break;
	  }

	}else{
	  if ( iteration>0 || (predState.isValid() &&
			       im->layer()->surface().bounds().inside(predState.localPosition())) ){
	    // add invalid hit
	    /*TempTrajectory newTraj(traj);
	    updateTrajectory(newTraj,*im);
	    candidates.push_back(newTraj);  // FIXME: avoid useless copy */
            candidates.push_back(traj); 
            updateTrajectory(candidates.back(), *im);
	    found = true;
	    break;	    
	  }
	}
      }
      if ( found )  break;
    }
    if unlikely( theDbgFlg && !found ) cout << "TrajectorySegmentBuilder::updateWithInvalidHit: "
				    << " did not find invalid hit on 1st iteration" << endl;
    if ( found )  break;
  }

  if unlikely( theDbgFlg && (!found) )
	       cout << "TrajectorySegmentBuilder::updateWithInvalidHit: "
		    << " did not find invalid hit" << endl;
}

vector<TrajectoryMeasurement>
TrajectorySegmentBuilder::unlockedMeasurements (const vector<TM>& measurements) const
{
//   if ( !theLockHits )  return measurements;

  //========== B.M. to be ported later ===============
  vector<TM> result;
  result.reserve(measurements.size());

  //RecHitEqualByChannels recHitEqual(false,true);

  for ( auto const & m : measurements) {
    auto const & testHit = m.recHitR();
    if unlikely( !testHit.isValid() )  continue;
    bool found(false);
    if likely( theLockHits ) {
      for ( auto const & h : theLockedHits) {
	if ( h->hit()->sharesInput(testHit.hit(), TrackingRecHit::all) ) {
	  found = true;
	  break;
	}
      }
    }
    if likely( !found )  result.push_back(m);
  }
  return result;
  //================================= 
  //return measurements; // temporary solution before RecHitEqualByChannels is ported
}

void
TrajectorySegmentBuilder::lockMeasurement (const TM& measurement)
{
  theLockedHits.push_back(measurement.recHit());
}



// ================= B.M. to be ported later ===============================
void
TrajectorySegmentBuilder::cleanCandidates (vector<TempTrajectory>& candidates) const
{
  //
  // remove candidates which are subsets of others
  // assumptions: no invalid hits and no duplicates
  //
  if ( candidates.size()<=1 )  return;
  //RecHitEqualByChannels recHitEqual(false,true);
  //
  const int NC = candidates.size();
  int index[NC]; for (int i=0; i!=NC; ++i) index[i]=i;
  std::sort(index,index+NC,[&candidates](int i, int j) { return lessByFoundHits(candidates[i],candidates[j]);});
//   cout << "SortedCandidates.foundHits";
//   for (auto i1 : index) 
//     cout << " " << candidates[i1].foundHits();
//   cout << endl;
  //
  for ( auto i1 = index; i1!=index+NC-1; ++i1) {
    // get measurements of candidate to be checked
    const TempTrajectory::DataContainer & measurements1 = candidates[*i1].measurements();
    for ( auto i2=i1+1; i2!=index+NC; ++i2 ) {
      // no duplicates: two candidates of same size are different
      if ( candidates[*i2].foundHits()==candidates[*i1].foundHits() )  continue;
      // get measurements of "reference"
      const TempTrajectory::DataContainer & measurements2 = candidates[*i2].measurements();
      //
      // use the fact that TMs are ordered:
      // start search in trajectory#1 from last hit match found
      //
      bool allFound(true);
      TempTrajectory::DataContainer::const_iterator from2 = measurements2.rbegin(), im2end = measurements2.rend();
      for ( TempTrajectory::DataContainer::const_iterator im1=measurements1.rbegin(),im1end = measurements1.rend();
	    im1!=im1end; --im1 ) {
	// redundant protection - segments should not contain invalid RecHits
	// assert( im1->recHit()->isValid());
	bool found(false);
	for ( TempTrajectory::DataContainer::const_iterator im2=from2; im2!=im2end; --im2 ) {
	  // redundant protection - segments should not contain invalid RecHits
	  // assert (im2->recHit()->isValid());
	  if ( im1->recHitR().hit()->sharesInput(im2->recHitR().hit(), TrackingRecHit::all) ) {
	    found = true;
	    from2 = im2; --from2;
	    break;
	  }
	}
	if ( !found ) {
	  allFound = false;
	  break;
	}
      }
      if ( allFound ) { candidates[*i1].invalidate(); statCount.invalid();}
    }
  }

  /* will remove while coping
  candidates.erase(std::remove_if( candidates.begin(),candidates.end(),
				   [&](TempTrajectory const & t) { return !t.isValid();}),
				   // std::not1(std::mem_fun_ref(&TempTrajectory::isValid))),
 //                                boost::bind(&TempTrajectory::isValid,_1)), 
                                   candidates.end()); 
#ifdef DBG_TSB
  cout << "TSB: cleanCandidates: reduced from " << sortedCandidates.size()
       << " to " << candidates.size() << " candidates" << endl;
#endif


  */
}

//==================================================
