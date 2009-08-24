// -*- C++ -*-
//
// Package:     Core
// Class  :     prepareTrack
// $Id: prepareTrack.cc,v 1.4 2009/08/21 16:57:51 dmytro Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

// user include files
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "Fireworks/Tracks/interface/prepareTrack.h"


//
// constants, enums and typedefs
//
//
// static data member definitions
//
namespace fireworks {
  TEveTrack*
  prepareTrack(const reco::Track& track,
	       TEveTrackPropagator* propagator,
	       Color_t color,
	       const std::vector<TEveVector>& extraRefPoints)
  {
    // To make use of all available information, we have to order states
    // properly first. Propagator should take care of y=0 transition.
    
    std::vector<State> refStates;
    TEveVector trackMomentum( track.px(), track.py(), track.pz() );
    refStates.push_back(State(TEveVector(track.vertex().x(),
					 track.vertex().y(),
					 track.vertex().z()),
			      trackMomentum));
    if ( track.extra().isAvailable() ) {
      refStates.push_back(State(TEveVector( track.innerPosition().x(),
					    track.innerPosition().y(),
					    track.innerPosition().z() ),
				TEveVector( track.innerMomentum().x(),
					    track.innerMomentum().y(),
					    track.innerMomentum().z() )));
      refStates.push_back(State(TEveVector( track.outerPosition().x(),
					    track.outerPosition().y(),
					    track.outerPosition().z() ),
				TEveVector( track.outerMomentum().x(),
					    track.outerMomentum().y(),
					    track.outerMomentum().z() )));
    }
    for ( std::vector<TEveVector>::const_iterator point = extraRefPoints.begin();
	  point != extraRefPoints.end(); ++point )
      refStates.push_back(State(*point));
    std::sort( refStates.begin(), refStates.end(), StateOrdering(trackMomentum) );
    
    //
    // * if the first state has non-zero momentum use it as a starting point
    //   and all other points as PathMarks to follow
    // * if the first state has only position, try the last state. If it has
    //   momentum we propagate backword, if not, we look for the first one 
    //   on left that has momentum and ignore all earlier.
    // 
      
    TEveRecTrack t;
    t.fBeta = 1.;
    t.fSign = track.charge();
    
    if ( refStates.front().valid ){
      t.fV = refStates.front().position;
      t.fP = refStates.front().momentum;
      TEveTrack* trk = new TEveTrack(&t,propagator);
      trk->SetBreakProjectedTracks(TEveTrack::kBPTAlways);
      trk->SetMainColor(color);
      for( unsigned int i(1); i<refStates.size()-1; ++i){
	TEvePathMark mark( TEvePathMark::kDaughter );
	mark.fV = refStates[i].position;
	trk->AddPathMark( mark );
      }
      if ( refStates.size()>1 ){
	TEvePathMark mark( TEvePathMark::kDecay );
	mark.fV = refStates.back().position;
	trk->AddPathMark( mark );
      }
      return trk;
    }

    if ( refStates.back().valid ){
      t.fV = refStates.back().position;
      t.fP = refStates.back().momentum * (-1);
      TEveTrack* trk = new TEveTrack(&t,propagator);
      trk->SetBreakProjectedTracks(TEveTrack::kBPTAlways);
      trk->SetMainColor(color);
      unsigned int i(refStates.size()-1);
      for(; i>0; --i){
	TEvePathMark mark( TEvePathMark::kDaughter );
	mark.fV = refStates[i].position;
	trk->AddPathMark( mark );
      }
      if ( refStates.size()>1 ){
	TEvePathMark mark( TEvePathMark::kDecay );
	mark.fV = refStates.front().position;
	trk->AddPathMark( mark );
      }
      return trk;
    }
    
    unsigned int i(0);
    while ( i<refStates.size() && !refStates[i].valid ) ++i;
    assert ( i < refStates.size() );

    t.fV = refStates[i].position;
    t.fP = refStates[i].momentum;
    TEveTrack* trk = new TEveTrack(&t,propagator);
    trk->SetBreakProjectedTracks(TEveTrack::kBPTAlways);
    trk->SetMainColor(color);
    
    for( unsigned int j(i+1); j<refStates.size()-1; ++j){
      TEvePathMark mark( TEvePathMark::kDaughter );
      mark.fV = refStates[j].position;
      trk->AddPathMark( mark );
    }
    if ( i < refStates.size() ){
      TEvePathMark mark( TEvePathMark::kDecay );
      mark.fV = refStates.back().position;
      trk->AddPathMark( mark );
    }
    return trk;
  }
  TEveTrack*
   prepareTrack(const reco::Candidate& track,
		TEveTrackPropagator* propagator,
		Color_t color)
  {
    TEveRecTrack t;
    t.fBeta = 1.;
    t.fP = TEveVector( track.px(), track.py(), track.pz() );
    t.fV = TEveVector( track.vertex().x(), track.vertex().y(), track.vertex().z() );
    t.fSign = track.charge();
    TEveTrack* trk = new TEveTrack(&t,propagator);
    trk->SetMainColor(color);
    return trk;
  }
}
