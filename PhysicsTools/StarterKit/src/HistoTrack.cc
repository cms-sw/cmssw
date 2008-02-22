#include "PhysicsTools/StarterKit/interface/HistoTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <iostream>
#include <sstream>

using pat::HistoTrack;
using reco::RecoChargedCandidate;
using namespace std;

// Constructor:


HistoTrack::HistoTrack(std::string dir, std::string groupName, std::string groupLabel,
		       double pt1, double pt2, double m1, double m2)
  : HistoGroup<RecoChargedCandidate>( dir, groupName, groupLabel, pt1, pt2, m1, m2)
{
  addHisto( h_dxy_ =
	    new PhysVarHisto( "trkDxy", "Track Impact Parameter, x-y", 20, -3, 3, currDir_, "", "vD" )
	   );
  addHisto( h_dz_ =
	    new PhysVarHisto( "trkDz", "Track Impact Parameter, z", 20, -3, 3, currDir_, "", "vD" )
	   );
  addHisto( h_nValid_ =
	    new PhysVarHisto( "trkNValid", "Number of Valid hits", 20, 0, 20, currDir_, "", "vD" )
	   );
  addHisto( h_nLost_ =
	    new PhysVarHisto( "trkNLost", "Number of Lost hits", 20, 0, 20, currDir_, "", "vD" )
	   );

}



void HistoTrack::fill( const RecoChargedCandidate *track, uint iTrk )
{

  // First fill common 4-vector histograms

  HistoGroup<RecoChargedCandidate>::fill( track, iTrk);

  // fill relevant track histograms
  h_dxy_->fill( track->track()->dxy(), iTrk );
  h_dz_->fill( track->track()->dsz(), iTrk );
  h_nValid_->fill( track->track()->numberOfValidHits(), iTrk);
  h_nLost_->fill( track->track()->numberOfLostHits(), iTrk);
}


void HistoTrack::clearVec()
{
  HistoGroup<reco::RecoChargedCandidate>::clearVec();

  h_dxy_->clearVec();
  h_dz_->clearVec();
  h_nValid_->clearVec();
  h_nLost_->clearVec();
}
