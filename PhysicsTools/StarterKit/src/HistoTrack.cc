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
	    new PhysVarHisto( groupLabel + "Dxy", "Track Impact Parameter, x-y", 20, -3, 3, currDir_, "", "vD" )
	   );
  addHisto( h_dz_ =
	    new PhysVarHisto( groupLabel + "Dz", "Track Impact Parameter, z", 20, -3, 3, currDir_, "", "vD" )
	   );
  addHisto( h_nValid_ =
	    new PhysVarHisto( groupLabel + "NValid", "Number of Valid hits", 20, 0, 20, currDir_, "", "vD" )
	   );
  addHisto( h_nLost_ =
	    new PhysVarHisto( groupLabel + "NLost", "Number of Lost hits", 20, 0, 20, currDir_, "", "vD" )
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


void HistoTrack::fill( const reco::ShallowCloneCandidate * pshallow, uint iTrk )
{


  // Get the underlying object that the shallow clone represents
  const reco::RecoChargedCandidate * track = dynamic_cast<const reco::RecoChargedCandidate*>(pshallow);

  if ( track == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a track" << endl;
    return;
  }

  

  // First fill common 4-vector histograms from shallow clone

  HistoGroup<reco::RecoChargedCandidate>::fill( pshallow, iTrk);

  // fill relevant track histograms
  h_dxy_->fill( track->track()->dxy(), iTrk );
  h_dz_->fill( track->track()->dsz(), iTrk );
  h_nValid_->fill( track->track()->numberOfValidHits(), iTrk);
  h_nLost_->fill( track->track()->numberOfLostHits(), iTrk);
}


void HistoTrack::fillCollection( const std::vector<RecoChargedCandidate> & coll ) 
{
 
  h_size_->fill( coll.size() );     //! Save the size of the collection.

  std::vector<RecoChargedCandidate>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoTrack::clearVec()
{
  HistoGroup<reco::RecoChargedCandidate>::clearVec();

  h_dxy_->clearVec();
  h_dz_->clearVec();
  h_nValid_->clearVec();
  h_nLost_->clearVec();
}
