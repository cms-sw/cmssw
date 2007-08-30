#include "DataFormats/VertexReco/interface/Vertex.h"

// $Id: Vertex.cc,v 1.11 2007/05/30 07:47:38 llista Exp $
using namespace reco;
using namespace std;

Vertex::Vertex( const Point & p , const Error & err, double chi2, double ndof, size_t size ) :
  chi2_( chi2 ), ndof_( ndof ), position_( p ) {
  tracks_.reserve( size );
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      covariance_[ idx ++ ] = err( i, j );
}

void Vertex::fill( Error & err ) const {
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      err( i, j ) = covariance_[ idx ++ ];
}

size_t Vertex::tracksSize() const
{
  return weights_.size();
}

Vertex::trackRef_iterator Vertex::tracks_begin() const
{
  return tracks_.begin();
}

Vertex::trackRef_iterator Vertex::tracks_end() const
{
//   if ( !(tracks_.size() ) ) createTracks();
  return tracks_.end();
  // return weights_.keys().end();
}

void Vertex::add ( const TrackRef & r, float w )
{
  tracks_.push_back ( r );
  weights_.push_back(w);

}

void Vertex::add ( const TrackRef & r, const Track & refTrack, float w )
{
  tracks_.push_back ( r );
  refittedTracks_.push_back ( refTrack );
  weights_.push_back(w);
}

void Vertex::removeTracks()
{
  weights_.clear();
  tracks_.clear();
  refittedTracks_.clear();
}

float Vertex::trackWeight ( const TrackRef & track ) const
{
  trackRef_iterator it = find(tracks_begin(), tracks_end(), track);
  if (it==tracks_end()) return 0.0;
  size_t pos = it - tracks_begin();
  return weights_[pos];
}

TrackRef Vertex::originalTrack(const Track & refTrack) const
{
  if (refittedTracks_.empty())
	throw cms::Exception("Vertex") << "No refitted tracks stored in vertex\n";
  std::vector<Track>::const_iterator it =
	find_if(refittedTracks_.begin(), refittedTracks_.end(), TrackEqual(refTrack));
  if (it==refittedTracks_.end())
	throw cms::Exception("Vertex") << "Refitted track not found in list\n";
  size_t pos = it - refittedTracks_.begin();
  return tracks_[pos];
}

Track Vertex::refittedTrack(const TrackRef & track) const
{
  if (refittedTracks_.empty())
	 throw cms::Exception("Vertex") << "No refitted tracks stored in vertex\n";
  trackRef_iterator it = find(tracks_begin(), tracks_end(), track);
  if (it==tracks_end()) throw cms::Exception("Vertex") << "Track not found in list\n";
  size_t pos = it - tracks_begin();
  return refittedTracks_[pos];
}
