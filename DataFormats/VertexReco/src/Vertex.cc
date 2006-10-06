#include "DataFormats/VertexReco/interface/Vertex.h"
// $Id: Vertex.cc,v 1.6 2006/07/20 14:34:40 llista Exp $
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

track_iterator Vertex::tracks_begin() const
{
  if ( !(tracks_.size() ) ) createTracks();
  return tracks_.begin();
  // return weights_.keys().begin();
}

void Vertex::createTracks() const
{
  for ( edm::AssociationMap < edm::OneToValue<reco::TrackCollection, double> >::const_iterator 
        i=weights_.begin(); i!=weights_.end() ; ++i )
  {
    tracks_.push_back ( i->key );
  }
}

track_iterator Vertex::tracks_end() const
{
  if ( !(tracks_.size() ) ) createTracks();
  return tracks_.end();
  // return weights_.keys().end();
}

void Vertex::add ( const TrackRef & r, double w )
{
  // tracks_.push_back ( r ); // FIXME should be removed
  weights_.insert(r,w);
}

void Vertex::removeTracks()
{
  weights_.clear();
  tracks_.clear();
}

double Vertex::trackWeight ( const TrackRef & r ) const
{
  return weights_.find(r)->val;
}
