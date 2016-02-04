#include "DataFormats/VertexReco/interface/Vertex.h"
#include <Math/GenVector/PxPyPzE4D.h>                                                                                                   
#include <Math/GenVector/PxPyPzM4D.h>       

// $Id: Vertex.cc,v 1.20 2011/03/03 05:25:19 dlange Exp $
using namespace reco;
using namespace std;

Vertex::Vertex( const Point & p , const Error & err, double chi2, double ndof, size_t size ) :
  chi2_( chi2 ), ndof_( ndof ), position_( p ) {
  tracks_.reserve( size );
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      covariance_[ idx ++ ] = err( i, j );
  validity_ = true;
}

Vertex::Vertex( const Point & p , const Error & err) :
  chi2_( 0.0 ), ndof_( 0 ), position_( p ) {
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      covariance_[ idx ++ ] = err( i, j );
  validity_ = true;
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

void Vertex::add ( const TrackBaseRef & r, float w )
{
  tracks_.push_back ( r );
  weights_.push_back(w*255);
}

void Vertex::add ( const TrackBaseRef & r, const Track & refTrack, float w )
{
  tracks_.push_back ( r );
  refittedTracks_.push_back ( refTrack );
  weights_.push_back(w*255);
}

void Vertex::removeTracks()
{
  weights_.clear();
  tracks_.clear();
  refittedTracks_.clear();
}

float Vertex::trackWeight ( const TrackBaseRef & track ) const
{
  trackRef_iterator it = find(tracks_begin(), tracks_end(), track);
  if (it==tracks_end()) return 0.0;
  size_t pos = it - tracks_begin();
  return weights_[pos]/255.;
}

float Vertex::trackWeight ( const TrackRef & track ) const
{
  return trackWeight(TrackBaseRef(track));
}


TrackBaseRef Vertex::originalTrack(const Track & refTrack) const
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

Track Vertex::refittedTrack(const TrackBaseRef & track) const
{
  if (refittedTracks_.empty())
	 throw cms::Exception("Vertex") << "No refitted tracks stored in vertex\n";
  trackRef_iterator it = find(tracks_begin(), tracks_end(), track);
  if (it==tracks_end()) throw cms::Exception("Vertex") << "Track not found in list\n";
  size_t pos = it - tracks_begin();
  return refittedTracks_[pos];
}

Track Vertex::refittedTrack(const TrackRef & track) const
{
  return refittedTrack(TrackBaseRef(track));
}

math::XYZTLorentzVectorD Vertex::p4(float mass,float minWeight) const
{

 math::XYZTLorentzVectorD sum;
 ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > vec;

 if(hasRefittedTracks()) {
 for(std::vector<Track>::const_iterator iter = refittedTracks_.begin();
     iter != refittedTracks_.end(); ++iter) {
   if (trackWeight(originalTrack(*iter)) >=minWeight) {
   vec.SetPx(iter->px());
   vec.SetPy(iter->py());
   vec.SetPz(iter->pz());
   vec.SetM(mass);
   sum += vec;
   }
  }
 }
 else
 {
 for(std::vector<reco::TrackBaseRef>::const_iterator iter = tracks_begin();
            iter != tracks_end(); iter++) {
  if (trackWeight(*iter) >=minWeight) {
   vec.SetPx((*iter)->px());
   vec.SetPy((*iter)->py());
   vec.SetPz((*iter)->pz());
   vec.SetM(mass);
   sum += vec;
   }
  }
 }
 return sum;
}

unsigned int Vertex::nTracks(float minWeight) const
{
 int n=0;
 if(hasRefittedTracks()) {
 for(std::vector<Track>::const_iterator iter = refittedTracks_.begin(); iter != refittedTracks_.end(); ++iter) 
   if (trackWeight(originalTrack(*iter)) >=minWeight) 
     n++;
 }
 else
 {
  for(std::vector<reco::TrackBaseRef>::const_iterator iter = tracks_begin(); iter != tracks_end(); iter++) 
   if (trackWeight(*iter) >=minWeight) 
    n++;  
 } 
 return n;
}

