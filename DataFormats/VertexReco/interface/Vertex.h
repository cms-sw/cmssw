#ifndef VertexReco_Vertex_h
#define VertexReco_Vertex_h
//
// $Id: Vertex.h,v 1.7 2005/11/21 12:55:16 llista Exp $
//
// RECO Vertex class
//
#include <Rtypes.h>
#include "DataFormats/TrackReco/interface/Error.h"
#include "DataFormats/TrackReco/interface/Vector.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefVector.h"
#include <vector>

namespace reco {

  class Track;

  class Vertex {
  public:
    typedef Vector3D Point;
    typedef Error3D Error;
    typedef edm::Ref<std::vector<Track> > TrackRef;
    typedef edm::RefVector<std::vector<Track> > TrackRefs;
    typedef TrackRefs::iterator tracks_iterator;
    Vertex() { }
    Vertex( double chi2, unsigned short ndof, 
	    double x, double y, double z, const Error & err, 
	    size_t size );
    void add( const TrackRef & r ) { tracks_.push_back( r ); }
    tracks_iterator tracks_begin() const { return tracks_.begin(); }
    tracks_iterator tracks_end() const { return tracks_.end(); }
    size_t tracksSize() const { return tracks_.size(); }
    double chi2() const { return chi2_; }
    unsigned short ndof() const { return ndof_; }
    double normalizedChi2() const { return chi2_ / ndof_; }
    const Point & position() const { return position_; }
    const Error & error() const { return error_; }
    double x() const { return position_.get< 0 >(); }
    double y() const { return position_.get< 1 >(); }
    double z() const { return position_.get< 2 >(); }
  private:
    Double32_t chi2_;
    unsigned short ndof_;
    Point position_;
    Error error_;
    TrackRefs tracks_;
  };
  
}

#endif
