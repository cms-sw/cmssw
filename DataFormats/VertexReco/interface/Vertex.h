#ifndef VertexReco_Vertex_h
#define VertexReco_Vertex_h
//
// $Id: Vertex.h,v 1.3 2005/12/15 20:42:51 llista Exp $
//
// RECO Vertex class
//
#include <Rtypes.h>
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/EDProduct/interface/RefVector.h"
#include <vector>
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace reco {

  class Track;

  class Vertex {
  public:
    typedef math::XYZPoint Point;
    typedef math::Error3D Error;
    Vertex() { }
    Vertex( const Point &, const Error &, 
	    double chi2, unsigned short ndof, size_t size );
    void add( const TrackRef & r ) { tracks_.push_back( r ); }
    track_iterator tracks_begin() const { return tracks_.begin(); }
    track_iterator tracks_end() const { return tracks_.end(); }
    size_t tracksSize() const { return tracks_.size(); }
    double chi2() const { return chi2_; }
    unsigned short ndof() const { return ndof_; }
    double normalizedChi2() const { return chi2_ / ndof_; }
    const Point & position() const { return position_; }
    const Error & error() const { return error_; }
    double x() const { return position_.X(); }
    double y() const { return position_.Y(); }
    double z() const { return position_.Z(); }
    double error( int i, int j ) { return error_( i, j ); }

  private:
    Double32_t chi2_;
    unsigned short ndof_;
    Point position_;
    Error error_;
    TrackRefs tracks_;
  };
  
}

#endif
