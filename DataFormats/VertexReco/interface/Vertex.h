#ifndef VertexReco_Vertex_h
#define VertexReco_Vertex_h
/** \class reco::Vertex
 *  
 * A reconstructed Vertex containing position and error
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
 *
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <vector>
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace reco {

  class Track;

  class Vertex {
  public:
    /// point in the space
    typedef math::XYZPoint Point;
    /// covariance error matrix (3x3)
    typedef math::Error3D Error;
    /// default constructor
    Vertex() { }
    /// constructor from values
    Vertex( const Point &, const Error &, 
	    double chi2, unsigned short ndof, size_t size );
    /// add a reference to a Track
    void add( const TrackRef & r ) { tracks_.push_back( r ); }
    /// first iterator over tracks
    track_iterator tracks_begin() const { return tracks_.begin(); }
    /// last iterator over tracks
    track_iterator tracks_end() const { return tracks_.end(); }
    /// number of tracks
    size_t tracksSize() const { return tracks_.size(); }
    /// chi-squares
    double chi2() const { return chi2_; }
    /// number of degrees of freedom
    unsigned short ndof() const { return ndof_; }
    /// chi-squared divided by n.d.o.f.
    double normalizedChi2() const { return chi2_ / ndof_; }
    /// position 
    const Point & position() const { return position_; }
    /// covariance matrix (3x3)
    const Error & error() const { return error_; }
    /// x coordinate 
    double x() const { return position_.X(); }
    /// y coordinate 
    double y() const { return position_.Y(); }
    /// y coordinate 
    double z() const { return position_.Z(); }
    /// (i, j)-th element of error matrix, i, j = 0, ... 2
    double error( int i, int j ) { return error_( i, j ); }

  private:
    /// chi-sqared
    Double32_t chi2_;
    /// number of degrees of freedom
    unsigned short ndof_;
    /// position
    Point position_;
    /// covariance matrix (3x3)
    Error error_;
    /// reference to tracks
    TrackRefs tracks_;
  };
  
}

#endif
