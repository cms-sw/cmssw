#ifndef VertexReco_Vertex_h
#define VertexReco_Vertex_h
/** \class reco::Vertex
 *  
 * A reconstructed Vertex providing position, error, chi2, ndof 
 * and reconstrudted tracks
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Vertex.h,v 1.20 2006/09/17 13:25:11 vanlaer Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include <iostream>

namespace reco {

  class Track;

  class Vertex {
  public:
    /// point in the space
    typedef math::XYZPoint Point;
    /// error matrix dimension
    enum { dimension = 3 };
    /// covariance error matrix (3x3)
    typedef math::Error<dimension>::type Error;
    /// covariance error matrix (3x3)
    typedef math::Error<dimension>::type CovarianceMatrix;
    /// matix size
    enum { size = dimension * ( dimension + 1 ) / 2 };
    /// index type
    typedef unsigned int index;
    /// default constructor
    Vertex() { }
    /// constructor from values
    Vertex( const Point &, const Error &, double chi2, double ndof, size_t size );
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
    /** Number of degrees of freedom
     *  Meant to be Double32_t for soft-assignment fitters: 
     *  tracks may contribute to the vertex with fractional weights.
     *  The ndof is then = to the sum of the track weights.
     *  see e.g. CMS NOTE-2006/032, CMS NOTE-2004/002
     */
    double ndof() const { return ndof_; }
    /// chi-squared divided by n.d.o.f.
    double normalizedChi2() const { return chi2_ / ndof_; }
    /// position 
    const Point & position() const { return position_; }
    /// x coordinate 
    double x() const { return position_.X(); }
    /// y coordinate 
    double y() const { return position_.Y(); }
    /// y coordinate 
    double z() const { return position_.Z(); }
    /// error on x
    double xError() const { return sqrt( covariance(0, 0) ); }
    /// error on y
    double yError() const { return sqrt( covariance(1, 1) ); }
    /// error on z
    double zError() const { return sqrt( covariance(2, 2) ); }
    /// (i, j)-th element of error matrix, i, j = 0, ... 2
    double error( int i, int j ) const {
      std::cout << "reco::Vertex::error(i, j) OBSOLETE, use covariance(i, j)"
		<< std::endl;
      return covariance_[ idx( i, j ) ]; 
    }
    /// (i, j)-th element of error matrix, i, j = 0, ... 2
    double & error( int i, int j ) { 
      std::cout << "reco::Vertex::error(i, j) & NON-CONST, use covariance(i, j)"
		<< std::endl;
      return covariance_[ idx( i, j ) ]; 
    }
    /// (i, j)-th element of error matrix, i, j = 0, ... 2
    double & covariance( int i, int j ) {
      std::cout << "reco::Vertex::covariance(i, j) & NON-CONST, use covariance(i, j)" << std::endl;
      return covariance_[ idx( i, j ) ];
    }
    double covariance( int i, int j ) const { 
      return covariance_[ idx( i, j ) ]; 
    }
    /// return SMatrix
    CovarianceMatrix covariance() const { Error m; fill( m ); return m; }
    /// return SMatrix
    Error error() const { Error m; fill( m ); return m; }
    /// fill SMatrix
    void fill( CovarianceMatrix & v ) const;

  private:
    /// chi-sqared
    Double32_t chi2_;
    /// number of degrees of freedom
    Double32_t ndof_;
    /// position
    Point position_;
    /// covariance matrix (3x3) as vector
    Double32_t covariance_[ size ];
    /// reference to tracks
    TrackRefVector tracks_;
    /// position index
    index idx( index i, index j ) const {
      int a = ( i <= j ? i : j ), b = ( i <= j ? j : i );
      return b * ( b + 1 ) / 2 + a;
    }
  };
  
}

#endif
