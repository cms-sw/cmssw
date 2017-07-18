/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef DataFormats_ProtonReco_ProtonTrack_h
#define DataFormats_ProtonReco_ProtonTrack_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include <set>

/**
 * FIXME make use of the general reco::Candidate object, with appropriate set'ters and get'ters
 */

namespace reco
{
  class ProtonTrack
  {
    public:
      ProtonTrack() :
        xi_( 0. ), xi_unc_( 0. ),
        isValid_( false ) {}
      ProtonTrack( const Local3DPoint& vtx, const Local3DVector& dir, float xi, float xi_unc=0. ) :
        vertex_( vtx ), direction_( dir ), xi_( xi ), xi_unc_( xi_unc ),
        isValid_( true ) {}
      ~ProtonTrack() {}

      void setVertex( const Local3DPoint& vtx ) { vertex_ = vtx; }
      const Local3DPoint& vertex() const { return vertex_; }

      void setDirection( const Local3DVector& dir ) { direction_ = dir; }
      const Local3DVector& direction() const { return direction_; }

      void setXi( float xi ) { xi_ = xi; }
      float xi() const { return xi_; }

      void setValid( bool valid=true ) { isValid_ = valid; }
      bool valid() const { return isValid_; }

      // TODO: add proper getters, setters
      enum { rmSingleRP, rmMultipleRP } method;

      enum { sector45, sector56 } lhcSector;

      double fitChiSq;

      std::set<unsigned int> contributingRPIds;

    private:

      // TODO: describe, mention CMS coordinate notation
      Local3DPoint vertex_;
      Local3DVector direction_;

      // TODO: describe
      float xi_;
      float xi_unc_;

      // TODO: rename to fit valid?
      bool isValid_;
  };
}

#endif
