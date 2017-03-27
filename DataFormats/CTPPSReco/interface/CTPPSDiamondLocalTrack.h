/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSDiamondLocalTrack
#define DataFormats_CTPPSReco_CTPPSDiamondLocalTrack

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

//----------------------------------------------------------------------------------------------------

class CTPPSDiamondLocalTrack
{
  public:
    CTPPSDiamondLocalTrack() :
      chi_squared_( 0. ), valid_( true ), t_( 0. ), t_sigma_( 0. ), ts_index_( 0 ), mh_( 0 ) {}
    CTPPSDiamondLocalTrack( const math::XYZPoint& pos0, const math::XYZPoint& pos0_sigma, float chisq, float t, float t_sigma, int oot_idx, int mult_hits ) :
      pos0_( pos0 ), pos0_sigma_( pos0_sigma ),
      chi_squared_( chisq ), valid_( false ),
      t_( t ), t_sigma_( t_sigma ), ts_index_( oot_idx ), mh_( mult_hits ) {}
    virtual ~CTPPSDiamondLocalTrack() {}

    //--- spatial get'ters

    inline float getX0() const { return pos0_.x(); }
    inline float getX0Sigma() const { return pos0_sigma_.x(); }

    inline float getY0() const { return pos0_.y(); }
    inline float getY0Sigma() const { return pos0_sigma_.y(); }

    inline float getZ0() const { return pos0_.z(); }

    inline float getChiSquared() const { return chi_squared_; }
    
    //--- spatial set'ters

    inline void setPosition( const math::XYZPoint& pos0 ) { pos0_ = pos0; }
    inline void setPositionSigma( const math::XYZPoint& pos0_sigma ) { pos0_sigma_ = pos0_sigma; }

    inline void setChiSquared( const float chisq ) { chi_squared_ = chisq; }

    inline bool isValid() const { return valid_; }
    inline void setValid( bool valid ) { valid_ = valid; }

    //--- temporal get'ters

    inline float getT() const { return t_; }
    inline float getTSigma() const { return t_sigma_; }
    
    //--- temporal set'ters

    inline void setT( const float t ) { t_ = t; }
    inline void setTSigma( const float t_sigma ) { t_sigma_ = t_sigma; }
    
    inline void setOOTIndex( const int i ) { ts_index_ = i; }
    inline int getOOTIndex() const { return ts_index_; }
    
    inline void setMultipleHits( const int i ) { mh_ = i; }
    inline int getMultipleHits() const { return mh_; }

  private:
    //--- spatial information

    /// initial track position
    math::XYZPoint pos0_;
    /// error on the initial track position
    math::XYZPoint pos0_sigma_;

    /// fit chi^2
    float chi_squared_;

    /// fit valid?
    bool valid_;

    //--- timing information

    float t_;
    float t_sigma_;
    /// Time slice index
    int ts_index_;
    /// Multiple hits counter
    int mh_;

};

inline bool operator<( const CTPPSDiamondLocalTrack& lhs, const CTPPSDiamondLocalTrack& rhs )
{
  // start to sort by temporal coordinate
  if ( lhs.getT() < rhs.getT() ) return true;
  if ( lhs.getT() > rhs.getT() ) return false;
  // then sort by x-position
  return ( lhs.getX0() < rhs.getX0() );
}

#endif
