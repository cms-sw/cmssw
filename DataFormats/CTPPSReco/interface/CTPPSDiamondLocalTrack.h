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
      chi_squared_( 0. ), valid_( true ), t_( 0. ), t_sigma_( 0. ) {}
    virtual ~CTPPSDiamondLocalTrack() {}

    //--- spatial get'ters

    inline float getX0() const { return pos0_.x(); }
    inline float getX0Sigma() const { return pos0_sigma_.x(); }

    inline float getY0() const { return pos0_.y(); }
    inline float getY0Sigma() const { return pos0_sigma_.y(); }

    inline float getZ0() const { return pos0_.z(); }

    inline float getChiSquared() const { return chi_squared_; }
    
    //--- spatial set'ters

    inline void setX0( const float& x0 ) { pos0_.SetX( x0 ); }
    inline void setX0Sigma( const float& x0_sigma ) { pos0_sigma_.SetX( x0_sigma ); }

    inline void setY0( const float& y0 ) { pos0_.SetY( y0 ); }
    inline void setY0Sigma( const float& y0_sigma ) { pos0_sigma_.SetY( y0_sigma ); }

    inline void setZ0( const float& z ) { pos0_.SetZ(z); }

    inline void setChiSquared( const float& chisq ) { chi_squared_ = chisq; }

    inline bool isValid() const { return valid_; }
    inline void setValid( bool valid ) { valid_ = valid; }

    //--- temporal get'ters

    inline float getT() const { return t_; }
    inline float getTSigma() const { return t_sigma_; }
    
    //--- temporal set'ters

    inline void setT( const float& t) { t_ = t; }
    inline void setTSigma( const float& t_sigma ) { t_sigma_ = t_sigma; }

    friend bool operator<( const CTPPSDiamondLocalTrack&, const CTPPSDiamondLocalTrack& );

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

};

#endif
