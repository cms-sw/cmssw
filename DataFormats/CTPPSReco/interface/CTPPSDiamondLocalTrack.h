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
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

//----------------------------------------------------------------------------------------------------

class CTPPSDiamondLocalTrack
{
  public:
    CTPPSDiamondLocalTrack() :
      tx_( 0. ), tx_sigma_( 0. ),
      ty_( 0. ), ty_sigma_( 0. ),
      chi_squared_( 0. ),
      valid_( false ) {;}
    virtual ~CTPPSDiamondLocalTrack() {;}

    //--- spatial get'ters

    inline double getX0() const { return pos0_.x(); }
    inline double getX0Sigma() const { return x0_sigma_; }

    inline double getY0() const { return pos0_.y(); }
    inline double getY0Sigma() const { return y0_sigma_; }

    inline double getZ0() const { return pos0_.z(); }

    inline double getTx() const { return tx_; }
    inline double getTxSigma() const { return tx_sigma_; }

    inline double getTy() const { return ty_; }
    inline double getTySigma() const { return ty_sigma_; }

    inline double getChiSquared() const { return chi_squared_; }
    inline void setChiSquared( const double& chisq ) { chi_squared_ = chisq; }

    //inline double getChiSquaredOverNDF() const { return chiSquared_ / (track_hits_vector_.size() - 4); }

    //FIXME include timing validity checks too?
    inline bool isValid() const { return valid_; }
    inline void setValid(bool valid) { valid_ = valid; }

    //--- temporal get'ters

    inline double getT() const { return t_; }
    inline double getTSigma() const { return t_sigma_; }

    friend bool operator<( const CTPPSDiamondLocalTrack&, const CTPPSDiamondLocalTrack& );

  private:
    //FIXME placeholder variables only!
    // replace us with values computed from tracking algorithm

    //--- spatial information

    math::XYZPoint pos0_;

    double x0_sigma_, y0_sigma_;
    // track angular information (FIXME: move to a momentum_ 3-vector?)
    double tx_, tx_sigma_;
    double ty_, ty_sigma_;

    /// fit chi^2
    double chi_squared_;

    /// fit valid?
    bool valid_;

    //--- timing information

    double t_;
    double t_sigma_;

};

#endif
