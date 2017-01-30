/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*   Nicola Minafra (nicola.minafra@cern.ch)
*
****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSDiamondRecHit
#define DataFormats_CTPPSReco_CTPPSDiamondRecHit

#include "DataFormats/CTPPSDigi/interface/HPTDCErrorFlags.h"

/// Reconstructed hit in diamond detectors.
class CTPPSDiamondRecHit
{
  public:
    CTPPSDiamondRecHit() :
      x_( 0. ), x_width_( 0. ), y_( 0. ), y_width_( 0. ),
      t_( 0. ), tot_( 0. ),
      ts_index_( 0 ), hptdc_err_( 0 )
    {}
    CTPPSDiamondRecHit( double x, double x_width, double y, double y_width, double t, double tot, int oot_idx, const HPTDCErrorFlags& hptdc_err ) :
      x_( x ), x_width_( x_width ), y_( y ), y_width_( y_width ),
      t_( t ), tot_( tot ),
      ts_index_( oot_idx ), hptdc_err_( hptdc_err )
    {}

    inline void setX( const double& x ) { x_ = x; }
    inline double getX() const { return x_; }

    inline void setY( const double& y ) { y_ = y; }
    inline double getY() const { return y_; }

    inline void setXWidth( const double& xwidth ) { x_width_ = xwidth; }
    inline double getXWidth() const { return x_width_; }

    inline void setYWidth( const double& ywidth ) { y_width_ = ywidth; }
    inline double getYWidth() const { return y_width_; }

    inline void setT( const double& t ) { t_ = t; }
    inline double getT() const { return t_; }

    inline void setToT( const double& tot ) { tot_ = tot;  }
    inline double getToT() const { return tot_; }

    inline void setOOTIndex( const int& i ) { ts_index_ = i; }
    inline int getOOTIndex() const { return ts_index_; }

    inline void setHPTDCErrorFlags( const HPTDCErrorFlags& err ) { hptdc_err_ = err; }
    inline HPTDCErrorFlags getHPTDCErrorFlags() const { return hptdc_err_; }

  private:
    double x_, x_width_;
    double y_, y_width_;
    double t_, tot_;
    /// Time slice index
    int ts_index_;
    HPTDCErrorFlags hptdc_err_;
};

//----------------------------------------------------------------------------------------------------

inline bool operator<( const CTPPSDiamondRecHit& l, const CTPPSDiamondRecHit& r )
{
  // only sort by leading edge time
  return ( l.getT() < r.getT() );
}

#endif
