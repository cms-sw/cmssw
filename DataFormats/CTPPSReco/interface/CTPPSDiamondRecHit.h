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

/// Reconstructed hit in diamond detectors.
class CTPPSDiamondRecHit
{
  public:
    CTPPSDiamondRecHit() :
      x_( 0. ), x_width_( 0. ), y_( 0. ), y_width_( 0. ),
      t_( 0. ), tot_( 0. ),
      ts_index_( 0 ) {;}
    CTPPSDiamondRecHit(double x, double x_width, double y, double y_width, double t, double tot, int oot_idx) :
      x_( x ), x_width_( x_width ), y_( y ), y_width_( y_width ),
      t_( t ), tot_( tot ),
      ts_index_( oot_idx ) {;}

    inline double getX() const { return x_; }
    inline double getXWidth() const { return x_width_; }

    inline double getY() const { return y_; }
    inline double getYWidth() const { return y_width_; }

    inline double getT() const { return t_; }
    inline double getToT() const { return tot_; }

    inline int getOOTIndex() const { return ts_index_; }

  private:
    double x_, x_width_;
    double y_, y_width_;
    double t_, tot_;
    /// Time slice index
    int ts_index_;
};

//----------------------------------------------------------------------------------------------------

inline bool operator<( const CTPPSDiamondRecHit& l, const CTPPSDiamondRecHit& r )
{
  //FIXME only sort by leading edge time?
  return ( l.getT() < r.getT() );
}

#endif
