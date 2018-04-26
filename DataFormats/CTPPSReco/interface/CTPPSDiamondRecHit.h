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

#include "DataFormats/CTPPSReco/interface/CTPPSTimingRecHit.h"
#include "DataFormats/CTPPSDigi/interface/HPTDCErrorFlags.h"

/// Reconstructed hit in diamond detectors.
class CTPPSDiamondRecHit : public CTPPSTimingRecHit
{
  public:
    CTPPSDiamondRecHit() :
      CTPPSTimingRecHit(),
      ts_index_( 0 ), hptdc_err_( 0 ), mh_( false )
    {}
    CTPPSDiamondRecHit( float x, float x_width, float y, float y_width, float z, float z_width, float t, float tot, float t_precision, int oot_idx, const HPTDCErrorFlags& hptdc_err, const bool mh ) :
      CTPPSTimingRecHit( x, x_width, y, y_width, z, z_width, t, tot, t_precision ),
      ts_index_( oot_idx ), hptdc_err_( hptdc_err ), mh_( mh )
    {}
    
    static constexpr int TIMESLICE_WITHOUT_LEADING = -10;
    
    inline void setOOTIndex( const int& i ) { ts_index_ = i; }
    inline int getOOTIndex() const { return ts_index_; }

    inline void setMultipleHits( const bool mh ) { mh_ = mh; }
    inline bool getMultipleHits() const { return mh_; }

    inline void setHPTDCErrorFlags( const HPTDCErrorFlags& err ) { hptdc_err_ = err; }
    inline HPTDCErrorFlags getHPTDCErrorFlags() const { return hptdc_err_; }

  private:
    /// Time slice index
    int ts_index_;
    HPTDCErrorFlags hptdc_err_;
    bool mh_;
};

#endif

