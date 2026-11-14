#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelPhase2Topology.h"

/**
   * Topology for rectangular pixel detector with BIG pixels.
   */
// Modified for the large pixles.
// 2023/11 and 2025/07 - Re-written for phase-2 pixels (IT planar and OT PS-p).
// G.Bardelli (UniFI/INFN)
// F.Luongo (UniTO/INFN)
// E.Migliore (UniTO/INFN)

//--------------------------------------------------------------------
// PixelTopology interface.
// Transform LocalPoint in cm to measurement in pitch units.
std::pair<float, float> RectangularPixelPhase2Topology::pixel(const LocalPoint& p) const {
  // check limits
  float py = p.y();
  float px = p.x();

#ifdef EDM_ML_DEBUG
#define EPSCM 0
#define EPSDBG 1e-5
  // This will catch points which are outside the active sensor area.
  // In the digitizer during the early induce_signal phase non valid
  // location are passed here. They are cleaned later.

  std::ostringstream debugstr;
  debugstr << "py = " << py << ", m_yoffset = " << m_yoffset << "px = " << px << ", m_xoffset = " << m_xoffset << "\n";

  if (py < m_yoffset)  // m_yoffset is negative
  {
    debugstr << " wrong lp y " << py << " " << m_yoffset << "\n";
    py = m_yoffset + EPSCM;  // make sure it is in, add an EPS in cm
  }
  if (py > -m_yoffset) {
    debugstr << " wrong lp y " << py << " " << -m_yoffset << "\n";
    py = -m_yoffset - EPSCM;
  }
  if (px < m_xoffset)  // m_xoffset is negative
  {
    debugstr << " wrong lp x " << px << " " << m_xoffset << "\n";
    px = m_xoffset + EPSCM;
  }
  if (px > -m_xoffset) {
    debugstr << " wrong lp x " << px << " " << -m_xoffset << "\n";
    px = -m_xoffset - EPSCM;
  }

  if (!debugstr.str().empty())
    LogDebug("RectangularPixelPhase2Topology") << debugstr.str();
#endif  // EDM_ML_DEBUG

  float mpY = ConvertPixelToPitch(py, m_COLS_PER_ROC, m_ROCS_Y, m_yoffset, m_pitchy, m_BIG_PIX_PER_ROC_Y, m_BIG_PIX_PITCH_Y, m_BIG_PIX_TWOEDGES_Y);

  /*
#ifdef EDM_ML_DEBUG
  if (iybin0 > m_COLS_PER_ROC) {
    LogDebug("RectangularPixelPhase2Topology") << " very bad, newbiny " << iybin0 << "\n"
                                               << py << " " << m_yoffset << " " << m_pitchy << " " << newybin << " "
                                               << iybin << " " << fractionY << " " << iybin0 << " " << m_COLS_PER_ROC;
  }
#endif  // EDM_ML_DEBUG
*/
#ifdef EDM_ML_DEBUG
  if (mpY < 0. - EPSDBG || mpY > m_ROCS_Y * m_COLS_PER_ROC + EPSDBG) 
      LogDebug("RectangularPixelPhase2Topology" << " bad pix y " << mpY << " " << py << " " << m_yoffset << " " << m_pitchy;
#endif  // EDM_ML_DEBUG

  /*
#ifdef EDM_ML_DEBUG

  if (ixbin0 > m_ROWS_PER_ROC || ixbin0 < 0)  //  ixbin < 0 outside range
  {
    LogDebug("RectangularPixelPhase2Topology")
        << " very bad, newbinx " << ixbin << "\n"
        << px << " " << m_xoffset << " " << m_pitchx << " " << newxbin << " " << ixbin << " " << fractionX;
  }
#endif  // EDM_ML_DEBUG
*/

  float mpX = ConvertPixelToPitch(px, m_ROWS_PER_ROC, m_ROCS_X, m_xoffset, m_pitchx, m_BIG_PIX_PER_ROC_X, m_BIG_PIX_PITCH_X, m_BIG_PIX_TWOEDGES_X);

#ifdef EDM_ML_DEBUG
  if (mpX < 0. - EPSDBG || mpX > m_ROCS_X * m_ROWS_PER_ROC + EPSDBG) 
      LogDebug("RectangularPixelPhase2Topology") << " bad pix x " << mpX << " " << px << " " << m_xoffset << " " << m_pitchx;
#endif  // EDM_ML_DEBUG
  
  return std::pair<float, float>(mpX, mpY);
}

  //---
  float RectangularPixelPhase2Topology::ConvertPixelToPitch(const float local_pos, const int n_rocrowsorcols, const int nroc, const float sensoractive_half_size, const float pitch, const int big_pix_per_roc, const float big_pix_pitch, const bool big_pix_twoedges) const {
 
    int ibin;
    float fraction;

    float roc_size = (n_rocrowsorcols-big_pix_per_roc)*pitch + big_pix_per_roc*big_pix_pitch;
    float local_pos_new = local_pos - sensoractive_half_size; // distance from sensor edge
    int index_roc = (int)(local_pos_new/roc_size); // roc number (strating from 0)
    float local_pos_roc = local_pos_new - index_roc*roc_size; // distance from (first) roc edge 
    
    int big_pix_1 = 0;

    if ( !big_pix_twoedges ){
        // Big Pixels on one edge of the ROC only 
        if ( index_roc%2 == 1 ) big_pix_1 = big_pix_per_roc;
    } else {
        big_pix_1 = big_pix_per_roc/2;
    }

    float odd_corr = ((n_rocrowsorcols-big_pix_per_roc)%2 == 1) ? 0.5*pitch : 0 ; // correction in case the number of standard pixel is odd

    float w = big_pix_1*big_pix_pitch + pitch*0.5*(n_rocrowsorcols-big_pix_per_roc);

    float fbin_tmp = (fabs(local_pos_roc - w) + odd_corr)/pitch; // distance from the center of the standard pixel array (in pitch units)
    int ibin_tmp;

    if ( fbin_tmp < 0.5*(n_rocrowsorcols-big_pix_per_roc) + odd_corr/pitch) {
        // standard pixel
        ibin_tmp = (int)fbin_tmp;
        fraction = fbin_tmp - ibin_tmp;
    } else {
        // big pixel
        float fbin_big_tmp = (fabs(local_pos_roc - w) - pitch*0.5*(n_rocrowsorcols-big_pix_per_roc)) / big_pix_pitch;
        int ibin_big_tmp = (int)fbin_big_tmp;
        fraction = fbin_big_tmp - ibin_big_tmp;
        ibin_tmp = ibin_big_tmp + (n_rocrowsorcols-big_pix_per_roc)/2; 
        if ( (n_rocrowsorcols-big_pix_per_roc)%2 == 1 ) ibin_tmp += 1;
    }
    if ( local_pos_roc < w ) {
        ibin = index_roc*n_rocrowsorcols + big_pix_1 + (n_rocrowsorcols-big_pix_per_roc)/2 - ibin_tmp; 
        if ( (n_rocrowsorcols-big_pix_per_roc)%2 == 0 ) ibin -= 1; // correction in case the number of standard pixel is even ...2, 1, 0, 0, 1, 2, ...
        fraction = 1.-fraction;
    } else {
        ibin = index_roc*n_rocrowsorcols + big_pix_1 + (n_rocrowsorcols-big_pix_per_roc)/2 + ibin_tmp;  
    }
    return ibin+fraction;

}

//----------------------------------------------------------------------
// Topology interface, go from Masurement to Local corrdinates
// pixel coordinates (mp) -> cm (LocalPoint)
LocalPoint RectangularPixelPhase2Topology::localPosition(const MeasurementPoint& mp) const {
  float mpy = mp.y();  // measurements
  float mpx = mp.x();

#ifdef EDM_ML_DEBUG
#define EPS 0
  // check limits
  std::ostringstream debugstr;

  if (mpy < 0.) {
    debugstr << " wrong mp y, fix " << mpy << " " << 0 << "\n";
    mpy = 0.;
  }
  if (mpy >= m_ncols) {
    debugstr << " wrong mp y, fix " << mpy << " " << m_ncols << "\n";
    mpy = float(m_ncols) - EPS;  // EPS is a small number
  }
  if (mpx < 0.) {
    debugstr << " wrong mp x, fix " << mpx << " " << 0 << "\n";
    mpx = 0.;
  }
  if (mpx >= m_nrows) {
    debugstr << " wrong mp x, fix " << mpx << " " << m_nrows << "\n";
    mpx = float(m_nrows) - EPS;  // EPS is a small number
  }
  if (!debugstr.str().empty())
    LogDebug("RectangularPixelPhase2Topology") << debugstr.str();
#endif  // EDM_ML_DEBUG

  float lpY = localY(mpy);
  float lpX = localX(mpx);

  // Return it as a LocalPoint
  return LocalPoint(lpX, lpY);
}

//--------------------------------------------------------------------
//
// measurement to local transformation for X coordinate
// X coordinate is in the ROC row number direction
float RectangularPixelPhase2Topology::localX(const float mpx) const {
  float lpX = localCoordinate(mpx, m_ROWS_PER_ROC, m_ROCS_X, m_xoffset, m_pitchx, m_BIG_PIX_PER_ROC_X, m_BIG_PIX_PITCH_X, m_BIG_PIX_TWOEDGES_X);
  return lpX;
}

// measurement to local transformation for Y coordinate
// Y is in the ROC column number direction
float RectangularPixelPhase2Topology::localY(const float mpy) const {
  float lpY = localCoordinate(mpy, m_COLS_PER_ROC, m_ROCS_Y, m_yoffset, m_pitchy, m_BIG_PIX_PER_ROC_Y, m_BIG_PIX_PITCH_Y, m_BIG_PIX_TWOEDGES_Y);
  return lpY;
}

float RectangularPixelPhase2Topology::localCoordinate(const float mp_pos, const int n_rocrowsorcols, const int nroc, const float m_offset, const float pitch, const int big_pix_per_roc, const float big_pix_pitch, const bool big_pix_twoedges) const {

  float roc_size = (n_rocrowsorcols-big_pix_per_roc)*pitch + big_pix_per_roc*big_pix_pitch;

  int index_roc = (int)(mp_pos/n_rocrowsorcols); // roc number (strating from 0)
  float mp_pos_roc = mp_pos - index_roc*n_rocrowsorcols; // (float) pixel index in the individual ROC

  int big_pix_1 = 0;
  int big_pix_2 = 0; 

  if ( !big_pix_twoedges ){
    // Big Pixels on one edge of the ROC only 
    if ( index_roc%2 == 0 ) {
      big_pix_2 = big_pix_per_roc;
    } else {
      big_pix_1 = big_pix_per_roc;
    }
  } else {
    big_pix_1 = big_pix_per_roc/2;
    big_pix_2 = big_pix_per_roc/2;  
  }

  float lp_pos = index_roc*roc_size; 
  if ( mp_pos_roc > n_rocrowsorcols-big_pix_2 ) {
    // big pixel
    lp_pos += (mp_pos_roc - (n_rocrowsorcols-big_pix_2)) * big_pix_pitch +(n_rocrowsorcols-big_pix_per_roc)* pitch  + big_pix_1 * big_pix_pitch;
    } else if ( mp_pos_roc > big_pix_1 ) {
    // standard pixel
    lp_pos += (mp_pos_roc - big_pix_1) * pitch + big_pix_1 * big_pix_pitch;
    } else {
    // big pixel 
    lp_pos += mp_pos_roc * big_pix_pitch;
  }

  lp_pos += m_offset;

#ifdef EDM_ML_DEBUG
  if (lp_pos < m_offset || lp_pos > (-m_offset)) {
    LogDebug("RectangularPixelPhase2Topology")
        << " bad lp x " << lp_pos << "\n"
        << mp_pos << " " << m_offset;
  }
#endif  // EDM_ML_DEBUG

  return lp_pos;

}
///////////////////////////////////////////////////////////////////
// Get hit errors in LocalPoint coordinates (cm)
LocalError RectangularPixelPhase2Topology::localError(const MeasurementPoint& mp, const MeasurementError& me) const {

  float pitchy = m_pitchy;
  int binoffy = int(mp.y());
  if (isItBigPixelInY(binoffy))
    pitchy = m_BIG_PIX_PITCH_Y;

  float pitchx = m_pitchx;
  int binoffx = int(mp.x());
  if (isItBigPixelInX(binoffx))
    pitchx = m_BIG_PIX_PITCH_X;

  return LocalError(me.uu() * float(pitchx * pitchx), 0, me.vv() * float(pitchy * pitchy));
}

/////////////////////////////////////////////////////////////////////
// Get errors in pixel pitch units.
MeasurementError RectangularPixelPhase2Topology::measurementError(const LocalPoint& lp, const LocalError& le) const {
  float pitchy = m_pitchy;
  float pitchx = m_pitchx;

  return MeasurementError(le.xx() / float(pitchx * pitchx), 0, le.yy() / float(pitchy * pitchy));
}
