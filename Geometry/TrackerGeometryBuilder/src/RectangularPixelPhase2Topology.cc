#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelPhase2Topology.h"

/**
   * Topology for rectangular pixel detector with BIG pixels.
   */
// Modified for the large pixles.
// Danek Kotlinski & Michele Pioppi, 3/06.
// See documentation in the include file.

//--------------------------------------------------------------------
// PixelTopology interface.
// Transform LocalPoint in cm to measurement in pitch units.
std::pair<float, float> RectangularPixelPhase2Topology::pixel(const LocalPoint& p) const {
  // check limits
  float py = p.y();
  float px = p.x();

#ifdef EDM_ML_DEBUG
#define EPSCM 0
#define EPS 0
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

  float newybin = (py - m_yoffset) / m_pitchy;
  int iybin = int(newybin);
  float fractionY = newybin - iybin;

  // Normalize it all to 1 ROC
  int iybin0 = 0;
  int numROC = 0;
  float mpY = 0.;


  iybin0 = (iybin % m_COLS_PER_ROC);  // 0-51
  numROC = iybin / m_COLS_PER_ROC;    // 0-7
  mpY = float(numROC * m_COLS_PER_ROC + iybin0) + fractionY;
  
#ifdef EDM_ML_DEBUG

  if (iybin0 > m_COLS_PER_ROC) {
    LogDebug("RectangularPixelPhase2Topology") << " very bad, newbiny " << iybin0 << "\n"
					       << py << " " << m_yoffset << " " << m_pitchy << " " << newybin << " "
					       << iybin << " " << fractionY << " " << iybin0 << " " << numROC;
  }
#endif  // EDM_ML_DEBUG


#ifdef EDM_ML_DEBUG

  if (mpY < 0. || mpY >= 416.) {
    LogDebug("RectangularPixelPhase2Topology") << " bad pix y " << mpY << "\n"
                                         << py << " " << m_yoffset << " " << m_pitchy << " " << newybin << " " << iybin
                                         << " " << fractionY << " " << iybin0 << " " << numROC;
  }
#endif  // EDM_ML_DEBUG

  // In X
  float newxbin = (px - m_xoffset) / m_pitchx;
  int ixbin = int(newxbin);
  float fractionX = newxbin - ixbin;

#ifdef EDM_ML_DEBUG

  if (ixbin > 161 || ixbin < 0)  //  ixbin < 0 outside range
  {
    LogDebug("RectangularPixelPhase2Topology") << " very bad, newbinx " << ixbin << "\n"
                                         << px << " " << m_xoffset << " " << m_pitchx << " " << newxbin << " " << ixbin
                                         << " " << fractionX;
  }
#endif  // EDM_ML_DEBUG

  float mpX = float(ixbin) + fractionX;

#ifdef EDM_ML_DEBUG

  if (mpX < 0. || mpX >= 160.) {
    LogDebug("RectangularPixelPhase2Topology") << " bad pix x " << mpX << "\n"
                                         << px << " " << m_xoffset << " " << m_pitchx << " " << newxbin << " " << ixbin
                                         << " " << fractionX;
  }
#endif  // EDM_ML_DEBUG

  return std::pair<float, float>(mpX, mpY);
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
// measuremet to local transformation for X coordinate
// X coordinate is in the ROC row number direction
float RectangularPixelPhase2Topology::localX(const float mpx) const {
  int binoffx = int(mpx);                  // truncate to int
  float fractionX = mpx - float(binoffx);  // find the fraction
  float local_pitchx = m_pitchx;           // defaultpitch


#ifdef EDM_ML_DEBUG
  if (binoffx > m_ROWS_PER_ROC * m_ROCS_X)  // too large
    {
      LogDebug("RectangularPixelPhase2Topology")
	<< " very bad, binx " << binoffx << "\n"
	<< mpx << " " << binoffx << " " << fractionX << " " << local_pitchx << " " << m_xoffset << "\n";
    }
#endif
  
  
  // The final position in local coordinates
  float lpX = float(binoffx * m_pitchx) + fractionX * local_pitchx + m_xoffset;

#ifdef EDM_ML_DEBUG

  if (lpX < m_xoffset || lpX > (-m_xoffset)) {
    LogDebug("RectangularPixelPhase2Topology") << " bad lp x " << lpX << "\n"
                                         << mpx << " " << binoffx << " " << fractionX << " " << local_pitchx << " "
                                         << m_xoffset;
  }
#endif  // EDM_ML_DEBUG

  return lpX;
}

// measuremet to local transformation for Y coordinate
// Y is in the ROC column number direction
float RectangularPixelPhase2Topology::localY(const float mpy) const {
  int binoffy = int(mpy);                  // truncate to int
  float fractionY = mpy - float(binoffy);  // find the fraction
  float local_pitchy = m_pitchy;           // defaultpitch

#ifdef EDM_ML_DEBUG
  if (binoffy > m_ROCS_Y * m_COLS_PER_ROC)  // too large
    {
      LogDebug("RectangularPixelPhase2Topology")
	<< " very bad, biny " << binoffy << "\n"
	<< mpy << " " << binoffy << " " << fractionY << " " << local_pitchy << " " << m_yoffset;
    }
#endif

  // The final position in local coordinates
  float lpY = float(binoffy * m_pitchy) + fractionY * local_pitchy + m_yoffset;

#ifdef EDM_ML_DEBUG

  if (lpY < m_yoffset || lpY > (-m_yoffset)) {
    LogDebug("RectangularPixelPhase2Topology") << " bad lp y " << lpY << "\n"
                                         << mpy << " " << binoffy << " " << fractionY << " " << local_pitchy << " "
                                         << m_yoffset;
  }
#endif  // EDM_ML_DEBUG

  return lpY;
}

///////////////////////////////////////////////////////////////////
// Get hit errors in LocalPoint coordinates (cm)
LocalError RectangularPixelPhase2Topology::localError(const MeasurementPoint& mp, const MeasurementError& me) const {
  float pitchy = m_pitchy;
  int binoffy = int(mp.y());
  if (isItBigPixelInY(binoffy))
    pitchy = 2. * m_pitchy;

  float pitchx = m_pitchx;
  int binoffx = int(mp.x());
  if (isItBigPixelInX(binoffx))
    pitchx = 2. * m_pitchx;

  return LocalError(me.uu() * float(pitchx * pitchx), 0, me.vv() * float(pitchy * pitchy));
}

/////////////////////////////////////////////////////////////////////
// Get errors in pixel pitch units.
MeasurementError RectangularPixelPhase2Topology::measurementError(const LocalPoint& lp, const LocalError& le) const {
  float pitchy = m_pitchy;
  float pitchx = m_pitchx;

  return MeasurementError(le.xx() / float(pitchx * pitchx), 0, le.yy() / float(pitchy * pitchy));
}
