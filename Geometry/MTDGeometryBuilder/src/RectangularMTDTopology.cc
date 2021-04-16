#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

//--------------------------------------------------------------------
// PixelTopology interface.
// Transform LocalPoint in cm to measurement in pitch units.
std::pair<float, float> RectangularMTDTopology::pixel(const LocalPoint& p) const {
  // check limits
  float py = p.y();
  float px = p.x();

  // In Y
  float newybin = (py - m_yoffset) / m_pitchy;
  // In X
  float newxbin = (px - m_xoffset) / m_pitchx;

  return std::pair<float, float>(newxbin, newybin);
}

//----------------------------------------------------------------------
// Topology interface, go from Measurement to Local corrdinates
// pixel coordinates (mp) -> cm (LocalPoint)
LocalPoint RectangularMTDTopology::localPosition(const MeasurementPoint& mp) const {
  float mpy = mp.y();  // measurements
  float mpx = mp.x();

  float lpY = localY(mpy);
  float lpX = localX(mpx);

  // Return it as a LocalPoint
  return LocalPoint(lpX, lpY);
}

//--------------------------------------------------------------------
//
// measuremet to local transformation for X coordinate
float RectangularMTDTopology::localX(const float mpx) const {
  // The final position in local coordinates
  float lpX = mpx * m_pitchx + m_xoffset;

  return lpX;
}

float RectangularMTDTopology::localY(const float mpy) const {
  // The final position in local coordinates
  float lpY = mpy * m_pitchy + m_yoffset;

  return lpY;
}

///////////////////////////////////////////////////////////////////
// Get hit errors in LocalPoint coordinates (cm)
LocalError RectangularMTDTopology::localError(const MeasurementPoint& mp, const MeasurementError& me) const {
  return LocalError(me.uu() * float(m_pitchx * m_pitchx), 0, me.vv() * float(m_pitchy * m_pitchy));
}

/////////////////////////////////////////////////////////////////////
// Get errors in pixel pitch units.
MeasurementError RectangularMTDTopology::measurementError(const LocalPoint& lp, const LocalError& le) const {
  return MeasurementError(le.xx() / float(m_pitchx * m_pitchx), 0, le.yy() / float(m_pitchy * m_pitchy));
}
