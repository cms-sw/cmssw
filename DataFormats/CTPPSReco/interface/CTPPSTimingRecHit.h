/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSTimingRecHit
#define DataFormats_CTPPSReco_CTPPSTimingRecHit

/// Reconstructed hit in timing detectors.
class CTPPSTimingRecHit {
public:
  CTPPSTimingRecHit() : x_(0.), xWidth_(0.), y_(0.), yWidth_(0.), z_(0.), zWidth_(0.), t_(0.) {}
  CTPPSTimingRecHit(float x, float xWidth, float y, float yWidth, float z, float zWidth, float t)
      : x_(x), xWidth_(xWidth), y_(y), yWidth_(yWidth), z_(z), zWidth_(zWidth), t_(t) {}

  inline void setX(float x) { x_ = x; }
  inline float x() const { return x_; }

  inline void setY(float y) { y_ = y; }
  inline float y() const { return y_; }

  inline void setZ(float z) { z_ = z; }
  inline float z() const { return z_; }

  inline void setXWidth(float xWidth) { xWidth_ = xWidth; }
  inline float xWidth() const { return xWidth_; }

  inline void setYWidth(float yWidth) { yWidth_ = yWidth; }
  inline float yWidth() const { return yWidth_; }

  inline void setZWidth(float zWidth) { zWidth_ = zWidth; }
  inline float zWidth() const { return zWidth_; }

  inline void setTime(float t) { t_ = t; }
  inline float time() const { return t_; }

protected:
  float x_, xWidth_;
  float y_, yWidth_;
  float z_, zWidth_;
  float t_;
};

//----------------------------------------------------------------------------------------------------

inline bool operator<(const CTPPSTimingRecHit &l, const CTPPSTimingRecHit &r) {
  // only sort by leading edge time
  return (l.time() < r.time());
}

#endif
