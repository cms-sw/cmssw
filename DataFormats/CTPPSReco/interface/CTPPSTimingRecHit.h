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
  CTPPSTimingRecHit()
      : x_(0.), x_width_(0.), y_(0.), y_width_(0.), z_(0.), z_width_(0.),
        t_(0.), tot_(0.), t_precision_(0.) {}
  CTPPSTimingRecHit(float x, float x_width, float y, float y_width, float z,
                    float z_width, float t, float tot, float t_precision)
      : x_(x), x_width_(x_width), y_(y), y_width_(y_width), z_(z),
        z_width_(z_width), t_(t), tot_(tot), t_precision_(t_precision) {}

  inline void setX(const float &x) { x_ = x; }
  inline float getX() const { return x_; }

  inline void setY(const float &y) { y_ = y; }
  inline float getY() const { return y_; }

  inline void setZ(const float &z) { z_ = z; }
  inline float getZ() const { return z_; }

  inline void setXWidth(const float &xwidth) { x_width_ = xwidth; }
  inline float getXWidth() const { return x_width_; }

  inline void setYWidth(const float &ywidth) { y_width_ = ywidth; }
  inline float getYWidth() const { return y_width_; }

  inline void setZWidth(const float &zwidth) { z_width_ = zwidth; }
  inline float getZWidth() const { return z_width_; }

  inline void setT(const float &t) { t_ = t; }
  inline float getT() const { return t_; }

  inline void setToT(const float &tot) { tot_ = tot; }
  inline float getToT() const { return tot_; }

  inline void setTPrecision(const float &t_precision) {
    t_precision_ = t_precision;
  }
  inline float getTPrecision() const { return t_precision_; }

protected:
  float x_, x_width_;
  float y_, y_width_;
  float z_, z_width_;
  float t_, tot_, t_precision_;
};

//----------------------------------------------------------------------------------------------------

inline bool operator<(const CTPPSTimingRecHit &l, const CTPPSTimingRecHit &r) {
  // only sort by leading edge time
  return (l.getT() < r.getT());
}

#endif
