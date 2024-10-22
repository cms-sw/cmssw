/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Author:
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef DataFormats_TotemReco_TotemT2RecHit_h
#define DataFormats_TotemReco_TotemT2RecHit_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class TotemT2RecHit {
public:
  TotemT2RecHit() = default;
  explicit TotemT2RecHit(const GlobalPoint&, float, float, float);

  const GlobalPoint centre() const { return centre_; }
  void setTime(float time) { time_ = time; }
  float time() const { return time_; }
  void setTimeUnc(float time_unc) { time_unc_ = time_unc; }
  float timeUnc() const { return time_unc_; }
  void setToT(float tot) { tot_ = tot; }
  float toT() const { return tot_; }

private:
  /// Tile centre position
  GlobalPoint centre_;
  /// Leading edge time
  float time_{0.f};
  /// Uncertainty on leading edge time
  float time_unc_{0.f};
  /// Time over threshold/pulse width
  float tot_{0.f};
};

bool operator<(const TotemT2RecHit&, const TotemT2RecHit&);

#endif
