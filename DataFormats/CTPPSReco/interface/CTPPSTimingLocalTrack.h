/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSTimingLocalTrack
#define DataFormats_CTPPSReco_CTPPSTimingLocalTrack

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CTPPSReco/interface/CTPPSTimingRecHit.h"

//----------------------------------------------------------------------------------------------------

class CTPPSTimingLocalTrack {
public:
  CTPPSTimingLocalTrack();
  CTPPSTimingLocalTrack(const math::XYZPoint& pos0, const math::XYZPoint& pos0_sigma, float t, float t_sigma);

  enum class CheckDimension { x, y, all };
  bool containsHit(const CTPPSTimingRecHit& recHit,
                   float tolerance = 0.1f,
                   CheckDimension check = CheckDimension::all) const;

  //--- spatial get'ters

  inline float x0() const { return pos0_.x(); }
  inline float x0Sigma() const { return pos0_sigma_.x(); }

  inline float y0() const { return pos0_.y(); }
  inline float y0Sigma() const { return pos0_sigma_.y(); }

  inline float z0() const { return pos0_.z(); }
  inline float z0Sigma() const { return pos0_sigma_.z(); }

  inline int numberOfHits() const { return num_hits_; }
  inline int numberOfPlanes() const { return num_planes_; }

  //--- spatial set'ters

  inline void setPosition(const math::XYZPoint& pos0) { pos0_ = pos0; }
  inline void setPositionSigma(const math::XYZPoint& pos0_sigma) { pos0_sigma_ = pos0_sigma; }

  inline void setNumOfHits(int num_hits) { num_hits_ = num_hits; }
  inline void setNumOfPlanes(int num_planes) { num_planes_ = num_planes; }

  //--- validity related members

  inline bool isValid() const { return valid_; }
  inline void setValid(bool valid) { valid_ = valid; }

  //--- temporal get'ters

  inline float time() const { return t_; }
  inline float timeSigma() const { return t_sigma_; }

  //--- temporal set'ters

  inline void setTime(float t) { t_ = t; }
  inline void setTimeSigma(float t_sigma) { t_sigma_ = t_sigma; }

private:
  //--- spatial information

  /// initial track position
  math::XYZPoint pos0_;
  /// error on the initial track position
  math::XYZPoint pos0_sigma_;

  /// number of hits participating in the track
  int num_hits_;

  /// number of planes participating in the track
  int num_planes_;

  /// fit valid?
  bool valid_;

  //--- timing information
  float t_;
  float t_sigma_;
};

/// Comparison operator
bool operator<(const CTPPSTimingLocalTrack& lhs, const CTPPSTimingLocalTrack& rhs);

#endif
