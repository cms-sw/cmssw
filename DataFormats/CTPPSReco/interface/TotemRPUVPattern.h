/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSReco_TotemRPUVPattern
#define DataFormats_CTPPSReco_TotemRPUVPattern

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"

/**
 *\brief A linear pattern in U or V projection.
 * The intercept b is taken at the middle of a RP:
 *     (geometry->GetRPDevice(RPId)->translation().z())
 * The global coordinate system is used (wrt. the beam). This is the same convention
 * as for the 1-RP track fits.
 **/
class TotemRPUVPattern {
public:
  enum ProjectionType { projInvalid, projU, projV };

  TotemRPUVPattern() : projection_(projInvalid), a_(0.), b_(0.), w_(0.), fittable_(false) {}

  ProjectionType projection() const { return projection_; }
  void setProjection(ProjectionType type) { projection_ = type; }

  double a() const { return a_; }
  void setA(double a) { a_ = a; }

  double b() const { return b_; }
  void setB(double b) { b_ = b; }

  double w() const { return w_; }
  void setW(double w) { w_ = w; }

  bool fittable() const { return fittable_; }
  void setFittable(bool fittable) { fittable_ = fittable; }

  void addHit(edm::det_id_type detId, const TotemRPRecHit &hit) { hits_.find_or_insert(detId).push_back(hit); }

  const edm::DetSetVector<TotemRPRecHit> &hits() const { return hits_; }

  friend bool operator<(const TotemRPUVPattern &l, const TotemRPUVPattern &r);

private:
  ProjectionType projection_;  ///< projection
  double a_;                   ///< slope in rad
  double b_;                   ///< intercept in mm
  double w_;                   ///< weight
  bool fittable_;              ///< whether this pattern is worth including in track fits

  edm::DetSetVector<TotemRPRecHit> hits_;  ///< hits associated with the pattern
};

//----------------------------------------------------------------------------------------------------

extern bool operator<(const TotemRPUVPattern &l, const TotemRPUVPattern &r);

#endif
