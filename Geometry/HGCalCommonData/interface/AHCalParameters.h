#ifndef Geometry_HGCalCommonData_AHCALPARAMETERS_H
#define Geometry_HGCalCommonData_AHCALPARAMETERS_H 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"

/** \class AHCalParameters
 *  Keeps parameters for AHCal
 */

class AHCalParameters {
public:
  /** Create geometry of AHCal */
  AHCalParameters(edm::ParameterSet const&);
  AHCalParameters() = delete;
  ~AHCalParameters() {}

  /// get maximum number of layers
  int maxDepth() const { return maxDepth_; }

  /// get the local coordinate in the plane and along depth
  double deltaX() const { return deltaX_; }
  double deltaY() const { return deltaY_; }
  double deltaZ() const { return deltaZ_; }
  double zFirst() const { return zFirst_; }

  /// Constants used
  static constexpr int kColumn_ = 100;
  static constexpr int kRow_ = 100;
  static constexpr int kSign_ = 10;
  static constexpr int kRowColumn_ = kRow_ * kColumn_;
  static constexpr int kSignRowColumn_ = kSign_ * kRowColumn_;

private:
  const int maxDepth_;
  const double deltaX_, deltaY_, deltaZ_, zFirst_;
};
#endif
