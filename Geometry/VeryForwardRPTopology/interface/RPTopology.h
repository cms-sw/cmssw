/****************************************************************************
*
* This is a part of TotemDQM and TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef Geometry_VeryForwardRPTopology_RPTopology
#define Geometry_VeryForwardRPTopology_RPTopology

#include "Math/Vector3D.h"

/**
 *\brief Geometrical and topological information on RP silicon detector.
 * Uses coordinate a frame with origin in the center of the wafer.
 **/
class RPTopology {
public:
  using Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;

  RPTopology();
  inline const Vector& GetStripReadoutAxisDir() const { return strip_readout_direction_; }
  inline const Vector& GetStripDirection() const { return strip_direction_; }
  inline const Vector& GetNormalDirection() const { return normal_direction_; }

  /// method converts strip number to a hit position [mm] in det readout coordinate
  /// in the origin in the middle of the si detector
  /// strip_no is assumed in the range 0 ... no_of_strips_ - 1
  inline double GetHitPositionInReadoutDirection(double strip_no) const
  //      { return y_width_/2. - last_strip_to_border_dist_ - strip_no * pitch_; }
  {
    return last_strip_to_border_dist_ + (no_of_strips_ - 1) * pitch_ - y_width_ / 2. - strip_no * pitch_;
  }

  inline double DetXWidth() const { return x_width_; }
  inline double DetYWidth() const { return y_width_; }
  inline double DetEdgeLength() const { return phys_edge_lenght_; }
  inline double DetThickness() const { return thickness_; }
  inline double DetPitch() const { return pitch_; }
  inline unsigned short DetStripNo() const { return no_of_strips_; }

  /// returns true if hit at coordinates u, v (in mm) falls into the sensitive area
  /// can take into account insensitive margin (in mm) at the beam-facing edge
  static bool IsHit(double u, double v, double insensitiveMargin = 0);

public:
  static const double sqrt_2;

  static const double pitch_;
  static const double thickness_;
  static const unsigned short no_of_strips_;
  static const double x_width_;
  static const double y_width_;
  static const double phys_edge_lenght_;
  static const double last_strip_to_border_dist_;
  static const double last_strip_to_center_dist_;

  Vector strip_readout_direction_;
  Vector strip_direction_;
  Vector normal_direction_;
};

#endif  //Geometry_VeryForwardRPTopology_RPTopology
