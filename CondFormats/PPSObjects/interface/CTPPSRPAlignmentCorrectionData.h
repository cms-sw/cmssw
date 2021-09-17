/****************************************************************************
 *
 * This is a part of CMS-TOTEM PPS offline software.
 * Authors:
 *  Jan Kašpar (jan.kaspar@gmail.com)
 *  Helena Malbouisson
 *  Clemencia Mora Herrera
 *
 ****************************************************************************/

#ifndef CondFormats_PPSObjects_CTPPSRPAlignmentCorrectionData
#define CondFormats_PPSObjects_CTPPSRPAlignmentCorrectionData

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include <Math/Rotation3D.h>
#include <Math/RotationZYX.h>

#include <vector>

/**
 *\brief Alignment correction for an element of the CT-PPS detector.
 * Within the geometry description, every sensor (more generally every element) is given
 * its <b>translation</b> and <b>rotation</b>. These two quantities shall be understood in
 * <b>local-to-global</b> coordinate transform. That is, if r_l is a point in local
 * coordinate system and x_g in global, then it holds
 \verbatim
    x_g = rotation * x_l + translation
 \endverbatim
 *
 * This class presents an alignment correction to the translation and rotation. It follows
 * these formulae:
 \verbatim
    translation_final = translation_correction + translation_original
    rotation_final = rotation_correction * rotation_original
 \endverbatim
 *
 * Alignment corrections can be added, following this prescription:
 \verbatim
    translation_final = translation_added + translation_original
    rotation_final = rotation_added * rotation_original
 \endverbatim
 *
 * NB: As follows from the above definitions, all translations are in the global space. This
 * means that the rotations do not act on them.
 *
 * Besides the values of rotations and translations, this class contains also uncertainties
 * for these paramaters (the _unc data memebers).
 *
 * The rotation is parameterized by 3 rotation parameters, the matrix is obtained by calling
 * ROOT::Math::RotationZYX(r_z, r_y, r_x), which corresponds to:
  \verbatim
      | 1     0        0    |   | cos r_y  0  +sin r_y |   | cos r_z  -sin r_z  0 |
  R = | 0 cos r_x  -sin r_x | * |    0     1     0     | * | sin r_z  cos r_z   0 |
      | 0 sin r_x  cos r_x  |   |-sin r_y  0  cos r_y  |   |    0        0      1 |
  \endverbatim
 **/
class CTPPSRPAlignmentCorrectionData {
protected:
  /// shift in mm; in global XYZ frame, which is not affected by (alignment) rotations!
  /// "_unc" denotes the shift uncertainties
  double sh_x, sh_y, sh_z;
  double sh_x_unc, sh_y_unc, sh_z_unc;

  /// the three rotation angles
  /// in rad
  double rot_x, rot_y, rot_z;
  double rot_x_unc, rot_y_unc, rot_z_unc;

public:
  CTPPSRPAlignmentCorrectionData()
      : sh_x(0.),
        sh_y(0.),
        sh_z(0.),
        sh_x_unc(0.),
        sh_y_unc(0.),
        sh_z_unc(0.),
        rot_x(0.),
        rot_y(0.),
        rot_z(0.),
        rot_x_unc(0.),
        rot_y_unc(0.),
        rot_z_unc(0.) {}

  /// full constructor, shifts in mm, rotations in rad
  CTPPSRPAlignmentCorrectionData(double _sh_x,
                                 double _sh_x_u,
                                 double _sh_y,
                                 double _sh_y_u,
                                 double _sh_z,
                                 double _sh_z_u,
                                 double _rot_x,
                                 double _rot_x_u,
                                 double _rot_y,
                                 double _rot_y_u,
                                 double _rot_z,
                                 double _rot_z_u);

  /// no uncertainty constructor, shifts in mm, rotation in rad
  CTPPSRPAlignmentCorrectionData(double _sh_x, double _sh_y, double _sh_z, double _rot_x, double _rot_y, double rot_z);

  inline double getShX() const { return sh_x; }
  inline void setShX(const double &v) { sh_x = v; }

  inline double getShXUnc() const { return sh_x_unc; }
  inline void setShXUnc(const double &v) { sh_x_unc = v; }

  inline double getShY() const { return sh_y; }
  inline void setShY(const double &v) { sh_y = v; }

  inline double getShYUnc() const { return sh_y_unc; }
  inline void setShYUnc(const double &v) { sh_y_unc = v; }

  inline double getShZ() const { return sh_z; }
  inline void setShZ(const double &v) { sh_z = v; }

  inline double getShZUnc() const { return sh_z_unc; }
  inline void setShZUnc(const double &v) { sh_z_unc = v; }

  inline double getRotX() const { return rot_x; }
  inline void setRotX(const double &v) { rot_x = v; }

  inline double getRotXUnc() const { return rot_x_unc; }
  inline void setRotXUnc(const double &v) { rot_x_unc = v; }

  inline double getRotY() const { return rot_y; }
  inline void setRotY(const double &v) { rot_y = v; }

  inline double getRotYUnc() const { return rot_y_unc; }
  inline void setRotYUnc(const double &v) { rot_y_unc = v; }

  inline double getRotZ() const { return rot_z; }
  inline void setRotZ(const double &v) { rot_z = v; }

  inline double getRotZUnc() const { return rot_z_unc; }
  inline void setRotZUnc(const double &v) { rot_z_unc = v; }

  math::XYZVectorD getTranslation() const { return math::XYZVectorD(sh_x, sh_y, sh_z); }

  math::XYZVectorD getTranslationUncertainty() const { return math::XYZVectorD(sh_x_unc, sh_y_unc, sh_z_unc); }

  ROOT::Math::Rotation3D getRotationMatrix() const {
    return ROOT::Math::Rotation3D(ROOT::Math::RotationZYX(rot_z, rot_y, rot_x));
  }

  /// merges (cumulates) alignements
  /// \param sumErrors if true, uncertainties are summed in quadrature, otherwise the uncertainties of this are not changed
  /// With the add... switches one can control which corrections are added.
  void add(const CTPPSRPAlignmentCorrectionData &, bool sumErrors = true, bool addSh = true, bool addRot = true);

  COND_SERIALIZABLE;
};

std::ostream &operator<<(std::ostream &s, const CTPPSRPAlignmentCorrectionData &corr);

#endif
