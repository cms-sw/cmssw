/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef Alignment_RPDataFormats_RPAlignmentCorrection
#define Alignment_RPDataFormats_RPAlignmentCorrection

#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include <vector>

/**
 *\brief Alignment correction or result of alignment procedure for a single RP sensor. 
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
    translation_final = translation_correction * translation_original
    rotation_final = rotation_correction * rotation_original
 \endverbatim
 * (see DetGeomDesc::ApplyAlignment)
 *
 * Alignment corrections can be added, following this prescription:
 \verbatim
    translation_final = translation_added * translation_original
    rotation_final = rotation_added * rotation_original
 \endverbatim
 *
 * NB: As follows from the above definitions, all translations are in the global space. This
 * means that the rotations do not act on them.
 *
 * Besides the values of rotations and translations, this class contains also uncertainties
 * for these paramaters (the _error data memebers).
 *
 * The RP silicon sensors are strip detectors and as such they measure one coordinate only.
 * Thus the basic outcome of any track-based alignment involves shifts in read-out direction.
 * This class contains these values in the `_r' data members.
 *
 * The rotation is parameterized by 3 rotation parameters, the matrix is obtained by calling
 * ROOT::Math::RotationZYX(r_z, r_y, r_x), which corresponds to:
  \verbatim
      | 1     0        0    |   | cos r_y  0  +sin r_y |   | cos r_z  -sin r_z  0 |
  R = | 0 cos r_x  -sin r_x | * |    0     1     0     | * | sin r_z  cos r_z   0 |
      | 0 sin r_x  cos r_x  |   |-sin r_y  0  cos r_y  |   |    0        0      1 |
  \endverbatim
 **/
class RPAlignmentCorrection
{
protected:
  /// shift in mm; in global XYZ frame, which is not affected by (alignment) rotations!
  /// currently implemented as ROOT::Math::DisplacementVector3D
  DDTranslation translation;
  
  /// the uncertainty of shift in mm (if known)  
  DDTranslation translation_error;
  
  /// translation in the readout direction, in mm; needed for track-based alignment results
  /// NOTE: no guarantee that its value would correspond to the 'translation' vector!
  double translation_r;
  
  /// the uncertainty of readout-dir. translation
  double translation_r_error; 

  /// the three rotation angles
  /// in rad
  double rotation_x, rotation_y, rotation_z;

  /// the uncertainties of the three rotation angles
  /// in rad
  double rotation_x_error, rotation_y_error, rotation_z_error;

public:
  /// full constructor, shifts in mm, rotations in rad
  RPAlignmentCorrection(double sh_r, double sh_r_e, double sh_x, double sh_x_e, double sh_y, double sh_y_e,
      double sh_z, double sh_z_e, double rot_x, double rot_x_e, double rot_y, double rot_y_e,
      double rot_z, double rot_z_e);

  /// constructor TB alignment, shifts in mm, rotation in rad
  RPAlignmentCorrection(double sh_r, double sh_r_e, double sh_x, double sh_x_e, double sh_y, double sh_y_e,
      double sh_z, double sh_z_e, double rot_z, double rot_z_e);

  /// no error constructor, shifts in mm, rotation in rad
  RPAlignmentCorrection(double sh_x = 0., double sh_y = 0., double sh_z = 0., double rot_z = 0.);

  const DDTranslation& Translation() const
    { return translation; }

  const DDTranslation& TranslationError() const
    { return translation_error; }

  double RotationZ() const
    { return rotation_z; } 

  double RotationZError() const
    { return rotation_z_error; } 

  /// returns rotation matrix build from the 3 rotation angles
  DDRotationMatrix RotationMatrix() const;

  double sh_r() const
    { return translation_r; }

  double sh_r_e() const
    { return translation_r_error; }

  double sh_x() const
    { return translation.x(); }

  double sh_x_e() const
    { return translation_error.x(); }

  double sh_y() const
    { return translation.y(); }

  double sh_y_e() const
    { return translation_error.y(); }

  double sh_z() const
    { return translation.z(); }

  double sh_z_e() const
    { return translation_error.z(); }

  double rot_z() const
    { return rotation_z; }

  double rot_z_e() const
    { return rotation_z_error; }
  
  void SetTranslationR(double sh_r, double sh_r_e = 0.);
  void SetTranslationZ(double sh_z, double sh_z_e = 0.);
  void SetRotationZ(double rot_z, double rot_z_e = 0.);

  /// merges (cumulates) alignements
  /// match between x, y and read-out shifts is not checked
  /// \param sumErrors if it is true, old and new alignment uncertainties are summed (in quadrature)
  /// if it is false, the uncertainties of the parameter (i.e. not the object) will be used
  /// With the add... switches one can control which corrections are added.
  void Add(const RPAlignmentCorrection&, bool sumErrors = true, bool addShR=true,
    bool addShZ=true, bool addRotZ=true);

  /// given (unit-length) readout direction vector (dx, dy), it converts 'translation_r' 
  /// to x and y components of 'translation'
  void ReadoutTranslationToXY(double dx, double dy);
  
  /// given (unit-length) readout direction vector (dx, dy), it removes the translation
  /// component perpendicular to the r-o direction, the parallel component is saved in
  /// 'translation_r'
  void XYTranslationToReadout(double dx, double dy);

  /// adds a multiple of 2pi, such that the rotation is then in range (-pi, +pi)
  void NormalizeRotationZ();
  
  /// prints the contents on the screen
  void Print() const;
  
  /// writes the contents in XML format to the output f
  void WriteXML(FILE *f, bool precise, bool wrErrors, bool wrSh_r, bool wrSh_xy, bool wrSh_z, bool wrRot_z) const;

  // ----- database API -----

  /// streams the class data for communication with database
  std::vector<double> getValues() const;
  
  /// sets the class data from a database record
  void setValues(const std::vector<double> &);
};

#endif

