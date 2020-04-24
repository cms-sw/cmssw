/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*	Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"

#include <Math/RotationZYX.h>

using namespace std;

//----------------------------------------------------------------------------------------------------
RPAlignmentCorrectionData::RPAlignmentCorrectionData(double sh_r, double sh_r_e, double sh_x, double sh_x_e,
  double sh_y, double sh_y_e, double sh_z, double sh_z_e, double rot_x, double rot_x_e, double rot_y,
  double rot_y_e, double rot_z, double rot_z_e) :
  translation(sh_x, sh_y, sh_z), translation_error(sh_x_e, sh_y_e, sh_z_e),
  translation_r(sh_r), translation_r_error(sh_r_e),
  rotation_x(rot_x), rotation_y(rot_y), rotation_z(rot_z), 
  rotation_x_error(rot_x_e), rotation_y_error(rot_y_e), rotation_z_error(rot_z_e)
{
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionData::RPAlignmentCorrectionData(double sh_r, double sh_r_e, double sh_x, double sh_x_e, 
    double sh_y, double sh_y_e, double sh_z, double sh_z_e, double rot_z, double rot_z_e) :
  translation(sh_x, sh_y, sh_z), translation_error(sh_x_e, sh_y_e, sh_z_e),
  translation_r(sh_r), translation_r_error(sh_r_e),
  rotation_x(0.), rotation_y(0.), rotation_z(rot_z), 
  rotation_x_error(0.), rotation_y_error(0.), rotation_z_error(rot_z_e)
{
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionData::RPAlignmentCorrectionData(double sh_x, double sh_y, double sh_z, double rot_z) : 
  translation(sh_x, sh_y, sh_z), translation_error(0., 0., 0.),
  translation_r(0.), translation_r_error(0.),
  rotation_x(0.), rotation_y(0.), rotation_z(rot_z), 
  rotation_x_error(0.), rotation_y_error(0.), rotation_z_error(0.)
{
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrectionData::add(const RPAlignmentCorrectionData &a, bool sumErrors, bool addShR,
  bool addShZ, bool addRotZ)
{
  /// TODO: proper adding of all three angles
  
  //printf(">> RPAlignmentCorrectionData::Add, sumErrors = %i\n", sumErrors);

  bool addShXY = addShR;
  
  if (addShR) {
    translation_r = a.translation_r + translation_r;
    if (sumErrors)
      translation_r_error = sqrt(a.translation_r_error*a.translation_r_error + translation_r_error*translation_r_error);
    else
      translation_r_error = a.translation_r_error;
  }

  if (addShXY) {
    translation.SetX(a.translation.X() + translation.X());
    translation.SetY(a.translation.Y() + translation.Y());
    if (sumErrors) {
      translation_error.SetX(sqrt(a.translation_error.X()*a.translation_error.X() + translation_error.X()*translation_error.X()));
      translation_error.SetY(sqrt(a.translation_error.Y()*a.translation_error.Y() + translation_error.Y()*translation_error.Y()));
    } else {
      translation_error.SetX(a.translation_error.X());
      translation_error.SetY(a.translation_error.Y());
    }
  }
  
  if (addShZ) {
    translation.SetZ(a.translation.Z() + translation.Z());
    if (sumErrors)
      translation_error.SetZ(sqrt(a.translation_error.Z()*a.translation_error.Z() + translation_error.Z()*translation_error.Z()));
    else
      translation_error.SetZ(a.translation_error.Z());
  }

  if (addRotZ) {
    rotation_z = a.rotation_z + rotation_z;
    if (sumErrors)
      rotation_z_error = sqrt(a.rotation_z_error*a.rotation_z_error + rotation_z_error*rotation_z_error);
    else
      rotation_z_error = a.rotation_z_error;
  }
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrectionData::readoutTranslationToXY(double dx, double dy)
{
  double tr_z = translation.z();
  translation.SetXYZ(translation_r*dx, translation_r*dy, tr_z);

  tr_z = translation_error.z();
  translation_error.SetXYZ(translation_r_error*dx, translation_r_error*dy, tr_z);
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrectionData::xyTranslationToReadout(double dx, double dy)
{
  double dot = dx*translation.x() + dy*translation.y();
  translation_r = dot;
  translation.SetXYZ(dot*dx, dot*dy, translation.z());

  // there is a very high correlation between x and y components of translation_error
  //double dot_error = sqrt(dx*dx * translation_error.x()*translation_error.x() + dy*dy * translation_error.y()*translation_error.y());
  double dot_error = sqrt(translation_error.x()*translation_error.x() + translation_error.y()*translation_error.y());
  translation_r_error = dot_error;
  translation_error.SetXYZ(dot_error*dx, dot_error*dy, translation_error.z());
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrectionData::setTranslationR(double sh_r, double sh_r_e)
{
  translation_r = sh_r;
  translation_r_error = sh_r_e;
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrectionData::setTranslationZ(double sh_z, double sh_z_e)
{
  translation.SetZ(sh_z);
  translation_error.SetZ(sh_z_e);
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrectionData::setRotationZ(double rot_z, double rot_z_e)
{
  rotation_z = rot_z;
  rotation_z_error = rot_z_e;
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrectionData::normalizeRotationZ()
{
  rotation_z -= floor( (rotation_z + M_PI) / 2. / M_PI ) * 2. * M_PI;
}


//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrectionData::print() const
{
  printf("shift: r=%.1f, x=%.1f, y=%.1f, z=%.1f, rotation: z=%.1f\n", sh_r()*1E3, 
    getTranslation().x()*1E3, getTranslation().y()*1E3, getTranslation().z()*1E3, rot_z()*1E3);
}


