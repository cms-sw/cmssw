/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*	Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/RPDataFormats/interface/RPAlignmentCorrection.h"

#include "Math/GenVector/RotationZYX.h"

using namespace std;

//----------------------------------------------------------------------------------------------------
RPAlignmentCorrection::RPAlignmentCorrection(double sh_r, double sh_r_e, double sh_x, double sh_x_e,
  double sh_y, double sh_y_e, double sh_z, double sh_z_e, double rot_x, double rot_x_e, double rot_y,
  double rot_y_e, double rot_z, double rot_z_e) :
  translation(sh_x, sh_y, sh_z), translation_error(sh_x_e, sh_y_e, sh_z_e),
  translation_r(sh_r), translation_r_error(sh_r_e),
  rotation_x(rot_x), rotation_y(rot_y), rotation_z(rot_z), 
  rotation_x_error(rot_x_e), rotation_y_error(rot_y_e), rotation_z_error(rot_z_e)
{
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrection::RPAlignmentCorrection(double sh_r, double sh_r_e, double sh_x, double sh_x_e, 
    double sh_y, double sh_y_e, double sh_z, double sh_z_e, double rot_z, double rot_z_e) :
  translation(sh_x, sh_y, sh_z), translation_error(sh_x_e, sh_y_e, sh_z_e),
  translation_r(sh_r), translation_r_error(sh_r_e),
  rotation_x(0.), rotation_y(0.), rotation_z(rot_z), 
  rotation_x_error(0.), rotation_y_error(0.), rotation_z_error(rot_z_e)
{
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrection::RPAlignmentCorrection(double sh_x, double sh_y, double sh_z, double rot_z) : 
  translation(sh_x, sh_y, sh_z), translation_error(0., 0., 0.),
  translation_r(0.), translation_r_error(0.),
  rotation_x(0.), rotation_y(0.), rotation_z(rot_z), 
  rotation_x_error(0.), rotation_y_error(0.), rotation_z_error(0.)
{
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrection::Add(const RPAlignmentCorrection &a, bool sumErrors, bool addShR,
  bool addShZ, bool addRotZ)
{
  /// TODO: proper adding of all three angles
  
  //printf(">> RPAlignmentCorrection::Add, sumErrors = %i\n", sumErrors);

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

void RPAlignmentCorrection::ReadoutTranslationToXY(double dx, double dy)
{
  double tr_z = translation.z();
  translation.SetXYZ(translation_r*dx, translation_r*dy, tr_z);

  tr_z = translation_error.z();
  translation_error.SetXYZ(translation_r_error*dx, translation_r_error*dy, tr_z);
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrection::XYTranslationToReadout(double dx, double dy)
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

void RPAlignmentCorrection::SetTranslationR(double sh_r, double sh_r_e)
{
  translation_r = sh_r;
  translation_r_error = sh_r_e;
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrection::SetTranslationZ(double sh_z, double sh_z_e)
{
  translation.SetZ(sh_z);
  translation_error.SetZ(sh_z_e);
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrection::SetRotationZ(double rot_z, double rot_z_e)
{
  rotation_z = rot_z;
  rotation_z_error = rot_z_e;
}

//----------------------------------------------------------------------------------------------------

#if 0
int RPAlignmentCorrection::ExtractRotationZ(const DDRotationMatrix &rotation, double &angle, double limit)
{
  double xx, xy, xz, yx, yy, yz, zx, zy, zz;
  rotation.GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz);

  if (fabs(zz - 1.) > limit || fabs(zx) > limit || fabs(zy) > limit || fabs(xz) > limit || fabs(yz) > limit)
    return 2;

  angle = -atan2(xy, xx); // in -pi to +pi range
  return 0;
}
#endif

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrection::NormalizeRotationZ()
{
  rotation_z -= floor( (rotation_z + M_PI) / 2. / M_PI ) * 2. * M_PI;
}

//----------------------------------------------------------------------------------------------------

DDRotationMatrix RPAlignmentCorrection::RotationMatrix() const
{
  return DDRotationMatrix(ROOT::Math::RotationZYX(rotation_z, rotation_y, rotation_x));
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrection::Print() const
{
  printf("shift: r=%.1f, x=%.1f, y=%.1f, z=%.1f, rotation: z=%.1f\n", sh_r()*1E3, 
    Translation().x()*1E3, Translation().y()*1E3, Translation().z()*1E3, rot_z()*1E3);
}

//----------------------------------------------------------------------------------------------------

#define WRITE(q, dig, lim) \
  if (precise) \
    fprintf(f, " " #q "=\"%.15E\"", q()*1E3);\
  else \
    if (fabs(q()*1E3) < lim && q() != 0) \
      fprintf(f, " " #q "=\"%+8.1E\"", q()*1E3);\
    else \
      fprintf(f, " " #q "=\"%+8." #dig "f\"", q()*1E3);

void RPAlignmentCorrection::WriteXML(FILE *f, bool precise, bool wrErrors, bool wrSh_r, bool wrSh_xy, 
  bool wrSh_z, bool wrRot_z) const
{
  if (wrSh_r) {
    WRITE(sh_r, 2, 0.1);
    if (wrErrors) {
      WRITE(sh_r_e, 2, 0.1);
    }
    /*
    fprintf(f, " sh_r=\"%+8.2f\"", sh_r()*1E3);
    if (wrErrors)
      if (fabs(sh_r_e())*1E3 < 0.1)
        fprintf(f, " sh_r_e=\"%+8.1E\"", sh_r_e()*1E3);
      else
        fprintf(f, " sh_r_e=\"%+8.2f\"", sh_r_e()*1E3);
    */
  }

  if (wrSh_xy) {
    WRITE(sh_x, 2, 0.1);
    WRITE(sh_y, 2, 0.1);
    if (wrErrors) {
      WRITE(sh_x_e, 2, 0.1);
      WRITE(sh_y_e, 2, 0.1);
    }
    /*
    fprintf(f, " sh_x=\"%+8.2f\" sh_y=\"%+8.2f\"", sh_x()*1E3, sh_y()*1E3);
    if (wrErrors) {
      if (fabs(sh_x_e())*1E3 < 0.1)
        fprintf(f, " sh_x_e=\"%+8.1E\"", sh_x_e()*1E3);
      else
        fprintf(f, " sh_x_e=\"%+8.2f\"", sh_x_e()*1E3);

      if (fabs(sh_y_e())*1E3 < 0.1)
        fprintf(f, " sh_y_e=\"%+8.1E\"", sh_y_e()*1E3);
      else
        fprintf(f, " sh_y_e=\"%+8.2f\"", sh_y_e()*1E3);
    }
    */
  }

  // TODO: add the other 2 rotations

  if (wrRot_z) {
    WRITE(rot_z, 3, 0.01);
    if (wrErrors) {
      WRITE(rot_z_e, 3, 0.01);
    }
    /*
    fprintf(f, " rot_z=\"%+8.3f\"", rot_z()*1E3);
    if (wrErrors)
      if (fabs(rot_z_e())*1E3 < 0.01)
        fprintf(f, " rot_z_e=\"%+8.1E\"", rot_z_e()*1E3);
      else
        fprintf(f, " rot_z_e=\"%+8.3f\"", rot_z_e()*1E3);
    */
  }

  if (wrSh_z) {
    WRITE(sh_z, 2, 0.1);
    if (wrErrors) {
      WRITE(sh_z_e, 2, 0.1);
    }

    /*
    fprintf(f, " sh_z=\"%+8.2f\"", sh_z()*1E3);
    if (wrErrors)
      if (fabs(sh_z_e())*1E3 < 0.1)
        fprintf(f, " sh_z_e=\"%+8.1E\"", sh_z_e()*1E3);
      else
        fprintf(f, " sh_z_e=\"%+8.2f\"", sh_z_e()*1E3);
    */
  }
}

#undef WRITE

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

vector<double> RPAlignmentCorrection::getValues() const
{
  vector<double> result;

  result.push_back(translation.x());
  result.push_back(translation.y());
  result.push_back(translation.z());
  
  result.push_back(translation_error.x());
  result.push_back(translation_error.y());
  result.push_back(translation_error.z());
  
  result.push_back(translation_r);
  result.push_back(translation_r_error);
  
  result.push_back(rotation_x);
  result.push_back(rotation_y);
  result.push_back(rotation_z);
  
  result.push_back(rotation_x_error);
  result.push_back(rotation_y_error);
  result.push_back(rotation_z_error);
  
  return result;
}

//----------------------------------------------------------------------------------------------------

void RPAlignmentCorrection::setValues(const std::vector<double> &data)
{
  if (data.size() != 14)
    throw cms::Exception("RPAlignmentCorrection::setValues") << "Data contain " << data.size()
    << " values, instead of 14, as expected." << endl;

  translation.SetXYZ(data[0], data[1], data[2]);
  translation_error.SetXYZ(data[3], data[4], data[5]);

  translation_r = data[6];
  translation_r_error = data[7];
  
  rotation_x = data[8];
  rotation_y = data[9];
  rotation_z = data[10];

  rotation_x_error = data[11];
  rotation_y_error = data[12];
  rotation_z_error = data[13];
}

