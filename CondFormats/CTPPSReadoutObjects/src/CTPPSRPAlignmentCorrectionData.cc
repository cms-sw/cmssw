/****************************************************************************
*
* This is a part of CMS-TOTEM PPS offline software.
* Authors: 
* Jan Kašpar (jan.kaspar@gmail.com)
* Helena Malbouisson
* Clemencia Mora Herrera 
*
****************************************************************************/

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionData.h"

#include <Math/RotationZYX.h>

using namespace std;

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionData::CTPPSRPAlignmentCorrectionData(double _sh_x, double _sh_x_u, double _sh_y, double _sh_y_u, double _sh_z, double _sh_z_u,
      double _rot_x, double _rot_x_u, double _rot_y, double _rot_y_u, double _rot_z, double _rot_z_u) :
  sh_x(_sh_x), sh_y(_sh_y), sh_z(_sh_z),
  sh_x_unc(_sh_x_u), sh_y_unc(_sh_y_u), sh_z_unc(_sh_z_u),
  rot_x(_rot_x), rot_y(_rot_y), rot_z(_rot_z),
  rot_x_unc(_rot_x_u), rot_y_unc(_rot_y_u), rot_z_unc(_rot_z_u)
{
}

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionData::CTPPSRPAlignmentCorrectionData(double _sh_x, double _sh_y, double _sh_z,
      double _rot_x, double _rot_y, double _rot_z) :
  sh_x(_sh_x), sh_y(_sh_y), sh_z(_sh_z),
  rot_x(_rot_x), rot_y(_rot_y), rot_z(_rot_z)
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentCorrectionData::add(const CTPPSRPAlignmentCorrectionData &a, bool sumErrors, bool addSh, bool addRot)
{
  if (addSh)
  {
    sh_x += a.sh_x;
    sh_y += a.sh_y;
    sh_z += a.sh_z;

    if (sumErrors)
    {
      sh_x_unc = sqrt(sh_x_unc*sh_x_unc + a.sh_x_unc*a.sh_x_unc);
      sh_y_unc = sqrt(sh_y_unc*sh_y_unc + a.sh_y_unc*a.sh_y_unc);
      sh_z_unc = sqrt(sh_z_unc*sh_z_unc + a.sh_z_unc*a.sh_z_unc);
    }
  }

  if (addRot)
  {
    rot_x += a.rot_x;
    rot_y += a.rot_y;
    rot_z += a.rot_z;

    if (sumErrors)
    {
      rot_x_unc = sqrt(rot_x_unc*rot_x_unc + a.rot_x_unc*a.rot_x_unc);
      rot_y_unc = sqrt(rot_y_unc*rot_y_unc + a.rot_y_unc*a.rot_y_unc);
      rot_z_unc = sqrt(rot_z_unc*rot_z_unc + a.rot_z_unc*a.rot_z_unc);
    }
  }
}

//----------------------------------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& s, const CTPPSRPAlignmentCorrectionData &corr)
{
  s << fixed
    << "shift (um) " << std::setprecision(1)
      << "x = " << corr.getShX()*1E3 << " +- " << corr.getShXUnc()*1E3
      << ", y = " << corr.getShY()*1E3 << " +- " << corr.getShYUnc()*1E3
      << ", z = " << corr.getShZ()*1E3 << " +- " << corr.getShZUnc()*1E3
    << ", rotation (mrad) " << std::setprecision(1)
      << "x = " << corr.getRotX()*1E3 << " +- " << corr.getRotXUnc()*1E3
      << ", y = " << corr.getRotY()*1E3 << " +- " << corr.getRotYUnc()*1E3
      << ", z = " << corr.getRotZ()*1E3 << " +- " << corr.getRotZUnc()*1E3;

  return s;
}
