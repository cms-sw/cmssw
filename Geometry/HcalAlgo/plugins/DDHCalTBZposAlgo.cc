///////////////////////////////////////////////////////////////////////////////
// File: DDHCalTBZposAlgo.cc
// Description: Position inside the mother by shifting along Z given by eta
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

class DDHCalTBZposAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDHCalTBZposAlgo();
  ~DDHCalTBZposAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double eta;      //Eta at which beam is focussed
  double theta;    //Corresponding theta value
  double shiftY;   //Shift along Y
  double shiftX;   //Shift along X
  double zoffset;  //Offset in z
  double dist;     //Radial distance
  double tilt;     //Tilt with respect to y-axis
  int copyNumber;  //Copy Number

  std::string idNameSpace;  //Namespace of this and ALL sub-parts
  std::string childName;    //Children name
};

DDHCalTBZposAlgo::DDHCalTBZposAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: Creating an instance";
#endif
}

DDHCalTBZposAlgo::~DDHCalTBZposAlgo() {}

void DDHCalTBZposAlgo::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments&,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments&) {
  eta = nArgs["Eta"];
  theta = 2.0 * atan(exp(-eta));
  shiftX = nArgs["ShiftX"];
  shiftY = nArgs["ShiftY"];
  zoffset = nArgs["Zoffset"];
  dist = nArgs["Distance"];
  tilt = nArgs["TiltAngle"];
  copyNumber = int(nArgs["Number"]);
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: Parameters for position"
                               << "ing--"
                               << " Eta " << eta << "\tTheta " << convertRadToDeg(theta) << "\tShifts " << shiftX
                               << ", " << shiftY << " along x, y "
                               << "axes; \tZoffest " << zoffset << "\tRadial Distance " << dist << "\tTilt angle "
                               << convertRadToDeg(tilt) << "\tcopyNumber " << copyNumber;

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: Parent " << parent().name() << "\tChild " << childName
                               << " NameSpace " << idNameSpace;
#endif
}

void DDHCalTBZposAlgo::execute(DDCompactView& cpv) {
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);

  double thetax = 90._deg - theta;
  double z = zoffset + dist * tan(thetax);
  double x = shiftX - shiftY * sin(tilt);
  double y = shiftY * cos(tilt);
  DDTranslation tran(x, y, z);
  DDRotation rot;
  if (tilt != 0) {
    std::string rotstr = "R" + formatAsDegrees(tilt);
    rot = DDRotation(DDName(rotstr, idNameSpace));
    if (!rot) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalZposAlgo: Creating a rotation " << DDName(rotstr, idNameSpace) << "\t90, "
                                   << convertRadToDeg(tilt) << ",90," << (90 + convertRadToDeg(tilt)) << ", 0, 0";
#endif
      rot = DDrot(DDName(rotstr, idNameSpace), 90._deg, tilt, 90._deg, (90._deg + tilt), 0.0, 0.0);
    }
  }
  cpv.position(child, mother, copyNumber, tran, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: " << child << " number " << copyNumber << " positioned in "
                               << mother << " at " << tran << " with " << rot;
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHCalTBZposAlgo, "hcal:DDHCalTBZposAlgo");
