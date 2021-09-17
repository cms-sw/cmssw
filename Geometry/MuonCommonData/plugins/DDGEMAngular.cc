///////////////////////////////////////////////////////////////////////////////
// File: DDGEMAngular.cc
// Description: Position inside the mother according to (eta,phi)
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

//#define EDM_ML_DEBUG

using namespace angle_units::operators;

class DDGEMAngular : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDGEMAngular();
  ~DDGEMAngular() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double startAngle;  //Start angle
  double stepAngle;   //Step  angle
  int invert;         //inverted or forward
  double rPos;        //Radial position of center
  double zoffset;     //Offset in z
  int n;              //Mumber of copies
  int startCopyNo;    //Start copy Number
  int incrCopyNo;     //Increment copy Number

  std::string rotns;        //Namespace for rotation matrix
  std::string idNameSpace;  //Namespace of this and ALL sub-parts
  std::string childName;    //Children name
};

DDGEMAngular::DDGEMAngular() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDGEMAngular: Creating an instance";
#endif
}

DDGEMAngular::~DDGEMAngular() {}

void DDGEMAngular::initialize(const DDNumericArguments& nArgs,
                              const DDVectorArguments&,
                              const DDMapArguments&,
                              const DDStringArguments& sArgs,
                              const DDStringVectorArguments&) {
  startAngle = nArgs["startAngle"];
  stepAngle = nArgs["stepAngle"];
  invert = int(nArgs["invert"]);
  rPos = nArgs["rPosition"];
  zoffset = nArgs["zoffset"];
  n = int(nArgs["n"]);
  startCopyNo = int(nArgs["startCopyNo"]);
  incrCopyNo = int(nArgs["incrCopyNo"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDGEMAngular: Parameters for positioning-- " << n << " copies in steps of "
                               << convertRadToDeg(stepAngle) << " from " << convertRadToDeg(startAngle)
                               << " (inversion flag " << invert << ") \trPos " << rPos << " Zoffest " << zoffset
                               << "\tStart and inremental "
                               << "copy nos " << startCopyNo << ", " << incrCopyNo;
#endif

  rotns = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDGEMAngular: Parent " << parent().name() << "\tChild " << childName
                               << "\tNameSpace " << idNameSpace << "\tRotation Namespace " << rotns;
#endif
}

void DDGEMAngular::execute(DDCompactView& cpv) {
  double phi = startAngle;
  int copyNo = startCopyNo;

  for (int ii = 0; ii < n; ii++) {
    double phitmp = phi;
    if (phitmp >= 2._pi)
      phitmp -= 2._pi;
    DDRotation rotation;
    std::string rotstr("RG");

    if (invert > 0)
      rotstr += "I";
    rotstr += formatAsDegrees(phitmp);
    rotation = DDRotation(DDName(rotstr, rotns));
    if (!rotation) {
      double thetax = 90.0_deg;
      double phix = invert == 0 ? (90.0_deg + phitmp) : (-90.0_deg + phitmp);
      double thetay = invert == 0 ? 0.0 : 180.0_deg;
      double phiz = phitmp;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "DDGEMAngular: Creating a new rotation " << DDName(rotstr, idNameSpace) << "\t "
                                   << convertRadToDeg(thetax) << ", " << convertRadToDeg(phix) << ", "
                                   << convertRadToDeg(thetay) << ", 0, " << convertRadToDeg(thetax) << ", "
                                   << convertRadToDeg(phiz);
#endif
      rotation = DDrot(DDName(rotstr, rotns), thetax, phix, thetay, 0., thetax, phiz);
    }

    DDTranslation tran(rPos * cos(phitmp), rPos * sin(phitmp), zoffset);

    DDName parentName = parent().name();
    cpv.position(DDName(childName, idNameSpace), parentName, copyNo, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonGeom") << "DDGEMAngular: " << DDName(childName, idNameSpace) << " number " << copyNo
                                 << " positioned in " << parentName << " at " << tran << " with " << rotstr << " "
                                 << rotation;
#endif
    phi += stepAngle;
    copyNo += incrCopyNo;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDGEMAngular, "muon:DDGEMAngular");
