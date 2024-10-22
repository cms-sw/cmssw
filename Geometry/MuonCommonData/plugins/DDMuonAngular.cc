///////////////////////////////////////////////////////////////////////////////
// File: DDMuonAngular.cc
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
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

using namespace angle_units::operators;

//#define EDM_ML_DEBUG

class DDMuonAngular : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDMuonAngular();
  ~DDMuonAngular() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double startAngle;  //Start angle
  double stepAngle;   //Step  angle
  double zoffset;     //Offset in z
  int n;              //Mumber of copies
  int startCopyNo;    //Start copy Number
  int incrCopyNo;     //Increment copy Number

  std::string rotns;        //Namespace for rotation matrix
  std::string idNameSpace;  //Namespace of this and ALL sub-parts
  std::string childName;    //Children name
};

DDMuonAngular::DDMuonAngular() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDMuonAngular: Creating an instance";
#endif
}

DDMuonAngular::~DDMuonAngular() {}

void DDMuonAngular::initialize(const DDNumericArguments& nArgs,
                               const DDVectorArguments&,
                               const DDMapArguments&,
                               const DDStringArguments& sArgs,
                               const DDStringVectorArguments&) {
  startAngle = nArgs["startAngle"];
  stepAngle = nArgs["stepAngle"];
  zoffset = nArgs["zoffset"];
  n = int(nArgs["n"]);
  startCopyNo = int(nArgs["startCopyNo"]);
  incrCopyNo = int(nArgs["incrCopyNo"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDMuonAngular: Parameters for positioning-- " << n << " copies in steps of "
                               << convertRadToDeg(stepAngle) << " from " << convertRadToDeg(startAngle) << " \tZoffest "
                               << zoffset << "\tStart and inremental copy nos " << startCopyNo << ", " << incrCopyNo;
#endif
  rotns = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDMuonAngular debug: Parent " << parent().name() << "\tChild " << childName
                               << "\tNameSpace " << idNameSpace << "\tRotation Namespace " << rotns;
#endif
}

void DDMuonAngular::execute(DDCompactView& cpv) {
  double phi = startAngle;
  int copyNo = startCopyNo;

  for (int ii = 0; ii < n; ii++) {
    double phitmp = phi;
    if (phitmp >= 2._pi)
      phitmp -= 2._pi;
    DDRotation rotation;
    std::string rotstr("NULL");

    if (std::abs(phitmp) >= 1.0_deg) {
      rotstr = "R" + formatAsDegrees(phitmp);
      rotation = DDRotation(DDName(rotstr, rotns));
      if (!rotation) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("MuonGeom") << "DDMuonAngular: Creating a new rotation " << DDName(rotstr, idNameSpace)
                                     << "\t90, " << convertRadToDeg(phitmp) << ", 90, "
                                     << convertRadToDeg(phitmp + 90._deg) << ", 0, 0";
#endif
        rotation = DDrot(DDName(rotstr, rotns), 90._deg, phitmp, 90._deg, 90._deg + phitmp, 0., 0.);
      }
    }

    DDTranslation tran(0, 0, zoffset);

    DDName parentName = parent().name();
    cpv.position(DDName(childName, idNameSpace), parentName, copyNo, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonGeom") << "DDMuonAngular: " << DDName(childName, idNameSpace) << " number " << copyNo
                                 << " positioned in " << parentName << " at " << tran << " with " << rotstr << " "
                                 << rotation;
#endif
    phi += stepAngle;
    copyNo += incrCopyNo;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDMuonAngular, "muon:DDMuonAngular");
