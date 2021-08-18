///////////////////////////////////////////////////////////////////////////////
// File: DDHCalTestBeamAlgo.cc
// Description: Position inside the mother according to (eta,phi)
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
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

using namespace angle_units::operators;

class DDHCalTestBeamAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDHCalTestBeamAlgo();
  ~DDHCalTestBeamAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double eta;        //Eta at which beam is focussed
  double phi;        //Phi    ................
  double theta;      //Corresponding theta value
  double distance;   //Distance of the centre of rotation
  double distanceZ;  //Distance along x-axis of the centre of rotation
  double dist;       //Overall distance
  double dz;         //Half length along z of the volume to be placed
  int copyNumber;    //Copy Number

  std::string idNameSpace;  //Namespace of this and ALL sub-parts
  std::string childName;    //Children name
};

DDHCalTestBeamAlgo::DDHCalTestBeamAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTestBeamAlgo: Creating an instance";
#endif
}

DDHCalTestBeamAlgo::~DDHCalTestBeamAlgo() {}

void DDHCalTestBeamAlgo::initialize(const DDNumericArguments& nArgs,
                                    const DDVectorArguments&,
                                    const DDMapArguments&,
                                    const DDStringArguments& sArgs,
                                    const DDStringVectorArguments&) {
  eta = nArgs["Eta"];
  theta = 2.0 * atan(exp(-eta));
  phi = nArgs["Phi"];
  distance = nArgs["Dist"];
  distanceZ = nArgs["DistZ"];
  dz = nArgs["Dz"];
  copyNumber = int(nArgs["Number"]);
  dist = (distance + distanceZ / sin(theta));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTestBeamAlgo: Parameters for position"
                               << "ing--"
                               << " Eta " << eta << "\tPhi " << convertRadToDeg(phi) << "\tTheta "
                               << convertRadToDeg(theta) << "\tDistance " << distance << "/" << distanceZ << "/" << dist
                               << "\tDz " << dz << "\tcopyNumber " << copyNumber;
#endif
  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTestBeamAlgo:Parent " << parent().name() << "\tChild " << childName
                               << " NameSpace " << idNameSpace;
#endif
}

void DDHCalTestBeamAlgo::execute(DDCompactView& cpv) {
  double thetax = 90._deg + theta;
  double sthx = sin(thetax);
  if (std::abs(sthx) > 1.e-12)
    sthx = 1. / sthx;
  else
    sthx = 1.;
  double phix = atan2(sthx * cos(theta) * sin(phi), sthx * cos(theta) * cos(phi));
  double thetay = 90._deg;
  double phiy = 90._deg + phi;
  double thetaz = theta;
  double phiz = phi;

  DDRotation rotation;
  std::string rotstr = childName;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTestBeamAlgo: Creating a rotation " << rotstr << "\t"
                               << convertRadToDeg(thetax) << "," << convertRadToDeg(phix) << ","
                               << convertRadToDeg(thetay) << "," << convertRadToDeg(phiy) << ","
                               << convertRadToDeg(thetaz) << "," << convertRadToDeg(phiz);
#endif
  rotation = DDrot(DDName(rotstr, idNameSpace), thetax, phix, thetay, phiy, thetaz, phiz);

  double r = dist * sin(theta);
  double xpos = r * cos(phi);
  double ypos = r * sin(phi);
  double zpos = dist * cos(theta);
  DDTranslation tran(xpos, ypos, zpos);

  DDName parentName = parent().name();
  cpv.position(DDName(childName, idNameSpace), parentName, copyNumber, tran, rotation);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTestBeamAlgo: " << DDName(childName, idNameSpace) << " number " << copyNumber
                               << " positioned in " << parentName << " at " << tran << " with " << rotation;

  xpos = (dist - dz) * sin(theta) * cos(phi);
  ypos = (dist - dz) * sin(theta) * sin(phi);
  zpos = (dist - dz) * cos(theta);

  edm::LogInfo("HCalGeom") << "DDHCalTestBeamAlgo: Suggested Beam position "
                           << "(" << xpos << ", " << ypos << ", " << zpos << ") and (dist, eta, phi) = (" << (dist - dz)
                           << ", " << eta << ", " << phi << ")";
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHCalTestBeamAlgo, "hcal:DDHCalTestBeamAlgo");
