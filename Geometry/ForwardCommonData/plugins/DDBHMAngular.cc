///////////////////////////////////////////////////////////////////////////////
// File: DDBHMAngular.cc
// Description: Position inside the mother according to phi
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDutils.h"

using namespace geant_units::operators;

//#define EDM_ML_DEBUG

class DDBHMAngular : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDBHMAngular();

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  int units;    //Number of copies
  double rr;    //Radial position of the detectors
  double dphi;  //the distance in phi between the detectors

  std::string rotMat;     //Name of the rotation matrix
  std::string childName;  //Children name
};

DDBHMAngular::DDBHMAngular() { edm::LogVerbatim("ForwardGeom") << "DDBHMAngular test: Creating an instance"; }

void DDBHMAngular::initialize(const DDNumericArguments& nArgs,
                              const DDVectorArguments&,
                              const DDMapArguments&,
                              const DDStringArguments& sArgs,
                              const DDStringVectorArguments&) {
  units = int(nArgs["number"]);
  rr = nArgs["radius"];
  dphi = nArgs["deltaPhi"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDBHMAngular debug: Parameters for positioning-- " << units
                                  << " copies at radius " << convertMmToCm(rr) << " cm with delta(phi) "
                                  << convertRadToDeg(dphi);
#endif
  rotMat = sArgs["Rotation"];
  childName = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDBHMAngular debug: Parent " << parent().name() << "\tChild " << childName
                                  << "\tRotation matrix " << rotMat;
#endif
}

void DDBHMAngular::execute(DDCompactView& cpv) {
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  DDName parentName = parent().name();
  std::string rotstr = DDSplit(rotMat).first;
  DDRotation rot;
  if (rotstr != "NULL") {
    std::string rotns = DDSplit(rotMat).second;
    rot = DDRotation(DDName(rotstr, rotns));
  }

  static const double fac1 = 0.5;
  static const double fac2 = 1.5;
  static const double fac3 = 14.5;
  static const double fac4 = 15.5;
  for (int jj = 0; jj < units; jj++) {
    double driverX(0), driverY(0), driverZ(0);
    if (jj < 16) {
      driverX = rr * cos((jj + fac1) * dphi);
      driverY = sqrt(rr * rr - driverX * driverX);
    } else if (jj == 16) {
      driverX = rr * cos(fac4 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == 17) {
      driverX = rr * cos(fac3 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == 18) {
      driverX = rr * cos(fac1 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == 19) {
      driverX = rr * cos(fac2 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    }
    DDTranslation tran(driverX, driverY, driverZ);

    cpv.position(child, parentName, jj + 1, tran, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardGeom") << "DDBHMAngular test: " << child << " number " << jj + 1 << " positioned in "
                                    << parentName << " at " << tran << " with " << rot;
#endif
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDBHMAngular, "bhmalgo:DDBHMAngular");
