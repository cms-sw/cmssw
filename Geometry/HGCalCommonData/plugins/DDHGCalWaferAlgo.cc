///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalWaferAlgo.cc
// Description: Position inside the mother according to (eta,phi)
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

class DDHGCalWaferAlgo : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalWaferAlgo();
  ~DDHGCalWaferAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:
  double cellSize;                      // Cell Size
  int cellType;                         // Type (1 fine; 2 coarse)
  std::vector<std::string> childNames;  // Names of children
  std::vector<int> positionX;           // Position in X
  std::vector<int> positionY;           // Position in Y
  std::vector<double> angles;           // Rotation angle
  std::vector<int> detectorType;        // Detector type
  std::string idName;                   // Name of the "parent" volume.
  std::string rotns;                    // Namespace for rotation matrix
  std::string idNameSpace;              // Namespace of this and ALL sub-parts
  DDName parentName;                    // Parent name
};

DDHGCalWaferAlgo::DDHGCalWaferAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo: Creating an instance";
#endif
}

DDHGCalWaferAlgo::~DDHGCalWaferAlgo() {}

void DDHGCalWaferAlgo::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments& vArgs,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments& vsArgs) {
  cellSize = nArgs["CellSize"];
  cellType = (int)(nArgs["CellType"]);
  childNames = vsArgs["ChildNames"];
  positionX = dbl_to_int(vArgs["PositionX"]);
  positionY = dbl_to_int(vArgs["PositionY"]);
  angles = vArgs["Angles"];
  detectorType = dbl_to_int(vArgs["DetectorType"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << childNames.size() << " children: " << childNames[0] << "; " << childNames[1]
                                << " positioned " << positionX.size() << " times with cell size " << cellSize;
  for (unsigned int k = 0; k < positionX.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] x " << positionX[k] << " y " << positionY[k] << " angle "
                                  << angles[k] << " detector " << detectorType[k];
#endif
  rotns = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  parentName = parent().name();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo debug: Parent " << parentName << " NameSpace " << idNameSpace
                                << " for Rotation " << rotns;
#endif
}

void DDHGCalWaferAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalWaferAlgo...";
#endif
  double dx = 0.5 * cellSize;
  double dy = 0.5 * dx * tan(30._deg);

  for (unsigned int k = 0; k < positionX.size(); ++k) {
    std::string name(childNames[detectorType[k]]);
    DDRotation rotation;
    std::string rotstr("NULL");
    if (angles[k] != 0) {
      double phi = convertDegToRad(angles[k]);
      rotstr = "R" + formatAsDegrees(phi);
      rotation = DDRotation(DDName(rotstr, rotns));
      if (!rotation) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo: Creating new rotation " << DDName(rotstr, rotns) << "\t90, "
                                      << angles[k] << ", 90, " << (angles[k] + 90) << ", 0, 0";
#endif
        rotation = DDrot(DDName(rotstr, rotns), 90._deg, phi, 90._deg, (90._deg + phi), 0, 0);
      }
    }
    double xpos = dx * positionX[k];
    double ypos = dy * positionY[k];
    DDTranslation tran(xpos, ypos, 0);
    int copy = cellType * 1000 + k;
    cpv.position(DDName(name, idNameSpace), parentName, copy, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo: " << DDName(name, idNameSpace) << " number " << copy
                                  << " positioned in " << parentName << " at " << tran << " with " << rotation;
#endif
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalWaferAlgo, "hgcal:DDHGCalWaferAlgo");
