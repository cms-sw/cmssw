///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalWaferAlgo.cc
// Description: Position inside the mother according to (eta,phi)
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

class DDHGCalWaferAlgo : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalWaferAlgo();

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:
  double cellSize_;                      // Cell Size
  int cellType_;                         // Type (1 fine; 2 coarse)
  std::vector<std::string> childNames_;  // Names of children
  std::vector<int> positionX_;           // Position in X
  std::vector<int> positionY_;           // Position in Y
  std::vector<double> angles_;           // Rotation angle
  std::vector<int> detectorType_;        // Detector type
  std::string rotns_;                    // Namespace for rotation matrix
  std::string idNameSpace_;              // Namespace of this and ALL sub-parts
  DDName parentName_;                    // Parent name
};

DDHGCalWaferAlgo::DDHGCalWaferAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo: Creating an instance";
#endif
}

void DDHGCalWaferAlgo::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments& vArgs,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments& vsArgs) {
  cellSize_ = nArgs["CellSize"];
  cellType_ = (int)(nArgs["CellType"]);
  childNames_ = vsArgs["ChildNames"];
  positionX_ = dbl_to_int(vArgs["PositionX"]);
  positionY_ = dbl_to_int(vArgs["PositionY"]);
  angles_ = vArgs["Angles"];
  detectorType_ = dbl_to_int(vArgs["DetectorType"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << childNames_.size() << " children: " << childNames_[0] << "; " << childNames_[1]
                                << " positioned " << positionX_.size() << " times with cell size " << cellSize_;
  for (unsigned int k = 0; k < positionX_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] x " << positionX_[k] << " y " << positionY_[k] << " angle "
                                  << angles_[k] << " detector " << detectorType_[k];
#endif
  rotns_ = sArgs["RotNameSpace"];
  idNameSpace_ = DDCurrentNamespace::ns();
  parentName_ = parent().name();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo debug: Parent " << parentName_ << " NameSpace " << idNameSpace_
                                << " for Rotation " << rotns_;
#endif
}

void DDHGCalWaferAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalWaferAlgo...";
#endif
  double dx = 0.5 * cellSize_;
  double dy = 0.5 * dx * tan(30._deg);

  for (unsigned int k = 0; k < positionX_.size(); ++k) {
    std::string name(childNames_[detectorType_[k]]);
    DDRotation rotation;
    std::string rotstr("NULL");
    if (angles_[k] != 0) {
      double phi = convertDegToRad(angles_[k]);
      rotstr = "R" + formatAsDegrees(phi);
      rotation = DDRotation(DDName(rotstr, rotns_));
      if (!rotation) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo: Creating new rotation " << DDName(rotstr, rotns_)
                                      << "\t90, " << angles_[k] << ", 90, " << (angles_[k] + 90) << ", 0, 0";
#endif
        rotation = DDrot(DDName(rotstr, rotns_), 90._deg, phi, 90._deg, (90._deg + phi), 0, 0);
      }
    }
    double xpos = dx * positionX_[k];
    double ypos = dy * positionY_[k];
    DDTranslation tran(xpos, ypos, 0);
    int copy = HGCalTypes::packCellType6(cellType_, k);
    cpv.position(DDName(name, idNameSpace_), parentName_, copy, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo: " << DDName(name, idNameSpace_) << " number " << copy
                                  << " positioned in " << parentName_ << " at " << tran << " with " << rotation;
#endif
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalWaferAlgo, "hgcal:DDHGCalWaferAlgo");
