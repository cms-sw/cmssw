#include <algorithm>
#include <cmath>

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalWafer.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

DDHGCalWafer::DDHGCalWafer() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer: Creating an instance";
#endif
}

DDHGCalWafer::~DDHGCalWafer() {}

void DDHGCalWafer::initialize(const DDNumericArguments& nArgs,
                              const DDVectorArguments& vArgs,
                              const DDMapArguments&,
                              const DDStringArguments& sArgs,
                              const DDStringVectorArguments& vsArgs) {
  waferSize_ = nArgs["WaferSize"];
  cellType_ = (int)(nArgs["CellType"]);
  nColumns_ = (int)(nArgs["NColumns"]);
  nBottomY_ = (int)(nArgs["NBottomY"]);
  childNames_ = vsArgs["ChildNames"];
  nCellsRow_ = dbl_to_int(vArgs["NCellsRow"]);
  angleEdges_ = dbl_to_int(vArgs["AngleEdges"]);
  detectorType_ = dbl_to_int(vArgs["DetectorType"]);
  idNameSpace_ = DDCurrentNamespace::ns();
  parentName_ = parent().name();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom")
    << childNames_.size() << " children: " << childNames_[0] << "; "
    << childNames_[1] << " in namespace " << idNameSpace_
    << " positioned in " << nCellsRow_.size() << " rows and "
    << nColumns_ << " columns with lowest column at " << nBottomY_
    << " in mother " << parentName_ << " of size " << waferSize_;
  for (unsigned int k = 0; k < nCellsRow_.size(); ++k)
    edm::LogVerbatim("HGCalGeom")
      << "[" << k << "] Ncells " << nCellsRow_[k] << " Edge rotations "
      << angleEdges_[2 * k] << ":" << angleEdges_[2 * k + 1]
      << " Type of edge cells " << detectorType_[2 * k] << ":"
      << detectorType_[2 * k + 1];
#endif
}

void DDHGCalWafer::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalWafer...";
#endif
  double dx = 0.5 * waferSize_ / nColumns_;
  double dy = 0.5 * dx * tan(30._deg);
  int ny = nBottomY_;
  int kount(0);

  for (unsigned int ir = 0; ir < nCellsRow_.size(); ++ir) {
    int nx = 1 - nCellsRow_[ir];
    double ypos = dy * ny;
    for (int ic = 0; ic < nCellsRow_[ir]; ++ic) {
      std::string name(childNames_[0]), rotstr("NULL");
      int irot(0);
      if (ic == 0) {
        name = childNames_[detectorType_[2 * ir]];
        irot = angleEdges_[2 * ir];
      } else if (ic + 1 == nCellsRow_[ir]) {
        name = childNames_[detectorType_[2 * ir + 1]];
        irot = angleEdges_[2 * ir + 1];
      }
      DDRotation rot;
      if (irot != 0) {
	double phi = convertDegToRad(irot);
        rotstr = "R" + formatAsDegrees(phi);
        rot = DDRotation(DDName(rotstr, idNameSpace_));
        if (!rot) {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom")
	    << "DDHGCalWaferAlgo: Creating new rotation "
	    << DDName(rotstr, idNameSpace_) << "\t90, " << irot
	    << ", 90, " << (irot + 90) << ", 0, 0";
#endif
          rot = DDrot(DDName(rotstr, idNameSpace_), 90._deg, phi, 90._deg,
                      (90._deg+phi), 0, 0);
        }
      }
      double xpos = dx * nx;
      nx += 2;
      DDTranslation tran(xpos, ypos, 0);
      int copy = cellType_ * 1000 + kount;
      cpv.position(DDName(name, idNameSpace_), parentName_, copy, tran, rot);
      ++kount;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom")
	<< "DDHGCalWafer: " << DDName(name, idNameSpace_) << " number "
	<< copy << " positioned in " << parentName_ << " at " << tran
	<< " with " << rot;
#endif
    }
    ny += 6;
  }
}
