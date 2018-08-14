#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalWafer.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

DDHGCalWafer::DDHGCalWafer() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HGCalGeom") << "DDHGCalWafer test: Creating an instance";
#endif
}

DDHGCalWafer::~DDHGCalWafer() {}

void DDHGCalWafer::initialize(const DDNumericArguments & nArgs,
			      const DDVectorArguments & vArgs,
			      const DDMapArguments & ,
			      const DDStringArguments & sArgs,
			      const DDStringVectorArguments & vsArgs) {

  waferSize_   = nArgs["WaferSize"];
  cellType_    = (int)(nArgs["CellType"]);
  nColumns_    = (int)(nArgs["NColumns"]);
  nBottomY_    = (int)(nArgs["NBottomY"]);
  childNames_  = vsArgs["ChildNames"];
  nCellsRow_   = dbl_to_int(vArgs["NCellsRow"]);
  angleEdges_ = dbl_to_int(vArgs["AngleEdges"]);
  detectorType_= dbl_to_int(vArgs["DetectorType"]);
  idNameSpace_ = DDCurrentNamespace::ns();
  parentName_  = parent().name(); 
#ifdef EDM_ML_DEBUG
  std::cout << childNames_.size() << " children: " << childNames_[0] << "; "
	    << childNames_[1] << " in namespace " << idNameSpace_ 
	    << " positioned in " << nCellsRow_.size() << " rows and " 
	    << nColumns_ << " columns with lowest column at " << nBottomY_ 
	    << " in mother " << parentName_ << " of size " << waferSize_ 
	    << std::endl;
  for (unsigned int k=0; k<nCellsRow_.size(); ++k)
    std::cout << "[" << k << "] Ncells " << nCellsRow_[k] << " Edge rotations "
	      << angleEdges_[2*k] << ":" << angleEdges_[2*k+1]
	      << " Type of edge cells " << detectorType_[2*k] << ":"
	      << detectorType_[2*k+1] << std::endl;
#endif
}

void DDHGCalWafer::execute(DDCompactView& cpv) {
  
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HGCalGeom") << "==>> Constructing DDHGCalWafer...";
#endif
  double dx = 0.5*waferSize_/nColumns_;
  double dy = 0.5*dx*tan(30.0*CLHEP::deg);
  int    ny = nBottomY_;
  int    kount(0);

  for (unsigned int ir=0; ir<nCellsRow_.size(); ++ir) {
    int    nx   = 1 - nCellsRow_[ir];
    double ypos = dy*ny;
    for (int ic=0; ic<nCellsRow_[ir]; ++ic) {
      std::string name(childNames_[0]), rotstr("NULL");
      int         irot(0);
      if (ic == 0) {
	name   = childNames_[detectorType_[2*ir]];
	irot   = angleEdges_[2*ir];
      } else if (ic+1== nCellsRow_[ir]) {
	name   = childNames_[detectorType_[2*ir+1]];
	irot   = angleEdges_[2*ir+1];
      }
      DDRotation rot;
      if (irot != 0) {
	if (irot >=0 && irot < 100) rotstr = "R0"; 
	else                        rotstr = "R"; 
	rotstr = rotstr + std::to_string(irot);
	rot    = DDRotation(DDName(rotstr, idNameSpace_)); 
	if (!rot) {
#ifdef EDM_ML_DEBUG
	  std::cout << "DDHGCalWaferAlgo: Creating new rotation "
		    << DDName(rotstr, idNameSpace_) << "\t90, " << irot 
		    << ", 90, " << (irot+90) << ", 0, 0" << std::endl;
#endif
	  rot = DDrot(DDName(rotstr, idNameSpace_), 90*CLHEP::deg, 
		      irot*CLHEP::deg, 90*CLHEP::deg, (90+irot)*CLHEP::deg, 
		      0*CLHEP::deg,0*CLHEP::deg);
	}
      }
      double xpos = dx*nx;
      nx         += 2;
      DDTranslation tran(xpos, ypos, 0);
      int copy = cellType_*1000+kount;
      cpv.position(DDName(name,idNameSpace_), parentName_, copy, tran, rot);
      ++kount;
#ifdef EDM_ML_DEBUG
      std::cout << "DDHGCalWafer: " << DDName(name,idNameSpace_) << " number "
		<< copy << " positioned in " << parentName_ << " at " << tran 
		<< " with " << rot << std::endl;
#endif
    }
    ny         += 6;
  }
}
