#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalWafer8.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

DDHGCalWafer8::DDHGCalWafer8() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer8: Creating an instance";
#endif
}

DDHGCalWafer8::~DDHGCalWafer8() {}

void DDHGCalWafer8::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments &,
			       const DDMapArguments &,
			       const DDStringArguments & sArgs,
			       const DDStringVectorArguments & vsArgs) {

  waferSize_   = nArgs["WaferSize"];
  waferT_      = nArgs["WaferThick"];
  waferSepar_  = nArgs["SensorSeparation"];
  mouseBite_   = nArgs["MouseBite"];
  nCells_      = (int)(nArgs["NCells"]);
  cellType_    = (int)(nArgs["CellType"]);
  material_    = sArgs["Material"];
  cellNames_   = vsArgs["CellNames"];
  nameSpace_   = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer8: Wafer 2r " << waferSize_
				<< " T " << waferT_ << " Half Separation " 
				<< waferSepar_ << " Mouse bite radius "
				<< mouseBite_ << " Cells/Wafer " << nCells_ 
				<< " Cell Type " << cellType_
				<< " Material " << material_ << " Names "
				<< parent().name() << " NameSpace " 
				<< nameSpace_ << " # of cells " 
				<< cellNames_.size();
  for (unsigned int k=0; k<cellNames_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer8: Cell[" << k << "] " 
				  << cellNames_[k];
#endif
}

void DDHGCalWafer8::execute(DDCompactView& cpv) {
  
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Executing DDHGCalWafer8...";
#endif

  static const double sqrt3 = std::sqrt(3.0);
  double rM  = 0.5*(waferSize_ + waferSepar_);
  double RM2 = rM/sqrt3;
  double R   = waferSize_/(3.0*nCells_);
  double r   = 0.5*R*sqrt3;

  // First the full cell
  std::vector<double> xM = {rM,0,-rM,-rM,0,rM};
  std::vector<double> yM = {RM2,2*RM2,RM2,-RM2,-2*RM2,-RM2};
  std::vector<double> zw = {-0.5*waferT_,0.5*waferT_};
  std::vector<double> zx(2,0), zy(2,0), scale(2,1.0);
  DDName parentName = parent().name(); 
  DDSolid solid = DDSolidFactory::extrudedpolygon(parentName,
						  xM, yM, zw, zx, zy, scale); 
  DDName matName(DDSplit(material_).first, DDSplit(material_).second);
  DDMaterial matter(matName);
  DDLogicalPart glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer8: " << solid.name() 
				<< " extruded polygon made of " << matName
				<< " z|x|y|s (0) " << zw[0] << ":" << zx[0] 
				<< ":" << zy[0] << ":" << scale[0] 
				<< " z|x|y|s (1) " << zw[1] << ":" << zx[1] 
				<< ":" << zy[1] << ":" << scale[1] << " and "
				<< xM.size() << " edges";
  for (unsigned int k=0; k<xM.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xM[k] << ":" << yM[k];
#endif

  DDRotation    rot;
  for (int u=0; u<2*nCells_; ++u) {
    for (int v=0; v<2*nCells_; ++v) {
      if (((v-u) < nCells_) && (u-v) <= nCells_) {
	int    n2 = nCells_/2;
	double yp = (u-0.5*v-n2)*2*r;
	double xp = (1.5*(v-nCells_)+1.0)*R;
	int    cell(0);
	if      ((u == 0)           && (v == 0))           cell = 7;
	else if ((u == 0)           && (v == nCells_-1))   cell = 8;
	else if ((u == nCells_)     && (v == 2*nCells_-1)) cell = 9;
	else if ((u == 2*nCells_-1) && (v == 2*nCells_-1)) cell = 10;
	else if ((u == 2*nCells_-1) && (v == nCells_-1))   cell = 11;
	else if ((u == nCells_)     && (v == 0))           cell = 12;
	else if (u == 0)                                   cell = 1;
	else if ((v-u) == (nCells_-1))                     cell = 4;
	else if (v == (2*nCells_-1))                       cell = 2;
	else if (u == (2*nCells_-1))                       cell = 5;
	else if ((u-v) == nCells_)                         cell = 3;
	else if (v == 0)                                   cell = 6;
	DDTranslation tran(xp,yp,0);
	int copy = (cellType_*100 + v)*100 + u;
	cpv.position(DDName(cellNames_[cell]), glog, copy, tran, rot);
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer8: " << cellNames_[cell]
				      << " number " << copy << " position in "
				      << glog.name() << " at " << tran 
				      << " with " <<rot;
#endif
      }
    }
  }

}
