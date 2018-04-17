#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalCell.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

DDHGCalCell::DDHGCalCell() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Creating an instance";
#endif
}

DDHGCalCell::~DDHGCalCell() {}

void DDHGCalCell::initialize(const DDNumericArguments & nArgs,
			     const DDVectorArguments &,
			     const DDMapArguments &,
			     const DDStringArguments & sArgs,
			     const DDStringVectorArguments & vsArgs) {

  waferSize_   = nArgs["WaferSize"];
  waferT_      = nArgs["WaferThick"];
  cellT_       = nArgs["CellThick"];
  nCells_      = (int)(nArgs["NCells"]);
  posSens_     = (int)(nArgs["PosSensitive"]);
  material_    = sArgs["Material"];
  fullCN_      = sArgs["FullCell"];
  fullSensN_   = sArgs["FullSensitive"];
  truncCN_     = vsArgs["TruncatedCell"];
  truncSensN_  = vsArgs["TruncatedSensitive"];
  extenCN_     = vsArgs["ExtendedCell"];
  extenSensN_  = vsArgs["ExtendedSensitive"];
  cornrCN_     = vsArgs["CornerCell"];
  cornrSensN_  = vsArgs["CornerSensitive"];
  nameSpace_   = DDCurrentNamespace::ns();
  if ((truncCN_.size() != truncSensN_.size()) ||
      (extenCN_.size() != extenSensN_.size()) ||
      (cornrCN_.size() != cornrSensN_.size()))
    edm::LogWarning("HGCalGeom") << "The number of cells & sensitive differ:"
				 << " Truncated " << truncCN_.size() << ":"
				 << truncSensN_.size() << " Extended "
				 << extenCN_.size() <<":" << extenSensN_.size()
				 << " Corners " << cornrCN_.size() << ":"
				 << cornrSensN_.size();
  if ((truncCN_.size() != 3) || (extenCN_.size() != 3) ||
      (cornrCN_.size() != 6))
    edm::LogWarning("HGCalGeom") << "DDHGCalCell: The number of cells does not"
				 << " match with Standard: Truncated " 
				 << truncCN_.size() << ":3 Extended " 
				 << extenCN_.size() <<":3" << " Corners " 
				 << cornrCN_.size() << ":6";
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Wafer r " << waferSize_
				<< " T " << waferT_ << " Cell T " << cellT_
				<< " Cells/Wafer " << nCells_ << " Material "
				<< material_ << "Sensitive Position "
				<< posSens_ << " NameSpace " << nameSpace_
				<< " Full Cell: " << fullCN_ << ":"
				<< fullSensN_;
  for (unsigned int k=0; k<truncCN_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Truncated Cell[" << k
				  << "] " << truncCN_[k] << ":"
				  << truncSensN_[k];
  for (unsigned int k=0; k<extenCN_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Extended Cell[" << k
				  << "] " << extenCN_[k] << ":"
				  << extenSensN_[k];
  for (unsigned int k=0; k<cornrCN_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Corner Cell[" << k
				  << "] " << cornrCN_[k] << ":"
				  << cornrSensN_[k];
#endif
}

void DDHGCalCell::execute(DDCompactView& cpv) {
  
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Executing DDHGCalCell...";
#endif

  DDName matName(DDSplit(material_).first, DDSplit(material_).second);
  DDMaterial matter(matName);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << matName 
				<< " initialized at " << &matter;
#endif
  DDLogicalPart glog1, glog2;

  static const double sqrt3 = std::sqrt(3.0);
  double R   = waferSize_/(3.0*nCells_);
  double r   = 0.5*R*sqrt3;
  double dx1 = R;
  double dx2 = 0.5*dx1;
  double dx3 = 2.5*dx2;
  double dx4 = 0.5*dx2;
  double dy1 = r;
  double dy2 = 0.5*dy1;
  double dy3 = 1.5*dy1;
  std::vector<double> xx = {dx1,dx2,-dx2,-dx1,-dx2,dx2,
			    dx3,-dx4,-dx1,-dx1,-dx4,dx3};
  std::vector<double> yy = {0,dy1,dy1,0,-dy1,-dy1,
			    dy2,dy3,dy1,-dy1,-dy3,-dy2};
  double zpos = (posSens_ == 0) ? -0.5*(waferT_-cellT_) : 0.5*(waferT_-cellT_);
  DDTranslation tran(0,0,zpos);
  
  // First the full cell
  std::vector<double> xw = {xx[0],xx[1],xx[2],xx[3],xx[4],xx[5]};
  std::vector<double> yw = {yy[0],yy[1],yy[2],yy[3],yy[4],yy[5]};
  std::vector<double> zw = {-0.5*waferT_,0.5*waferT_};
  std::vector<double> zx(2,0), zy(2,0), scale(2,1.0);
  DDSolid solid = DDSolidFactory::extrudedpolygon(DDName(fullCN_, nameSpace_),
						  xw, yw, zw, zx, zy, scale); 
  glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() 
				<< " extruded polygon made of " << matName
				<< " z|x|y|s (0) " << zw[0] << ":" << zx[0] 
				<< ":" << zy[0] << ":" << scale[0] 
				<< " z|x|y|s (1) " << zw[1] << ":" << zx[1] 
				<< ":" << zy[1] << ":" << scale[1] << " and "
				<< xw.size() << " edges";
  for (unsigned int k=0; k<xw.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
  std::vector<double> zc = {-0.5*cellT_,0.5*cellT_};
  solid = DDSolidFactory::extrudedpolygon(DDName(fullSensN_, nameSpace_),
					  xw, yw, zc, zx, zy, scale);
  glog2 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() 
				<< " extruded polygon made of " << matName
				<< " z|x|y|s (0) " << zc[0] << ":" << zx[0] 
				<< ":" << zy[0] << ":" << scale[0] 
				<< " z|x|y|s (1) " << zc[1] << ":" << zx[1] 
				<< ":" << zy[1] << ":" << scale[1] << " and "
				<< xw.size() << " edges";
  for (unsigned int k=0; k<xw.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
  DDRotation    rot;
  cpv.position(glog2, glog1, 1, tran, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() 
				<< " number 1 position in " << glog1.name()
				<< " at " << tran << " with " << rot;
#endif

  static const int ir0[] = {0,1,0};
  static const int ir1[] = {1,2,1};
  static const int ir2[] = {2,3,3};
  static const int ir3[] = {3,4,4};
  static const int ir4[] = {5,5,5};
  for (unsigned int i=0; i<truncCN_.size(); ++i) {
    std::vector<double> xw = {xx[ir0[i]],xx[ir1[i]],xx[ir2[i]],xx[ir3[i]],xx[ir4[i]]};
    std::vector<double> yw = {yy[ir0[i]],yy[ir1[i]],yy[ir2[i]],yy[ir3[i]],yy[ir4[i]]};
    solid = DDSolidFactory::extrudedpolygon(DDName(truncCN_[i], nameSpace_),
					    xw, yw, zw, zx, zy, scale); 
    glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() 
				  << " extruded polygon made of " << matName
				  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] 
				  << ":" << zy[0] << ":" << scale[0] 
				  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] 
				  << ":" << zy[1] << ":" << scale[1] << " and "
				  << xw.size() << " edges";
    for (unsigned int k=0; k<xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" 
				    << yw[k];
#endif
    solid = DDSolidFactory::extrudedpolygon(DDName(truncSensN_[i], nameSpace_),
					    xw, yw, zc, zx, zy, scale);
    glog2 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() 
				  << " extruded polygon made of " << matName
				  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] 
				  << ":" << zy[0] << ":" << scale[0] 
				  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] 
				  << ":" << zy[1] << ":" << scale[1] << " and "
				  << xw.size() << " edges";
    for (unsigned int k=0; k<xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" 
				    << yw[k];
#endif
    cpv.position(glog2, glog1, 1, tran, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() 
				  << " number 1 position in " << glog1.name()
				  << " at " << tran << " with " << rot;
#endif
  }

  static const int ie0[] = {1,5,0};
  static const int ie1[] = {2,6,1};
  static const int ie2[] = {3,7,8};
  static const int ie3[] = {10,3,9};
  static const int ie4[] = {11,4,5};
  for (unsigned int i=0; i<extenCN_.size(); ++i) {
    std::vector<double> xw = {xx[ie0[i]],xx[ie1[i]],xx[ie2[i]],xx[ie3[i]],xx[ie4[i]]};
    std::vector<double> yw = {yy[ie0[i]],yy[ie1[i]],yy[ie2[i]],yy[ie3[i]],yy[ie4[i]]};
    solid = DDSolidFactory::extrudedpolygon(DDName(extenCN_[i], nameSpace_),
					    xw, yw, zw, zx, zy, scale); 
    glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() 
				  << " extruded polygon made of " << matName
				  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] 
				  << ":" << zy[0] << ":" << scale[0] 
				  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] 
				  << ":" << zy[1] << ":" << scale[1] << " and "
				  << xw.size() << " edges";
    for (unsigned int k=0; k<xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" 
				    << yw[k];
#endif
    solid = DDSolidFactory::extrudedpolygon(DDName(extenSensN_[i], nameSpace_),
					    xw, yw, zc, zx, zy, scale);
    glog2 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() 
				  << " extruded polygon made of " << matName
				  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] 
				  << ":" << zy[0] << ":" << scale[0] 
				  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] 
				  << ":" << zy[1] << ":" << scale[1] << " and "
				  << xw.size() << " edges";
    for (unsigned int k=0; k<xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" 
				    << yw[k];
#endif
    cpv.position(glog2, glog1, 1, tran, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() 
				  << " number 1 position in " << glog1.name()
				  << " at " << tran << " with " << rot;
#endif
  }

  static const int ic0[] = {0,1,1,1,1,0};
  static const int ic1[] = {1,2,2,7,3,1};
  static const int ic2[] = {8,3,3,3,4,3};
  static const int ic3[] = {3,5,10,4,5,9};
  static const int ic4[] = {5,11,5,5,6,5};
  for (unsigned int i=0; i<cornrCN_.size(); ++i) {
    std::vector<double> xw = {xx[ic0[i]],xx[ic1[i]],xx[ic2[i]],xx[ic3[i]],xx[ic4[i]]};
    std::vector<double> yw = {yy[ic0[i]],yy[ic1[i]],yy[ic2[i]],yy[ic3[i]],yy[ic4[i]]};
    solid = DDSolidFactory::extrudedpolygon(DDName(cornrCN_[i], nameSpace_),
					    xw, yw, zw, zx, zy, scale); 
    glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() 
				  << " extruded polygon made of " << matName
				  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] 
				  << ":" << zy[0] << ":" << scale[0] 
				  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] 
				  << ":" << zy[1] << ":" << scale[1] << " and "
				  << xw.size() << " edges";
    for (unsigned int k=0; k<xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" 
				    << yw[k];
#endif
    solid = DDSolidFactory::extrudedpolygon(DDName(cornrSensN_[i], nameSpace_),
					    xw, yw, zc, zx, zy, scale);
    glog2 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() 
				  << " extruded polygon made of " << matName
				  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] 
				  << ":" << zy[0] << ":" << scale[0] 
				  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] 
				  << ":" << zy[1] << ":" << scale[1] << " and "
				  << xw.size() << " edges";
    for (unsigned int k=0; k<xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" 
				    << yw[k];
#endif
    cpv.position(glog2, glog1, 1, tran, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() 
				  << " number 1 position in " << glog1.name()
				  << " at " << tran << " with " << rot;
#endif
  }

}
