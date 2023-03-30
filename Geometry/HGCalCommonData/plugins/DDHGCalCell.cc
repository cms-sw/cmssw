#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include <string>
#include <vector>

//#define EDM_ML_DEBUG

class DDHGCalCell : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalCell() {}

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:
  double waferSize_;                               // Wafer Size
  double waferSeparation_;                         // Wafer Saperation
  int addWaferSeparation_;                         // Use wafer separation
  double waferT_;                                  // Wafer Thickness
  double cellT_;                                   // Cell Thickness
  int nCells_;                                     // Number of columns (8:12)
  int posSens_;                                    // Position depleted layer
  std::string material_;                           // Name of the material
  std::string fullCN_, fullSensN_;                 // Name of full cell
  std::vector<std::string> truncCN_, truncSensN_;  // Names of truncated cells
  std::vector<std::string> extenCN_, extenSensN_;  // Names of extended  cells
  std::vector<std::string> cornrCN_, cornrSensN_;  // Names of corner    cells
  std::string nameSpace_;                          // Namespace to be used
};

void DDHGCalCell::initialize(const DDNumericArguments& nArgs,
                             const DDVectorArguments&,
                             const DDMapArguments&,
                             const DDStringArguments& sArgs,
                             const DDStringVectorArguments& vsArgs) {
  waferSize_ = nArgs["WaferSize"];
  waferSeparation_ = nArgs["WaferSeparation"];
  addWaferSeparation_ = static_cast<int>(nArgs["AddWaferSeparation"]);
  waferT_ = nArgs["WaferThick"];
  cellT_ = nArgs["CellThick"];
  nCells_ = (int)(nArgs["NCells"]);
  posSens_ = (int)(nArgs["PosSensitive"]);
  material_ = sArgs["Material"];
  fullCN_ = sArgs["FullCell"];
  fullSensN_ = sArgs["FullSensitive"];
  truncCN_ = vsArgs["TruncatedCell"];
  truncSensN_ = vsArgs["TruncatedSensitive"];
  extenCN_ = vsArgs["ExtendedCell"];
  extenSensN_ = vsArgs["ExtendedSensitive"];
  cornrCN_ = vsArgs["CornerCell"];
  cornrSensN_ = vsArgs["CornerSensitive"];
  nameSpace_ = DDCurrentNamespace::ns();
  if ((truncCN_.size() != truncSensN_.size()) || (extenCN_.size() != extenSensN_.size()) ||
      (cornrCN_.size() != cornrSensN_.size()))
    edm::LogWarning("HGCalGeom") << "The number of cells & sensitive differ:"
                                 << " Truncated " << truncCN_.size() << ":" << truncSensN_.size() << " Extended "
                                 << extenCN_.size() << ":" << extenSensN_.size() << " Corners " << cornrCN_.size()
                                 << ":" << cornrSensN_.size();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Wafer r " << waferSize_ << " T " << waferT_ << " Cell T " << cellT_
                                << " Cells/Wafer " << nCells_ << " Material " << material_ << "Sensitive Position "
                                << posSens_ << " NameSpace " << nameSpace_ << ": Full Cell: " << fullCN_ << ":"
                                << fullSensN_ << " Separation " << waferSeparation_ << ":" << addWaferSeparation_;
  for (unsigned int k = 0; k < truncCN_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Truncated Cell[" << k << "] " << truncCN_[k] << ":"
                                  << truncSensN_[k];
  for (unsigned int k = 0; k < extenCN_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Extended Cell[" << k << "] " << extenCN_[k] << ":" << extenSensN_[k];
  for (unsigned int k = 0; k < cornrCN_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Corner Cell[" << k << "] " << cornrCN_[k] << ":" << cornrSensN_[k];
#endif
}

void DDHGCalCell::execute(DDCompactView& cpv) {
  DDName matName(DDSplit(material_).first, DDSplit(material_).second);
  DDMaterial matter(matName);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << matName << " initialized";
#endif
  DDLogicalPart glog1, glog2;

  static const double sqrt3 = std::sqrt(3.0);
  double R =
      (addWaferSeparation_ <= 1) ? waferSize_ / (3.0 * nCells_) : (waferSize_ + waferSeparation_) / (3.0 * nCells_);
  double r = 0.5 * R * sqrt3;
  double dx1 = R;
  double dx2 = 0.5 * dx1;
  double dx3 = 2.5 * dx2;
  double dx4 = 0.5 * dx2;
  double dy1 = r;
  double dy2 = 0.5 * dy1;
  double dy3 = 1.5 * dy1;
  std::vector<double> xx = {
      dx1, dx2, -dx2, -dx1, -dx2, dx2, dx3, dx1, dx4, -dx4, -dx1, -dx3, -dx3, -dx1, -dx4, dx4, dx1, dx3};
  std::vector<double> yy = {
      0, dy1, dy1, 0, -dy1, -dy1, dy2, dy1, dy3, dy3, dy1, dy2, -dy2, -dy1, -dy3, -dy3, -dy1, -dy2};
  double zpos = (posSens_ == 0) ? -0.5 * (waferT_ - cellT_) : 0.5 * (waferT_ - cellT_);
  DDTranslation tran(0, 0, zpos);

  double dx5 = (addWaferSeparation_ == 0) ? 0.0 : waferSeparation_ * 0.5;
  double dx6 = dx5 * 0.5;
  double dx7 = dx5;
  double dy4 = dx5 * 0.5 * sqrt3;
  double dy5 = dx5 * 2 / sqrt3;
  double dy6 = dy5 * 0.5;
  std::vector<double> txx = {dx5, dx6, -dx6, -dx5, -dx6, dx6, dx7, 0, -dx7, -dx7, 0, dx7};
  std::vector<double> tyy = {0, dy4, dy4, 0, -dy4, -dy4, dy6, dy5, dy6, -dy6, -dy5, -dy6};
  // First the full cell
  std::vector<double> xw = {xx[0], xx[1], xx[2], xx[3], xx[4], xx[5]};
  std::vector<double> yw = {yy[0], yy[1], yy[2], yy[3], yy[4], yy[5]};
  std::vector<double> zw = {-0.5 * waferT_, 0.5 * waferT_};
  std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
  DDSolid solid = DDSolidFactory::extrudedpolygon(DDName(fullCN_, nameSpace_), xw, yw, zw, zx, zy, scale);
  glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << matName
                                << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                << " and " << xw.size() << " edges";
  for (unsigned int k = 0; k < xw.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
  std::vector<double> zc = {-0.5 * cellT_, 0.5 * cellT_};
  solid = DDSolidFactory::extrudedpolygon(DDName(fullSensN_, nameSpace_), xw, yw, zc, zx, zy, scale);
  glog2 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << matName
                                << " z|x|y|s (0) " << zc[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                << " z|x|y|s (1) " << zc[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                << " and " << xw.size() << " edges";
  for (unsigned int k = 0; k < xw.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
  DDRotation rot;
  cpv.position(glog2, glog1, 1, tran, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() << " number 1 position in " << glog1.name() << " at "
                                << tran << " with no rotation";
#endif

  static constexpr int ir0[] = {0, 2, 4, 1, 3, 5};
  static constexpr int ir1[] = {1, 3, 5, 2, 4, 0};
  static constexpr int ir2[] = {2, 4, 0, 3, 5, 1};
  static constexpr int ir3[] = {3, 5, 1, 4, 0, 2};
  static constexpr int ir4[] = {5, 1, 3, 0, 2, 4};

  static constexpr int tr[] = {1, 3, 5, 2, 4, 0};
  for (unsigned int i = 0; i < truncCN_.size(); ++i) {
    std::vector<double> xw = {xx[ir0[i]], xx[ir1[i]], xx[ir2[i]], xx[ir3[i]] + txx[tr[i]], xx[ir4[i]] + txx[tr[i]]};
    std::vector<double> yw = {yy[ir0[i]], yy[ir1[i]], yy[ir2[i]], yy[ir3[i]] + tyy[tr[i]], yy[ir4[i]] + tyy[tr[i]]};
    solid = DDSolidFactory::extrudedpolygon(DDName(truncCN_[i], nameSpace_), xw, yw, zw, zx, zy, scale);
    glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << matName
                                  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                  << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
    solid = DDSolidFactory::extrudedpolygon(DDName(truncSensN_[i], nameSpace_), xw, yw, zc, zx, zy, scale);
    glog2 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << matName
                                  << " z|x|y|s (0) " << zc[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                  << " z|x|y|s (1) " << zc[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                  << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
    cpv.position(glog2, glog1, 1, tran, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() << " number 1 position in " << glog1.name()
                                  << " at " << tran << " with no rotation";
#endif
  }

  static constexpr int ie0[] = {1, 3, 5, 0, 2, 4};
  static constexpr int ie1[] = {2, 4, 0, 1, 3, 5};
  static constexpr int ie2[] = {3, 5, 1, 2, 4, 0};
  static constexpr int ie3[] = {14, 6, 10, 12, 16, 8};
  static constexpr int ie4[] = {17, 9, 13, 15, 7, 11};

  static constexpr int te[] = {2, 4, 0, 1, 3, 5};
  for (unsigned int i = 0; i < extenCN_.size(); ++i) {
    std::vector<double> xw = {xx[ie0[i]], xx[ie1[i]], xx[ie2[i]], xx[ie3[i]] + txx[te[i]], xx[ie4[i]] + txx[te[i]]};
    std::vector<double> yw = {yy[ie0[i]], yy[ie1[i]], yy[ie2[i]], yy[ie3[i]] + tyy[te[i]], yy[ie4[i]] + tyy[te[i]]};
    solid = DDSolidFactory::extrudedpolygon(DDName(extenCN_[i], nameSpace_), xw, yw, zw, zx, zy, scale);
    glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << matName
                                  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                  << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
    solid = DDSolidFactory::extrudedpolygon(DDName(extenSensN_[i], nameSpace_), xw, yw, zc, zx, zy, scale);
    glog2 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << matName
                                  << " z|x|y|s (0) " << zc[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                  << " z|x|y|s (1) " << zc[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                  << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
    cpv.position(glog2, glog1, 1, tran, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() << " number 1 position in " << glog1.name()
                                  << " at " << tran << " with no rotation";
#endif
  }

  static constexpr int ic0[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
  static constexpr int ic1[] = {1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0};
  static constexpr int ic2[] = {10, 3, 14, 5, 6, 1, 2, 12, 4, 16, 0, 8};
  static constexpr int ic3[] = {3, 5, 5, 1, 1, 3, 4, 4, 0, 0, 2, 2};
  static constexpr int ic4[] = {5, 17, 1, 9, 3, 13, 15, 0, 7, 2, 11, 4};

  static constexpr int tc[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
  for (unsigned int i = 0; i < cornrCN_.size(); ++i) {
    std::vector<double> xw = {
        xx[ic0[i]], xx[ic1[i]], xx[ic2[i]] + txx[tc[i]], xx[ic3[i]] + txx[tc[i] + 6], xx[ic4[i]] + txx[(tc[i] + 1) % 6]};
    std::vector<double> yw = {
        yy[ic0[i]], yy[ic1[i]], yy[ic2[i]] + tyy[tc[i]], yy[ic3[i]] + tyy[tc[i] + 6], yy[ic4[i]] + tyy[(tc[i] + 1) % 6]};
    solid = DDSolidFactory::extrudedpolygon(DDName(cornrCN_[i], nameSpace_), xw, yw, zw, zx, zy, scale);
    glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << matName
                                  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                  << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
    solid = DDSolidFactory::extrudedpolygon(DDName(cornrSensN_[i], nameSpace_), xw, yw, zc, zx, zy, scale);
    glog2 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << matName
                                  << " z|x|y|s (0) " << zc[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                  << " z|x|y|s (1) " << zc[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                  << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xw[k] << ":" << yw[k];
#endif
    cpv.position(glog2, glog1, 1, tran, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() << " number 1 position in " << glog1.name()
                                  << " at " << tran << " with no rotation";
#endif
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalCell, "hgcal:DDHGCalCell");
