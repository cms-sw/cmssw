///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalModule.cc
// Description: Geometry factory class for HGCal (EE and HESil)
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataFormats/Math/interface/angle_units.h"
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
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

class DDHGCalModule : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalModule();  // const std::string & name);

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:
  void constructLayers(const DDLogicalPart&, DDCompactView& cpv);
  double rMax(double z);
  void positionSensitive(DDLogicalPart& glog, double rin, double rout, DDCompactView& cpv);

private:
  static constexpr double tol_ = 0.00001;

  std::vector<std::string> wafer_;      // Wafers
  std::vector<std::string> materials_;  // Materials
  std::vector<std::string> names_;      // Names
  std::vector<double> thick_;           // Thickness of the material
  std::vector<int> copyNumber_;         // Initial copy numbers
  std::vector<int> layers_;             // Number of layers in a section
  std::vector<double> layerThick_;      // Thickness of each section
  std::vector<int> layerType_;          // Type of the layer
  std::vector<int> layerSense_;         // COntent of a layer (sensitive?)
  double zMinBlock_;                    // Starting z-value of the block
  double rMaxFine_;                     // Maximum r-value for fine wafer
  double waferW_;                       // Width of the wafer
  int sectors_;                         // Sectors
  std::vector<double> slopeB_;          // Slope at the lower R
  std::vector<double> slopeT_;          // Slopes at the larger R
  std::vector<double> zFront_;          // Starting Z values for the slopes
  std::vector<double> rMaxFront_;       // Corresponding rMax's
  std::string idNameSpace_;             // Namespace of this and ALL sub-parts
  std::unordered_set<int> copies_;      // List of copy #'s
};

DDHGCalModule::DDHGCalModule() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: Creating an instance";
#endif
}

void DDHGCalModule::initialize(const DDNumericArguments& nArgs,
                               const DDVectorArguments& vArgs,
                               const DDMapArguments&,
                               const DDStringArguments& sArgs,
                               const DDStringVectorArguments& vsArgs) {
  wafer_ = vsArgs["WaferName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << wafer_.size() << " wafers";
  for (unsigned int i = 0; i < wafer_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafer_[i];
#endif
  materials_ = vsArgs["MaterialNames"];
  names_ = vsArgs["VolumeNames"];
  thick_ = vArgs["Thickness"];
  copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << materials_.size() << " types of volumes";
  for (unsigned int i = 0; i < names_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names_[i] << " of thickness " << thick_[i]
                                  << " filled with " << materials_[i] << " first copy number " << copyNumber_[i];
#endif
  layers_ = dbl_to_int(vArgs["Layers"]);
  layerThick_ = vArgs["LayerThick"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << layers_.size() << " blocks";
  for (unsigned int i = 0; i < layers_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick_[i] << " with " << layers_[i]
                                  << " layers";
#endif
  layerType_ = dbl_to_int(vArgs["LayerType"]);
  layerSense_ = dbl_to_int(vArgs["LayerSense"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << layerType_.size() << " layers";
  for (unsigned int i = 0; i < layerType_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType_[i] << " sensitive class "
                                  << layerSense_[i];
#endif
  zMinBlock_ = nArgs["zMinBlock"];
  rMaxFine_ = nArgs["rMaxFine"];
  waferW_ = nArgs["waferW"];
  sectors_ = (int)(nArgs["Sectors"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: zStart " << zMinBlock_ << " rFineCoarse " << rMaxFine_
                                << " wafer width " << waferW_ << " sectors " << sectors_;
#endif
  slopeB_ = vArgs["SlopeBottom"];
  slopeT_ = vArgs["SlopeTop"];
  zFront_ = vArgs["ZFront"];
  rMaxFront_ = vArgs["RMaxFront"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: Bottom slopes " << slopeB_[0] << ":" << slopeB_[1] << " and "
                                << slopeT_.size() << " slopes for top";
  for (unsigned int i = 0; i < slopeT_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFront_[i] << " Rmax " << rMaxFront_[i] << " Slope "
                                  << slopeT_[i];
#endif
  idNameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: NameSpace " << idNameSpace_;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHGCalModule methods...
////////////////////////////////////////////////////////////////////

void DDHGCalModule::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalModule...";
#endif
  copies_.clear();
  constructLayers(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << copies_.size() << " different wafer copy numbers";
  int k(0);
  for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k)
    edm::LogVerbatim("HGCalGeom") << "Copy[" << k << "] : " << (*itr);
#endif
  copies_.clear();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalModule construction ...";
#endif
}

void DDHGCalModule::constructLayers(const DDLogicalPart& module, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: \t\tInside Layers";
#endif
  double zi(zMinBlock_);
  int laymin(0);
  const double tol(0.01);
  for (unsigned int i = 0; i < layers_.size(); i++) {
    double zo = zi + layerThick_[i];
    double routF = rMax(zi);
    int laymax = laymin + layers_[i];
    double zz = zi;
    double thickTot(0);
    for (int ly = laymin; ly < laymax; ++ly) {
      int ii = layerType_[ly];
      int copy = copyNumber_[ii];
      double rinB = (layerSense_[ly] == 0) ? (zo * slopeB_[0]) : (zo * slopeB_[1]);
      zz += (0.5 * thick_[ii]);
      thickTot += thick_[ii];

      std::string name = "HGCal" + names_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: Layer " << ly << ":" << ii << " Front " << zi << ", " << routF
                                    << " Back " << zo << ", " << rinB << " superlayer thickness " << layerThick_[i];
#endif
      DDName matName(DDSplit(materials_[ii]).first, DDSplit(materials_[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense_[ly] == 0) {
        double alpha = 1._pi / sectors_;
        double rmax = routF * cos(alpha) - tol;
        std::vector<double> pgonZ, pgonRin, pgonRout;
        pgonZ.emplace_back(-0.5 * thick_[ii]);
        pgonZ.emplace_back(0.5 * thick_[ii]);
        pgonRin.emplace_back(rinB);
        pgonRin.emplace_back(rinB);
        pgonRout.emplace_back(rmax);
        pgonRout.emplace_back(rmax);
        DDSolid solid =
            DDSolidFactory::polyhedra(DDName(name, idNameSpace_), sectors_, -alpha, 2._pi, pgonZ, pgonRin, pgonRout);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << solid.name() << " polyhedra of " << sectors_
                                      << " sectors covering " << convertRadToDeg(-alpha) << ":"
                                      << (360.0 + convertRadToDeg(-alpha)) << " with " << pgonZ.size() << " sections";
        for (unsigned int k = 0; k < pgonZ.size(); ++k)
          edm::LogVerbatim("HGCalGeom") << "[" << k << "] z " << pgonZ[k] << " R " << pgonRin[k] << ":" << pgonRout[k];
#endif
      } else {
        DDSolid solid = DDSolidFactory::tubs(DDName(name, idNameSpace_), 0.5 * thick_[ii], rinB, routF, 0.0, 2._pi);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << solid.name() << " Tubs made of " << matName
                                      << " of dimensions " << rinB << ", " << routF << ", " << 0.5 * thick_[ii]
                                      << ", 0.0, 360.0";
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModule test position in: " << glog.name() << " number " << copy;
#endif
        positionSensitive(glog, rinB, routF, cpv);
      }
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << glog.name() << " number " << copy << " positioned in "
                                    << module.name() << " at " << r1 << " with " << rot;
#endif
      zz += (0.5 * thick_[ii]);
    }  // End of loop over layers in a block
    zi = zo;
    laymin = laymax;
    if (fabs(thickTot - layerThick_[i]) > tol_) {
      if (thickTot > layerThick_[i]) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " is smaller than thickness "
                                   << thickTot << " of all its components **** ERROR ****\n";
      } else {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " does not match with "
                                     << thickTot << " of the components\n";
      }
    }
  }  // End of loop over blocks
}

double DDHGCalModule::rMax(double z) {
  double r(0);
#ifdef EDM_ML_DEBUG
  unsigned int ik(0);
#endif
  for (unsigned int k = 0; k < slopeT_.size(); ++k) {
    if (z < zFront_[k])
      break;
    r = rMaxFront_[k] + (z - zFront_[k]) * slopeT_[k];
#ifdef EDM_ML_DEBUG
    ik = k;
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "rMax : " << z << ":" << ik << ":" << r;
#endif
  return r;
}

void DDHGCalModule::positionSensitive(DDLogicalPart& glog, double rin, double rout, DDCompactView& cpv) {
  double dx = 0.5 * waferW_;
  double dy = 3.0 * dx * tan(30._deg);
  double rr = 2.0 * dx * tan(30._deg);
  int ncol = (int)(2.0 * rout / waferW_) + 1;
  int nrow = (int)(rout / (waferW_ * tan(30._deg))) + 1;
  int incm(0), inrm(0);
#ifdef EDM_ML_DEBUG
  int kount(0), ntot(0), nin(0), nfine(0), ncoarse(0);
  edm::LogVerbatim("HGCalGeom") << glog.ddname() << " rout " << rout << " Row " << nrow << " Column " << ncol;
#endif
  for (int nr = -nrow; nr <= nrow; ++nr) {
    int inr = (nr >= 0) ? nr : -nr;
    for (int nc = -ncol; nc <= ncol; ++nc) {
      int inc = (nc >= 0) ? nc : -nc;
      if (inr % 2 == inc % 2) {
        double xpos = nc * dx;
        double ypos = nr * dy;
        auto const& corner = HGCalGeomTools::waferCorner(xpos, ypos, dx, rr, rin, rout, true);
#ifdef EDM_ML_DEBUG
        ++ntot;
#endif
        if (corner.first > 0) {
          int copy = HGCalTypes::packTypeUV(0, nc, nr);
          if (inc > incm)
            incm = inc;
          if (inr > inrm)
            inrm = inr;
#ifdef EDM_ML_DEBUG
          kount++;
#endif
          if (copies_.count(copy) == 0)
            copies_.insert(copy);
          if (corner.first == (int)(HGCalParameters::k_CornerSize)) {
            double rpos = std::sqrt(xpos * xpos + ypos * ypos);
            DDTranslation tran(xpos, ypos, 0.0);
            DDRotation rotation;
#ifdef EDM_ML_DEBUG
            ++nin;
#endif
            DDName name = (rpos < rMaxFine_) ? DDName(DDSplit(wafer_[0]).first, DDSplit(wafer_[0]).second)
                                             : DDName(DDSplit(wafer_[1]).first, DDSplit(wafer_[1]).second);
            cpv.position(name, glog.ddname(), copy, tran, rotation);
#ifdef EDM_ML_DEBUG
            if (rpos < rMaxFine_)
              ++nfine;
            else
              ++ncoarse;
            edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << name << " number " << copy << " positioned in "
                                          << glog.ddname() << " at " << tran << " with " << rotation;
#endif
          }
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: # of columns " << incm << " # of rows " << inrm << " and " << nin
                                << ":" << kount << ":" << ntot << " wafers (" << nfine << ":" << ncoarse << ") for "
                                << glog.ddname() << " R " << rin << ":" << rout;
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalModule, "hgcal:DDHGCalModule");
