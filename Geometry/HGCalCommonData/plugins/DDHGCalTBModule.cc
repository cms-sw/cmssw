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
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

class DDHGCalTBModule : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalTBModule();  // const std::string & name);
  ~DDHGCalTBModule() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:
  void constructLayers(const DDLogicalPart&, DDCompactView& cpv);
  double rMax(double z);
  void positionSensitive(DDLogicalPart& glog, int type, double rin, double rout, DDCompactView& cpv);

private:
  std::vector<std::string> wafer_;      // Wafers
  std::vector<std::string> covers_;     // Insensitive layers of hexagonal size
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
  double waferGap_;                     // Gap between 2 wafers
  double absorbW_;                      // Width of the absorber
  double absorbH_;                      // Height of the absorber
  int sectors_;                         // Sectors
  std::vector<double> slopeB_;          // Slope at the lower R
  std::vector<double> slopeT_;          // Slopes at the larger R
  std::vector<double> zFront_;          // Starting Z values for the slopes
  std::vector<double> rMaxFront_;       // Corresponding rMax's
  std::string idName_;                  // Name of the "parent" volume.
  std::string idNameSpace_;             // Namespace of this and ALL sub-parts
  std::unordered_set<int> copies_;      // List of copy #'s
};

DDHGCalTBModule::DDHGCalTBModule() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule info: Creating an instance";
#endif
}

DDHGCalTBModule::~DDHGCalTBModule() {}

void DDHGCalTBModule::initialize(const DDNumericArguments& nArgs,
                                 const DDVectorArguments& vArgs,
                                 const DDMapArguments&,
                                 const DDStringArguments& sArgs,
                                 const DDStringVectorArguments& vsArgs) {
  wafer_ = vsArgs["WaferName"];
  covers_ = vsArgs["CoverName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << wafer_.size() << " wafers";
  unsigned int i(0);
  for (auto wafer : wafer_) {
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafer;
    ++i;
  }
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << covers_.size() << " covers";
  i = 0;
  for (auto cover : covers_) {
    edm::LogVerbatim("HGCalGeom") << "Cover[" << i << "] " << cover;
    ++i;
  }
#endif
  materials_ = vsArgs["MaterialNames"];
  names_ = vsArgs["VolumeNames"];
  thick_ = vArgs["Thickness"];
  for (unsigned int i = 0; i < materials_.size(); ++i) {
    copyNumber_.emplace_back(1);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << materials_.size() << " types of volumes";
  for (unsigned int i = 0; i < names_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names_[i] << " of thickness " << thick_[i]
                                  << " filled with " << materials_[i] << " first copy number " << copyNumber_[i];
#endif
  layers_ = dbl_to_int(vArgs["Layers"]);
  layerThick_ = vArgs["LayerThick"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << layers_.size() << " blocks";
  for (unsigned int i = 0; i < layers_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick_[i] << " with " << layers_[i]
                                  << " layers";
#endif
  layerType_ = dbl_to_int(vArgs["LayerType"]);
  layerSense_ = dbl_to_int(vArgs["LayerSense"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << layerType_.size() << " layers";
  for (unsigned int i = 0; i < layerType_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType_[i] << " sensitive class "
                                  << layerSense_[i];
#endif
  zMinBlock_ = nArgs["zMinBlock"];
  rMaxFine_ = nArgs["rMaxFine"];
  waferW_ = nArgs["waferW"];
  waferGap_ = nArgs["waferGap"];
  absorbW_ = nArgs["absorberW"];
  absorbH_ = nArgs["absorberH"];
  sectors_ = (int)(nArgs["Sectors"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: zStart " << zMinBlock_ << " rFineCoarse " << rMaxFine_
                                << " wafer width " << waferW_ << " gap among wafers " << waferGap_ << " absorber width "
                                << absorbW_ << " absorber height " << absorbH_ << " sectors " << sectors_;
#endif
  slopeB_ = vArgs["SlopeBottom"];
  slopeT_ = vArgs["SlopeTop"];
  zFront_ = vArgs["ZFront"];
  rMaxFront_ = vArgs["RMaxFront"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: Bottom slopes " << slopeB_[0] << ":" << slopeB_[1] << " and "
                                << slopeT_.size() << " slopes for top";
  for (unsigned int i = 0; i < slopeT_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFront_[i] << " Rmax " << rMaxFront_[i] << " Slope "
                                  << slopeT_[i];
#endif
  idNameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: NameSpace " << idNameSpace_;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHGCalTBModule methods...
////////////////////////////////////////////////////////////////////

void DDHGCalTBModule::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalTBModule...";
#endif
  copies_.clear();
  constructLayers(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << copies_.size() << " different wafer copy numbers";
#endif
  copies_.clear();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalTBModule construction ...";
#endif
}

void DDHGCalTBModule::constructLayers(const DDLogicalPart& module, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule test: \t\tInside Layers";
#endif
  double zi(zMinBlock_);
  int laymin(0);
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
      edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: Layer " << ly << ":" << ii << " Front " << zi << ", " << routF
                                    << " Back " << zo << ", " << rinB << " superlayer thickness " << layerThick_[i];
#endif
      DDName matName(DDSplit(materials_[ii]).first, DDSplit(materials_[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense_[ly] == 0) {
        DDSolid solid = DDSolidFactory::box(DDName(name, idNameSpace_), absorbW_, absorbH_, 0.5 * thick_[ii]);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule test: " << solid.name() << " box of dimension " << absorbW_
                                      << ":" << absorbH_ << ":" << 0.5 * thick_[ii];
#endif
      } else {
        DDSolid solid = DDSolidFactory::tubs(DDName(name, idNameSpace_), 0.5 * thick_[ii], rinB, routF, 0.0, 2._pi);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << solid.name() << " Tubs made of " << matName
                                      << " of dimensions " << rinB << ", " << routF << ", " << 0.5 * thick_[ii]
                                      << ", 0.0, 360.0";
#endif
        positionSensitive(glog, layerSense_[ly], rinB, routF, cpv);
      }
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule test: " << glog.name() << " number " << copy
                                    << " positioned in " << module.name() << " at " << r1 << " with " << rot;
#endif
      zz += (0.5 * thick_[ii]);
    }  // End of loop over layers in a block
    zi = zo;
    laymin = laymax;
    if (fabs(thickTot - layerThick_[i]) < 0.00001) {
    } else if (thickTot > layerThick_[i]) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " is smaller than thickness "
                                 << thickTot << " of all its components **** ERROR ****\n";
    } else if (thickTot < layerThick_[i]) {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " does not match with "
                                   << thickTot << " of the components\n";
    }
  }  // End of loop over blocks
}

double DDHGCalTBModule::rMax(double z) {
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

void DDHGCalTBModule::positionSensitive(DDLogicalPart& glog, int type, double rin, double rout, DDCompactView& cpv) {
  double ww = (waferW_ + waferGap_);
  double dx = 0.5 * ww;
  double dy = 3.0 * dx * tan(30._deg);
  double rr = 2.0 * dx * tan(30._deg);
  int ncol = (int)(2.0 * rout / ww) + 1;
  int nrow = (int)(rout / (ww * tan(30._deg))) + 1;
  int incm(0), inrm(0);
  double xc[6], yc[6];
#ifdef EDM_ML_DEBUG
  int kount(0);
  edm::LogVerbatim("HGCalGeom") << glog.ddname() << " rout " << rout << " Row " << nrow << " Column " << ncol;
#endif
  for (int nr = -nrow; nr <= nrow; ++nr) {
    int inr = (nr >= 0) ? nr : -nr;
    for (int nc = -ncol; nc <= ncol; ++nc) {
      int inc = (nc >= 0) ? nc : -nc;
      if (inr % 2 == inc % 2) {
        double xpos = nc * dx;
        double ypos = nr * dy;
        xc[0] = xpos + dx;
        yc[0] = ypos - 0.5 * rr;
        xc[1] = xpos + dx;
        yc[1] = ypos + 0.5 * rr;
        xc[2] = xpos;
        yc[2] = ypos + rr;
        xc[3] = xpos - dx;
        yc[3] = ypos + 0.5 * rr;
        xc[4] = xpos + dx;
        yc[4] = ypos - 0.5 * rr;
        xc[5] = xpos;
        yc[5] = ypos - rr;
        bool cornerAll(true);
        for (int k = 0; k < 6; ++k) {
          double rpos = std::sqrt(xc[k] * xc[k] + yc[k] * yc[k]);
          if (rpos < rin || rpos > rout)
            cornerAll = false;
        }
        if (cornerAll) {
          double rpos = std::sqrt(xpos * xpos + ypos * ypos);
          DDTranslation tran(xpos, ypos, 0.0);
          DDRotation rotation;
          int copy = HGCalTypes::packTypeUV(0, nc, nr);
          DDName name;
          if (type == 1) {
            name = (rpos < rMaxFine_) ? DDName(DDSplit(wafer_[0]).first, DDSplit(wafer_[0]).second)
                                      : DDName(DDSplit(wafer_[1]).first, DDSplit(wafer_[1]).second);
          } else {
            name = DDName(DDSplit(covers_[type - 2]).first, DDSplit(covers_[type - 2]).second);
          }
          cpv.position(name, glog.ddname(), copy, tran, rotation);
          if (inc > incm)
            incm = inc;
          if (inr > inrm)
            inrm = inr;
          if (copies_.count(copy) == 0 && type == 1)
            copies_.insert(copy);
#ifdef EDM_ML_DEBUG
          kount++;
          edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << name << " number " << copy << " positioned in "
                                        << glog.ddname() << " at " << tran << " with " << rotation;
#endif
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: # of columns " << incm << " # of rows " << inrm << " and " << kount
                                << " wafers for " << glog.ddname();
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalTBModule, "hgcal:DDHGCalTBModule");
