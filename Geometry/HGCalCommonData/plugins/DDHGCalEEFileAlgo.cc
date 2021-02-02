///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalEEFileAlgo.cc
// Description: Geometry factory class for HGCal (EE and HESil) using
//              information from the file
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/GeantUnits.h"
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
#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

class DDHGCalEEFileAlgo : public DDAlgorithm {
public:
  DDHGCalEEFileAlgo();

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:
  void constructLayers(const DDLogicalPart&, DDCompactView& cpv);
  void positionSensitive(
      const DDLogicalPart& glog, double rin, double rout, double zpos, int layertype, int layer, DDCompactView& cpv);

private:
  HGCalGeomTools geomTools_;

  static constexpr double tol1_ = 0.01;
  static constexpr double tol2_ = 0.00001;

  std::vector<std::string> wafers_;     // Wafers
  std::vector<std::string> materials_;  // Materials
  std::vector<std::string> names_;      // Names
  std::vector<double> thick_;           // Thickness of the material
  std::vector<int> copyNumber_;         // Initial copy numbers
  std::vector<int> layers_;             // Number of layers in a section
  std::vector<double> layerThick_;      // Thickness of each section
  std::vector<int> layerType_;          // Type of the layer
  std::vector<int> layerSense_;         // Content of a layer (sensitive?)
  std::vector<int> layerCenter_;        // Centering of the wafers
  int firstLayer_;                      // Copy # of the first sensitive layer
  int absorbMode_;                      // Absorber mode
  int sensitiveMode_;                   // Sensitive mode
  double zMinBlock_;                    // Starting z-value of the block
  std::vector<int> waferIndex_;         // Wafer index for the types
  std::vector<int> waferProperty_;      // Wafer property
  double waferSize_;                    // Width of the wafer
  double waferSepar_;                   // Sensor separation
  int sectors_;                         // Sectors
  std::vector<double> slopeB_;          // Slope at the lower R
  std::vector<double> zFrontB_;         // Starting Z values for the slopes
  std::vector<double> rMinFront_;       // Corresponding rMin's
  std::vector<double> slopeT_;          // Slopes at the larger R
  std::vector<double> zFrontT_;         // Starting Z values for the slopes
  std::vector<double> rMaxFront_;       // Corresponding rMax's
  std::string nameSpace_;               // Namespace of this and ALL sub-parts
  std::unordered_set<int> copies_;      // List of copy #'s
  double alpha_, cosAlpha_;
};

DDHGCalEEFileAlgo::DDHGCalEEFileAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: Creating an instance";
#endif
}

void DDHGCalEEFileAlgo::initialize(const DDNumericArguments& nArgs,
                                   const DDVectorArguments& vArgs,
                                   const DDMapArguments&,
                                   const DDStringArguments& sArgs,
                                   const DDStringVectorArguments& vsArgs) {
  wafers_ = vsArgs["WaferNames"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << wafers_.size() << " wafers";
  for (unsigned int i = 0; i < wafers_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafers_[i];
#endif
  materials_ = vsArgs["MaterialNames"];
  names_ = vsArgs["VolumeNames"];
  thick_ = vArgs["Thickness"];
  copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << materials_.size() << " types of volumes";
  for (unsigned int i = 0; i < names_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names_[i] << " of thickness " << thick_[i]
                                  << " filled with " << materials_[i] << " first copy number " << copyNumber_[i];
#endif
  layers_ = dbl_to_int(vArgs["Layers"]);
  layerThick_ = vArgs["LayerThick"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "There are " << layers_.size() << " blocks";
  for (unsigned int i = 0; i < layers_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick_[i] << " with " << layers_[i]
                                  << " layers";
#endif
  layerType_ = dbl_to_int(vArgs["LayerType"]);
  layerSense_ = dbl_to_int(vArgs["LayerSense"]);
  firstLayer_ = (int)(nArgs["FirstLayer"]);
  absorbMode_ = (int)(nArgs["AbsorberMode"]);
  sensitiveMode_ = (int)(nArgs["SensitiveMode"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "First Layer " << firstLayer_ << " and "
                                << "Absober:Sensitive mode " << absorbMode_ << ":" << sensitiveMode_;
#endif
  layerCenter_ = dbl_to_int(vArgs["LayerCenter"]);
#ifdef EDM_ML_DEBUG
  for (unsigned int i = 0; i < layerCenter_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "LayerCenter [" << i << "] " << layerCenter_[i];
#endif
  if (firstLayer_ > 0) {
    for (unsigned int i = 0; i < layerType_.size(); ++i) {
      if (layerSense_[i] > 0) {
        int ii = layerType_[i];
        copyNumber_[ii] = firstLayer_;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "First copy number for layer type " << i << ":" << ii << " with "
                                      << materials_[ii] << " changed to " << copyNumber_[ii];
#endif
        break;
      }
    }
  } else {
    firstLayer_ = 1;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "There are " << layerType_.size() << " layers";
  for (unsigned int i = 0; i < layerType_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType_[i] << " sensitive class "
                                  << layerSense_[i];
#endif
  zMinBlock_ = nArgs["zMinBlock"];
  waferSize_ = nArgs["waferSize"];
  waferSepar_ = nArgs["SensorSeparation"];
  sectors_ = (int)(nArgs["Sectors"]);
  alpha_ = (1._pi) / sectors_;
  cosAlpha_ = cos(alpha_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "zStart " << zMinBlock_ << " wafer width " << waferSize_ << " separations "
                                << waferSepar_ << " sectors " << sectors_ << ":" << convertRadToDeg(alpha_) << ":"
                                << cosAlpha_;
#endif
  waferIndex_ = dbl_to_int(vArgs["WaferIndex"]);
  waferProperty_ = dbl_to_int(vArgs["WaferProperties"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "waferProperties with " << waferIndex_.size() << " entries";
  for (unsigned int k = 0; k < waferIndex_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << waferIndex_[k] << " ("
                                  << HGCalWaferIndex::waferLayer(waferIndex_[k]) << ", "
                                  << HGCalWaferIndex::waferU(waferIndex_[k]) << ", "
                                  << HGCalWaferIndex::waferV(waferIndex_[k]) << ") : ("
                                  << HGCalProperty::waferThick(waferProperty_[k]) << ":"
                                  << HGCalProperty::waferPartial(waferProperty_[k]) << ":"
                                  << HGCalProperty::waferOrient(waferProperty_[k]) << ")";
#endif
  slopeB_ = vArgs["SlopeBottom"];
  zFrontB_ = vArgs["ZFrontBottom"];
  rMinFront_ = vArgs["RMinFront"];
  slopeT_ = vArgs["SlopeTop"];
  zFrontT_ = vArgs["ZFrontTop"];
  rMaxFront_ = vArgs["RMaxFront"];
#ifdef EDM_ML_DEBUG
  for (unsigned int i = 0; i < slopeB_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Bottom Block [" << i << "] Zmin " << zFrontB_[i] << " Rmin " << rMinFront_[i]
                                  << " Slope " << slopeB_[i];
  for (unsigned int i = 0; i < slopeT_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Top Block [" << i << "] Zmin " << zFrontT_[i] << " Rmax " << rMaxFront_[i]
                                  << " Slope " << slopeT_[i];
#endif
  nameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: NameSpace " << nameSpace_;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHGCalEEFileAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHGCalEEFileAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalEEFileAlgo...";
  copies_.clear();
#endif
  constructLayers(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << copies_.size() << " different wafer copy numbers";
  int k(0);
  for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k) {
    edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
  }
  copies_.clear();
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalEEFileAlgo construction...";
#endif
}

void DDHGCalEEFileAlgo::constructLayers(const DDLogicalPart& module, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: \t\tInside Layers";
#endif
  double zi(zMinBlock_);
  int laymin(0);
  for (unsigned int i = 0; i < layers_.size(); i++) {
    double zo = zi + layerThick_[i];
    double routF = HGCalGeomTools::radius(zi, zFrontT_, rMaxFront_, slopeT_);
    int laymax = laymin + layers_[i];
    double zz = zi;
    double thickTot(0);
    for (int ly = laymin; ly < laymax; ++ly) {
      int ii = layerType_[ly];
      int copy = copyNumber_[ii];
      double hthick = 0.5 * thick_[ii];
      double rinB = HGCalGeomTools::radius(zo - tol1_, zFrontB_, rMinFront_, slopeB_);
      zz += hthick;
      thickTot += thick_[ii];

      std::string name = names_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: Layer " << ly << ":" << ii << " Front " << zi << ", "
                                    << routF << " Back " << zo << ", " << rinB << " superlayer thickness "
                                    << layerThick_[i];
#endif
      DDName matName(DDSplit(materials_[ii]).first, DDSplit(materials_[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense_[ly] < 1) {
        std::vector<double> pgonZ, pgonRin, pgonRout;
        if (layerSense_[ly] == 0 || absorbMode_ == 0) {
          double rmax = routF * cosAlpha_ - tol1_;
          pgonZ.emplace_back(-hthick);
          pgonZ.emplace_back(hthick);
          pgonRin.emplace_back(rinB);
          pgonRin.emplace_back(rinB);
          pgonRout.emplace_back(rmax);
          pgonRout.emplace_back(rmax);
        } else {
          HGCalGeomTools::radius(zz - hthick,
                                 zz + hthick,
                                 zFrontB_,
                                 rMinFront_,
                                 slopeB_,
                                 zFrontT_,
                                 rMaxFront_,
                                 slopeT_,
                                 -layerSense_[ly],
                                 pgonZ,
                                 pgonRin,
                                 pgonRout);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: z " << (zz - hthick) << ":" << (zz + hthick) << " with "
                                        << pgonZ.size() << " palnes";
          for (unsigned int isec = 0; isec < pgonZ.size(); ++isec)
            edm::LogVerbatim("HGCalGeom")
                << "[" << isec << "] z " << pgonZ[isec] << " R " << pgonRin[isec] << ":" << pgonRout[isec];
#endif
          for (unsigned int isec = 0; isec < pgonZ.size(); ++isec) {
            pgonZ[isec] -= zz;
            pgonRout[isec] = pgonRout[isec] * cosAlpha_ - tol1_;
          }
        }
        DDSolid solid =
            DDSolidFactory::polyhedra(DDName(name, nameSpace_), sectors_, -alpha_, 2._pi, pgonZ, pgonRin, pgonRout);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << solid.name() << " polyhedra of " << sectors_
                                      << " sectors covering " << convertRadToDeg(-alpha_) << ":"
                                      << convertRadToDeg(-alpha_ + 2._pi) << " with " << pgonZ.size()
                                      << " sections and filled with " << matName << ":" << &matter;
        for (unsigned int k = 0; k < pgonZ.size(); ++k)
          edm::LogVerbatim("HGCalGeom") << "[" << k << "] z " << pgonZ[k] << " R " << pgonRin[k] << ":" << pgonRout[k];
#endif
      } else {
        double rins =
            (sensitiveMode_ < 1) ? rinB : HGCalGeomTools::radius(zz + hthick - tol1_, zFrontB_, rMinFront_, slopeB_);
        double routs =
            (sensitiveMode_ < 1) ? routF : HGCalGeomTools::radius(zz - hthick, zFrontT_, rMaxFront_, slopeT_);
        DDSolid solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthick, rins, routs, 0.0, 2._pi);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << solid.name() << " Tubs made of " << matName << ":"
                                      << &matter << " of dimensions " << rinB << ":" << rins << ", " << routF << ":"
                                      << routs << ", " << hthick << ", 0.0, 360.0 and position " << glog.name()
                                      << " number " << copy << ":" << layerCenter_[copy - firstLayer_];
#endif
        positionSensitive(glog, rins, routs, zz, layerSense_[ly], (copy - firstLayer_), cpv);
      }
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << glog.name() << " number " << copy << " positioned in "
                                    << module.name() << " at " << r1 << " with " << rot;
#endif
      zz += hthick;
    }  // End of loop over layers in a block
    zi = zo;
    laymin = laymax;
    // Make consistency check of all the partitions of the block
    if (std::abs(thickTot - layerThick_[i]) >= tol2_) {
      if (thickTot > layerThick_[i]) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " is smaller than " << thickTot
                                   << ": thickness of all its components **** ERROR ****";
      } else {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " does not match with "
                                     << thickTot << " of the components";
      }
    }
  }  // End of loop over blocks
}

void DDHGCalEEFileAlgo::positionSensitive(
    const DDLogicalPart& glog, double rin, double rout, double zpos, int layertype, int layer, DDCompactView& cpv) {
  static const double sqrt3 = std::sqrt(3.0);
  int layercenter = layerCenter_[layer];
  double r = 0.5 * (waferSize_ + waferSepar_);
  double R = 2.0 * r / sqrt3;
  double dy = 0.75 * R;
  int N = (int)(0.5 * rout / r) + 2;
  const auto& xyoff = geomTools_.shiftXY(layercenter, (waferSize_ + waferSepar_));
#ifdef EDM_ML_DEBUG
  int ium(0), ivm(0), iumAll(0), ivmAll(0), kount(0), ntot(0), nin(0);
  std::vector<int> ntype(6, 0);
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << glog.ddname() << " rin:rout " << rin << ":" << rout
                                << " zpos " << zpos << " N " << N << " for maximum u, v;  r " << r << " R " << R
                                << " dy " << dy << " Shift " << xyoff.first << ":" << xyoff.second << " WaferSize "
                                << (waferSize_ + waferSepar_);
#endif
  for (int u = -N; u <= N; ++u) {
    for (int v = -N; v <= N; ++v) {
#ifdef EDM_ML_DEBUG
      int iu = std::abs(u);
      int iv = std::abs(v);
#endif
      int nr = 2 * v;
      int nc = -2 * u + v;
      double xpos = xyoff.first + nc * r;
      double ypos = xyoff.second + nr * dy;
      const auto& corner = HGCalGeomTools::waferCorner(xpos, ypos, r, R, rin, rout, false);
#ifdef EDM_ML_DEBUG
      ++ntot;
      if (((corner.first <= 0) && std::abs(u) < 5 && std::abs(v) < 5) || (std::abs(u) < 2 && std::abs(v) < 2)) {
        edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << glog.ddname() << " R " << rin << ":" << rout
                                      << "\n Z " << zpos << " LayerType " << layertype << " u " << u << " v " << v
                                      << " with " << corner.first << " corners";
      }
#endif
      int indx = HGCalWaferIndex::waferIndex((layer + firstLayer_), u, v, false);
      int type = HGCalWaferType::getType(indx, waferIndex_, waferProperty_);
      if (corner.first > 0 && type >= 0) {
        int copy = HGCalTypes::packTypeUV(type, u, v);
        if (layertype > 1)
          type += 3;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << " DDHGCalHEFileAlgo: " << wafers_[type] << " number " << copy << " type "
                                      << type << " layer:u:v:indx " << (layer + firstLayer_) << ":" << u << ":" << v
                                      << ":" << indx;
        if (iu > ium)
          ium = iu;
        if (iv > ivm)
          ivm = iv;
        kount++;
        if (copies_.count(copy) == 0)
          copies_.insert(copy);
#endif
        if (corner.first == (int)(HGCalParameters::k_CornerSize)) {
#ifdef EDM_ML_DEBUG
          if (iu > iumAll)
            iumAll = iu;
          if (iv > ivmAll)
            ivmAll = iv;
          ++nin;
#endif
          DDTranslation tran(xpos, ypos, 0.0);
          DDRotation rotation;
          DDName name = DDName(DDSplit(wafers_[type]).first, DDSplit(wafers_[type]).second);
          cpv.position(name, glog.ddname(), copy, tran, rotation);
#ifdef EDM_ML_DEBUG
          ++ntype[type];
          edm::LogVerbatim("HGCalGeom") << " DDHGCalEEFileAlgo: " << name << " number " << copy << " type " << layertype
                                        << ":" << type << " positioned in " << glog.ddname() << " at " << tran
                                        << " with " << rotation;
#endif
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: Maximum # of u " << ium << ":" << iumAll << " # of v " << ivm
                                << ":" << ivmAll << " and " << nin << ":" << kount << ":" << ntot << " wafers ("
                                << ntype[0] << ":" << ntype[1] << ":" << ntype[2] << ":" << ntype[3] << ":" << ntype[4]
                                << ":" << ntype[5] << ") for " << glog.ddname() << " R " << rin << ":" << rout;
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalEEFileAlgo, "hgcal:DDHGCalEEFileAlgo");
