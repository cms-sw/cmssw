/*
 * DDHGCalTBModuleX.cc
 *
 *  Created on: 27-Aug-2019
 *      Author: S. Banerjee
 */

#include <unordered_set>

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

namespace DDHGCalGeom {
  void constructLayers(const cms::DDNamespace& ns,
                       const std::vector<std::string>& wafers,
                       const std::vector<std::string>& covers,
                       const std::vector<int>& layerType,
                       const std::vector<int>& layerSense,
                       const std::vector<int>& maxModule,
                       const std::vector<std::string>& names,
                       const std::vector<std::string>& materials,
                       std::vector<int>& copyNumber,
                       const std::vector<double>& layerThick,
                       const double& absorbW,
                       const double& absorbH,
                       const double& waferTot,
                       const double& rMax,
                       const double& rMaxFine,
                       std::unordered_set<int>& copies,
                       int firstLayer,
                       int lastLayer,
                       double zFront,
                       double totalWidth,
                       bool ignoreCenter,
                       dd4hep::Volume& module) {
    static constexpr double tolerance = 0.00001;
    static const double tan30deg = tan(30._deg);
    double zi(zFront), thickTot(0);
    for (int ly = firstLayer; ly <= lastLayer; ++ly) {
      int ii = layerType[ly];
      int copy = copyNumber[ii];
      double zz = zi + (0.5 * layerThick[ii]);
      double zo = zi + layerThick[ii];
      thickTot += layerThick[ii];

      std::string name = "HGCal" + names[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: " << name << " Layer " << ly << ":" << ii << " Z "
                                    << convertCmToMm(zi) << ":" << convertCmToMm(zo) << " Thick "
                                    << convertCmToMm(layerThick[ii]) << " Sense " << layerSense[ly];
#endif
      dd4hep::Material matter = ns.material(materials[ii]);
      dd4hep::Volume glog;
      if (layerSense[ly] == 0) {
        dd4hep::Solid solid = dd4hep::Box(absorbW, absorbH, 0.5 * layerThick[ii]);
        ns.addSolidNS(ns.prepend(name), solid);
        glog = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: " << solid.name() << " box of dimension "
                                      << convertCmToMm(absorbW) << ":" << convertCmToMm(absorbH) << ":"
                                      << convertCmToMm(0.5 * layerThick[ii]);
#endif
        dd4hep::Position r1(0, 0, zz);
        module.placeVolume(glog, copy, r1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: " << glog.name() << " number " << copy << " positioned in "
                                      << module.name() << " at " << r1 << " with no rotation";
#endif
      } else if (layerSense[ly] > 0) {
        double dx = 0.5 * waferTot;
        double dy = 3.0 * dx * tan30deg;
        double rr = 2.0 * dx * tan30deg;
        int ncol = (int)(2.0 * rMax / waferTot) + 1;
        int nrow = (int)(rMax / (waferTot * tan30deg)) + 1;
#ifdef EDM_ML_DEBUG
        int incm(0), inrm(0);
        edm::LogVerbatim("HGCalGeom") << module.name() << " Copy " << copy << " Type " << layerSense[ly] << " rout "
                                      << convertCmToMm(rMax) << " Row " << nrow << " column " << ncol << " ncrMax "
                                      << maxModule[ly] << " Z " << convertCmToMm(zz) << " Center " << ignoreCenter
                                      << " name " << name << " matter " << matter.name();
        int kount(0);
#endif
        if (maxModule[ly] >= 0) {
          nrow = std::min(nrow, maxModule[ly]);
          ncol = std::min(ncol, maxModule[ly]);
        }
        for (int nr = -nrow; nr <= nrow; ++nr) {
          int inr = std::abs(nr);
          for (int nc = -ncol; nc <= ncol; ++nc) {
            int inc = std::abs(nc);
            if ((inr % 2 == inc % 2) && (!ignoreCenter || nc != 0 || nr != 0)) {
              double xpos = nc * dx;
              double ypos = nr * dy;
              double xc[6], yc[6];
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
                if (rpos > rMax)
                  cornerAll = false;
              }
              if (cornerAll) {
                double rpos = std::sqrt(xpos * xpos + ypos * ypos);
                dd4hep::Position tran(xpos, ypos, zz);
                int copyx = inr * 100 + inc;
                if (nc < 0)
                  copyx += 10000;
                if (nr < 0)
                  copyx += 100000;
                if (layerSense[ly] == 1) {
                  dd4hep::Solid solid = ns.solid(covers[0]);
                  std::string name0 = name + "M" + std::to_string(copyx);
                  name0 = ns.prepend(name0);
                  dd4hep::Volume glog1 = dd4hep::Volume(name0, solid, matter);
                  module.placeVolume(glog1, copy, tran);
#ifdef EDM_ML_DEBUG
                  edm::LogVerbatim("HGCalGeom")
                      << "DDHGCalTBModuleX: " << glog1.name() << " number " << copy << " positioned in "
                      << module.name() << " at " << tran << " with no rotation";
#endif
                  dd4hep::Volume glog2 = (rpos < rMaxFine) ? ns.volume(wafers[0]) : ns.volume(wafers[1]);
                  glog1.placeVolume(glog2, copyx);
#ifdef EDM_ML_DEBUG
                  edm::LogVerbatim("HGCalGeom")
                      << "DDHGCalTBModuleX: " << glog2.name() << " number " << copyx << " positioned in "
                      << glog1.name() << " at (0, 0, 0) with no rotation";
#endif
                  if (layerSense[ly] == 1)
                    copies.insert(copy);
                } else {
                  dd4hep::Volume glog2 = ns.volume(covers[layerSense[ly] - 1]);
                  copyx += (copy * 1000000);
                  module.placeVolume(glog2, copyx, tran);
#ifdef EDM_ML_DEBUG
                  edm::LogVerbatim("HGCalGeom")
                      << "DDHGCalTBModuleX: " << glog2.name() << " number " << copyx << " positioned in "
                      << module.name() << " at " << tran << " with no rotation";
#endif
                }
#ifdef EDM_ML_DEBUG
                if (inc > incm)
                  incm = inc;
                if (inr > inrm)
                  inrm = inr;
                kount++;
#endif
              }
            }
          }
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: # of columns " << incm << " # of rows " << inrm << " and "
                                      << kount << " wafers for " << module.name();
#endif
      }
      ++copyNumber[ii];
      zi = zo;
    }  // End of loop over layers in a block

    if (fabs(thickTot - totalWidth) < tolerance) {
    } else if (thickTot > totalWidth) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << totalWidth << " is smaller than " << thickTot
                                 << ": total thickness of all its components in " << module.name() << " Layers "
                                 << firstLayer << ":" << lastLayer << ":" << ignoreCenter << "**** ERROR ****";
    } else if (thickTot < totalWidth) {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " << totalWidth << " does not match with " << thickTot
                                   << " of the components in " << module.name() << " Layers " << firstLayer << ":"
                                   << lastLayer << ":" << ignoreCenter;
    }
  }
}  // namespace DDHGCalGeom

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  const auto& wafers = args.value<std::vector<std::string> >("WaferName");  // Wafers
  const auto& covers = args.value<std::vector<std::string> >("CoverName");  // Insensitive layers of hexagonal size
  const auto& genMat = args.value<std::string>("GeneralMaterial");          // General material used for blocks
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: Material " << genMat << " with " << wafers.size() << " wafers";
  unsigned int i(0);
  for (auto wafer : wafers) {
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafer;
    ++i;
  }
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: " << covers.size() << " covers";
  i = 0;
  for (auto cover : covers) {
    edm::LogVerbatim("HGCalGeom") << "Cover[" << i << "] " << cover;
    ++i;
  }
#endif
  const auto& materials = args.value<std::vector<std::string> >("MaterialNames");  // Material names in each layer
  const auto& names = args.value<std::vector<std::string> >("VolumeNames");        // Names of each layer
  const auto& layerThick = args.value<std::vector<double> >("Thickness");          // Thickness of the material
  std::vector<int> copyNumber;                                                     // Copy numbers (initiated to 1)
  for (unsigned int k = 0; k < layerThick.size(); ++k) {
    copyNumber.emplace_back(1);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: " << materials.size() << " types of volumes";
  for (unsigned int i = 0; i < names.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names[i] << " of thickness "
                                  << convertCmToMm(layerThick[i]) << " filled with " << materials[i]
                                  << " first copy number " << copyNumber[i];
#endif
  const auto& blockThick = args.value<std::vector<double> >("BlockThick");   // Thickness of each section
  const auto& inOut = args.value<int>("InOut");                              // Number of inner+outer parts
  const auto& layerFrontIn = args.value<std::vector<int> >("LayerFrontIn");  // First layer index (inner) in block
  const auto& layerBackIn = args.value<std::vector<int> >("LayerBackIn");    // Last layer index (inner) in block
  std::vector<int> layerFrontOut;                                            // First layer index (outner) in block
  std::vector<int> layerBackOut;                                             // Last layer index (outner) in block
  if (inOut > 1) {
    layerFrontOut = args.value<std::vector<int> >("LayerFrontOut");
    layerBackOut = args.value<std::vector<int> >("LayerBackOut");
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: " << blockThick.size() << " blocks with in/out " << inOut;
  for (unsigned int i = 0; i < blockThick.size(); ++i) {
    if (inOut > 1)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << convertCmToMm(blockThick[i])
                                    << " with inner layers " << layerFrontIn[i] << ":" << layerBackIn[i]
                                    << " and outer layers " << layerFrontOut[i] << ":" << layerBackOut[i];
    else
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << convertCmToMm(blockThick[i])
                                    << " with inner layers " << layerFrontIn[i] << ":" << layerBackIn[i];
  }
#endif
  const auto& layerType = args.value<std::vector<int> >("LayerType");    // Type of the layer
  const auto& layerSense = args.value<std::vector<int> >("LayerSense");  // Content of a layer
  const auto& maxModule = args.value<std::vector<int> >("MaxModule");    // Maximum # of row/column
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: " << layerType.size() << " layers";
  for (unsigned int i = 0; i < layerType.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType[i] << " sensitive class "
                                  << layerSense[i] << " and " << maxModule[i] << " maximum row/columns";
#endif
  const auto& zMinBlock = args.value<double>("zMinBlock");  // Starting z-value of the block
  const auto& rMaxFine = args.value<double>("rMaxFine");    // Maximum r-value for fine wafer
  const auto& waferW = args.value<double>("waferW");        // Width of the wafer
  const auto& waferGap = args.value<double>("waferGap");    // Gap between 2 wafers
  const auto& absorbW = args.value<double>("absorberW");    // Width of the absorber
  const auto& absorbH = args.value<double>("absorberH");    // Height of the absorber
  const auto& rMax = args.value<double>("rMax");            // Maximum radial extent
  const auto& rMaxB = args.value<double>("rMaxB");          // Maximum radial extent of a block
  double waferTot = waferW + waferGap;
  std::string idName = DDSplit(args.parentName()).first;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: zStart " << convertCmToMm(zMinBlock) << " rFineCoarse "
                                << convertCmToMm(rMaxFine) << " wafer width " << convertCmToMm(waferW)
                                << " gap among wafers " << convertCmToMm(waferGap) << " absorber width "
                                << convertCmToMm(absorbW) << " absorber height " << convertCmToMm(absorbH) << " rMax "
                                << convertCmToMm(rMax) << ":" << convertCmToMm(rMaxB);
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: NameSpace " << ns.name() << " Parent Name " << idName;
#endif
  std::unordered_set<int> copies;  // List of copy #'s
  copies.clear();

  dd4hep::Volume parent = ns.volume(args.parentName());
  double zi(zMinBlock);
  for (unsigned int i = 0; i < blockThick.size(); i++) {
    double zo = zi + blockThick[i];
    std::string name = idName + "Block" + std::to_string(i);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: Block " << i << ":" << name << " z " << convertCmToMm(zi) << ":"
                                  << convertCmToMm(zo) << " R " << convertCmToMm(rMaxB) << " T "
                                  << convertCmToMm(blockThick[i]);
#endif
    dd4hep::Material matter = ns.material(genMat);
    dd4hep::Solid solid = dd4hep::Tube(0, rMaxB, 0.5 * blockThick[i], 0.0, 2._pi);
    ns.addSolidNS(ns.prepend(name), solid);
    dd4hep::Volume glog = dd4hep::Volume(solid.name(), solid, matter);
    double zz = zi + 0.5 * blockThick[i];
    dd4hep::Position r1(0, 0, zz);
    parent.placeVolume(glog, i, r1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: " << glog.name() << " number " << i << " positioned in "
                                  << args.parentName() << " at " << r1 << " with no rotation";
    edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: \t\tInside Block " << i << " Layers " << layerFrontIn[i] << ":"
                                  << layerBackIn[i] << " zFront " << convertCmToMm(-0.5 * blockThick[i])
                                  << " thickness " << convertCmToMm(blockThick[i]) << " ignore Center 0";
#endif
    DDHGCalGeom::constructLayers(ns,
                                 wafers,
                                 covers,
                                 layerType,
                                 layerSense,
                                 maxModule,
                                 names,
                                 materials,
                                 copyNumber,
                                 layerThick,
                                 absorbW,
                                 absorbH,
                                 waferTot,
                                 rMax,
                                 rMaxFine,
                                 copies,
                                 layerFrontIn[i],
                                 layerBackIn[i],
                                 -0.5 * blockThick[i],
                                 blockThick[i],
                                 false,
                                 glog);
    if (inOut > 1) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: \t\tInside Block " << i << " Layers " << layerFrontOut[i]
                                    << ":" << layerBackOut[i] << " zFront " << convertCmToMm(-0.5 * blockThick[i])
                                    << " thickness " << convertCmToMm(blockThick[i]) << " ignore Center 1";
#endif
      DDHGCalGeom::constructLayers(ns,
                                   wafers,
                                   covers,
                                   layerType,
                                   layerSense,
                                   maxModule,
                                   names,
                                   materials,
                                   copyNumber,
                                   layerThick,
                                   absorbW,
                                   absorbH,
                                   waferTot,
                                   rMax,
                                   rMaxFine,
                                   copies,
                                   layerFrontOut[i],
                                   layerBackOut[i],
                                   -0.5 * blockThick[i],
                                   blockThick[i],
                                   true,
                                   glog);
    }
    zi = zo;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: All blocks are placed in " << convertCmToMm(zMinBlock) << ":"
                                << convertCmToMm(zi) << " with " << copies.size() << " different wafer copy numbers";
#endif

  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalTBModuleX, algorithm)
