#include "DataFormats/Math/interface/CMSUnits.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"

//#define EDM_ML_DEBUG

#ifdef EDM_ML_DEBUG
#include <unordered_set>
#endif
using namespace cms_units::operators;

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  const auto& wafers = args.value<std::vector<std::string> >("WaferName");  // Wafers
  const auto& covers = args.value<std::vector<std::string> >("CoverName");  // Insensitive layers of hexagonal size
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << wafers.size() << " wafers";
  unsigned int i(0);
  for (auto wafer : wafers) {
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafer;
    ++i;
  }
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << covers.size() << " covers";
  i = 0;
  for (auto cover : covers) {
    edm::LogVerbatim("HGCalGeom") << "Cover[" << i << "] " << cover;
    ++i;
  }
#endif
  const auto& materials = args.value<std::vector<std::string> >("MaterialNames");  // Materials
  const auto& names = args.value<std::vector<std::string> >("VolumeNames");        // Names
  const auto& thick = args.value<std::vector<double> >("Thickness");               // Thickness of the material
  std::vector<int> copyNumber;                                                     // Initial copy numbers
  for (unsigned int i = 0; i < materials.size(); ++i) {
    copyNumber.emplace_back(1);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << materials.size() << " types of volumes";
  for (unsigned int i = 0; i < names.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names[i] << " of thickness " << thick[i]
                                  << " filled with " << materials[i] << " first copy number " << copyNumber[i];
#endif
  const auto& layers = args.value<std::vector<int> >("Layers");             // Number of layers in a section
  const auto& layerThick = args.value<std::vector<double> >("LayerThick");  // Thickness of each section
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << layers.size() << " blocks";
  for (unsigned int i = 0; i < layers.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick[i] << " with " << layers[i]
                                  << " layers";
#endif
  const auto& layerType = args.value<std::vector<int> >("LayerType");    // Type of the layer
  const auto& layerSense = args.value<std::vector<int> >("LayerSense");  // Content of a layer (sensitive?)
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << layerType.size() << " layers";
  for (unsigned int i = 0; i < layerType.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType[i] << " sensitive class "
                                  << layerSense[i];
#endif
  const auto& zMinBlock = args.value<double>("zMinBlock");  // Starting z-value of the block
  const auto& rMaxFine = args.value<double>("rMaxFine");    // Maximum r-value for fine wafer
  const auto& waferW = args.value<double>("waferW");        // Width of the wafer
  const auto& waferGap = args.value<double>("waferGap");    // Gap between 2 wafers
  const auto& absorbW = args.value<double>("absorberW");    // Width of the absorber
  const auto& absorbH = args.value<double>("absorberH");    // Height of the absorber
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: zStart " << zMinBlock << " rFineCoarse " << rMaxFine
                                << " wafer width " << waferW << " gap among wafers " << waferGap << " absorber width "
                                << absorbW << " absorber height " << absorbH;
#endif
  const auto& slopeB = args.value<std::vector<double> >("SlopeBottom");   // Slope at the lower R
  const auto& slopeT = args.value<std::vector<double> >("SlopeTop");      // Slopes at the larger R
  const auto& zFront = args.value<std::vector<double> >("ZFront");        // Starting Z values for the slopes
  const auto& rMaxFront = args.value<std::vector<double> >("RMaxFront");  // Corresponding rMax's
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: Bottom slopes " << slopeB[0] << ":" << slopeB[1] << " and "
                                << slopeT.size() << " slopes for top";
  for (unsigned int i = 0; i < slopeT.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFront[i] << " Rmax " << rMaxFront[i] << " Slope "
                                  << slopeT[i];
#endif
  std::string idNameSpace = static_cast<std::string>(ns.name());  // Namespace of this and ALL sub-parts
  const auto& idName = args.parentName();                         // Name of the "parent" volume.
#ifdef EDM_ML_DEBUG
  std::unordered_set<int> copies;  // List of copy #'s
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: NameSpace " << idNameSpace << " Mother " << idName;
#endif

  // Mother module
  dd4hep::Volume module = ns.volume(idName);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalTBModule...";
#endif

  double zi(zMinBlock);
  double ww = (waferW + waferGap);
  double dx = 0.5 * ww;
  double dy = 3.0 * dx * tan(30._deg);
  double rr = 2.0 * dx * tan(30._deg);
  int laymin(0);
  for (unsigned int i = 0; i < layers.size(); i++) {
    double zo = zi + layerThick[i];
    double routF = HGCalGeomTools::radius(zi, zFront, rMaxFront, slopeT);
    int laymax = laymin + layers[i];
    double zz = zi;
    double thickTot(0);
    for (int ly = laymin; ly < laymax; ++ly) {
      int ii = layerType[ly];
      int copy = copyNumber[ii];
      double rinB = (layerSense[ly] == 0) ? (zo * slopeB[0]) : (zo * slopeB[1]);
      zz += (0.5 * thick[ii]);
      thickTot += thick[ii];

      std::string name = "HGCal" + names[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: Layer " << ly << ":" << ii << " Front " << zi << ", " << routF
                                    << " Back " << zo << ", " << rinB << " superlayer thickness " << layerThick[i];
#endif

      dd4hep::Material matter = ns.material(materials[ii]);
      dd4hep::Volume glog;
      if (layerSense[ly] == 0) {
        dd4hep::Solid solid = dd4hep::Box(absorbW, absorbH, 0.5 * thick[ii]);
        ns.addSolidNS(ns.prepend(name), solid);
        glog = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule test: " << solid.name() << " box of dimension " << absorbW
                                      << ":" << absorbH << ":" << 0.5 * thick[ii];
#endif
      } else {
        dd4hep::Solid solid = dd4hep::Tube(rinB, routF, 0.5 * thick[ii], 0.0, 2 * cms_units::piRadians);
        ns.addSolidNS(ns.prepend(name), solid);
        glog = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule: " << solid.name() << " Tubs made of " << materials[ii]
                                      << " of dimensions " << rinB << ", " << routF << ", " << 0.5 * thick[ii]
                                      << ", 0.0, 360.0";
#endif
        int ncol = static_cast<int>(2.0 * routF / ww) + 1;
        int nrow = static_cast<int>(routF / (ww * tan(30._deg))) + 1;
#ifdef EDM_ML_DEBUG
        int incm(0), inrm(0), kount(0), ntot(0), nin(0), nfine(0), ncoarse(0);
        edm::LogVerbatim("HGCalGeom") << glog.name() << " rout " << routF << " Row " << nrow << " Column " << ncol;
#endif
        double xc[6], yc[6];
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
                if (rpos < rinB || rpos > routF)
                  cornerAll = false;
              }
#ifdef EDM_ML_DEBUG
              ++ntot;
#endif
              if (cornerAll) {
                dd4hep::Volume glog1;
                if (layerSense[ly] == 1) {
                  double rpos = std::sqrt(xpos * xpos + ypos * ypos);
                  glog1 = (rpos < rMaxFine) ? ns.volume(wafers[0]) : ns.volume(wafers[1]);
#ifdef EDM_ML_DEBUG
                  ++nin;
                  if (rpos < rMaxFine)
                    ++nfine;
                  else
                    ++ncoarse;
#endif
                } else {
                  glog1 = ns.volume(covers[layerSense[ly] - 2]);
                }
                int copyL = inr * 100 + inc;
                if (nc < 0)
                  copyL += 10000;
                if (nr < 0)
                  copyL += 100000;
#ifdef EDM_ML_DEBUG
                if (inc > incm)
                  incm = inc;
                if (inr > inrm)
                  inrm = inr;
                kount++;
                copies.insert(copy);
#endif
                dd4hep::Position tran(xpos, ypos, 0.0);
                glog.placeVolume(glog1, copyL, tran);
#ifdef EDM_ML_DEBUG
                edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << glog1.name() << " number " << copyL
                                              << " positioned in " << glog.name() << " at " << tran;
#endif
              }
            }
          }
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: # of columns " << incm << " # of rows " << inrm << " and "
                                      << nin << ":" << kount << ":" << ntot << " wafers (" << nfine << ":" << ncoarse
                                      << ") for " << glog.name() << " R " << rinB << ":" << routF;
#endif
      }
      dd4hep::Position r1(0, 0, zz);
      module.placeVolume(glog, copy, r1);
      ++copyNumber[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModule test: " << glog.name() << " number " << copy
                                    << " positioned in " << module.name() << " at " << r1;
#endif
      zz += (0.5 * thick[ii]);
    }  // End of loop over layers in a block
    zi = zo;
    laymin = laymax;
    if (fabs(thickTot - layerThick[i]) < 0.00001) {
    } else if (thickTot > layerThick[i]) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " is smaller than thickness "
                                 << thickTot << " of all its components **** ERROR ****\n";
    } else if (thickTot < layerThick[i]) {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " does not match with "
                                   << thickTot << " of the components\n";
    }
  }  // End of loop over blocks

  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalTBModule, algorithm)
