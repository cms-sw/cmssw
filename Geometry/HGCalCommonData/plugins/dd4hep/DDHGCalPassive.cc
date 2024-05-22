///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalPassive.cc
// Description: Geometry factory class for the passive part of a full silicon
//              module
// Created by Sunanda Banerjee
///////////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <sstream>

#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

struct HGCalPassive {
  HGCalPassive() { throw cms::Exception("HGCalGeom") << "Wrong initialization to HGCalPassive"; }
  HGCalPassive(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: Creating an instance";
#endif
    std::string parentName = args.parentName();
    std::string material = args.value<std::string>("ModuleMaterial");  // Material name for mother volume
    double thick = args.value<double>("Thickness");                    // Thickness of the section
    double zMinBlock = args.value<double>("zMinBlock");                // z-position of the first layer
    double moduleThick = args.value<double>("ModuleThick");            // Thickness of the overall module
    std::vector<std::string> tagLayer =
        args.value<std::vector<std::string>>("TagLayer");  // Tag of the layer (to be added to name)
    std::vector<std::string> tagSector =
        args.value<std::vector<std::string>>("TagSector");  // Tag of the sector (to be added to name)
    int parts = args.value<int>("Parts");                   // number of parts in units of 30 degree
    double phi0 = args.value<double>("PhiStart");           // Start phi of the first cassette
    double dphi = (2._pi) / tagSector.size();               // delta phi of the cassette
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << tagLayer.size() << " Modules with base name " << parentName
                                  << " made of " << material << " T " << thick << " Sectors " << tagSector.size()
                                  << " Parts " << parts << " phi0 " << convertRadToDeg(phi0);
    for (unsigned int i = 0; i < tagLayer.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Layer " << i << " Tag " << tagLayer[i] << " T " << moduleThick;
    for (unsigned int i = 0; i < tagSector.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Sector " << i << " Tag " << tagSector[i] << " W " << convertRadToDeg(dphi);
#endif
    std::vector<std::string> layerNames = args.value<std::vector<std::string>>("LayerNames");  // Names of the layers
    std::vector<std::string> materials =
        args.value<std::vector<std::string>>("LayerMaterials");                          // Materials of the layers
    std::vector<double> layerThick = args.value<std::vector<double>>("LayerThickness");  // Thickness of layers
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << layerNames.size() << " types of volumes";
    for (unsigned int i = 0; i < layerNames.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << layerNames[i] << " of thickness " << layerThick[i]
                                    << " filled with " << materials[i];
#endif

    std::vector<int> layerType = args.value<std::vector<int>>("LayerType");  // Layer types
#ifdef EDM_ML_DEBUG
    std::ostringstream st1;
    for (unsigned int i = 0; i < layerType.size(); ++i)
      st1 << " [" << i << "] " << layerType[i];
    edm::LogVerbatim("HGCalGeom") << "There are " << layerType.size() << " blocks" << st1.str();
#endif

    double shiftTop = args.value<double>("ShiftTop");     // Tolerance at the top
    double shiftBot = args.value<double>("ShiftBottom");  // Tolerance at the bottom
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Shifts st the top " << shiftTop << " and at the bottom " << shiftBot;
#endif

    std::vector<double> slopeB = args.value<std::vector<double>>("SlopeBottom");    // Slope at the lower R
    std::vector<double> zFrontB = args.value<std::vector<double>>("ZFrontBottom");  // Starting Z values for the slopes
    std::vector<double> rMinFront = args.value<std::vector<double>>("RMinFront");   // Corresponding rMin's
    std::vector<double> slopeT = args.value<std::vector<double>>("SlopeTop");       // Slopes at the larger R
    std::vector<double> zFrontT = args.value<std::vector<double>>("ZFrontTop");     // Starting Z values for the slopes
    std::vector<double> rMaxFront = args.value<std::vector<double>>("RMaxFront");   // Corresponding rMax's
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < slopeB.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Bottom Block [" << i << "] Zmin " << zFrontB[i] << " Rmin " << rMinFront[i]
                                    << " Slope " << slopeB[i];
    for (unsigned int i = 0; i < slopeT.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Top Block [" << i << "] Zmin " << zFrontT[i] << " Rmax " << rMaxFront[i]
                                    << " Slope " << slopeT[i];

    edm::LogVerbatim("HGCalGeom") << "==>> Executing DDHGCalPassive...";
#endif

    static constexpr double tol = 0.00001;

    // Loop over Layers
    double zim(zMinBlock);
    //Loop over layers
    for (unsigned int j = 0; j < tagLayer.size(); ++j) {
      double routF = HGCalGeomTools::radius(zim, zFrontT, rMaxFront, slopeT) - shiftTop;
      double zo = zim + moduleThick;
      double rinB = HGCalGeomTools::radius(zo, zFrontB, rMinFront, slopeB) + shiftBot;
      zim += moduleThick;
      for (unsigned int k = 0; k < tagSector.size(); ++k) {
        std::string parentName = parentName + tagLayer[j] + tagSector[k];
        double phi1 = phi0 + k * dphi;
        double phi2 = phi1 + dphi;
        double phi0 = phi1 + 0.5 * dphi;
        // First the mother
        std::vector<double> xM, yM;
        if (parts == 1) {
          xM = {rinB * cos(phi1), routF * cos(phi1), routF * cos(phi2), rinB * cos(phi2)};
          yM = {rinB * sin(phi1), routF * sin(phi1), routF * sin(phi2), rinB * sin(phi2)};
        } else {
          xM = {rinB * cos(phi1),
                routF * cos(phi1),
                routF * cos(phi0),
                routF * cos(phi2),
                rinB * cos(phi2),
                rinB * cos(phi0)};
          yM = {rinB * sin(phi1),
                routF * sin(phi1),
                routF * sin(phi0),
                routF * sin(phi2),
                rinB * sin(phi2),
                rinB * sin(phi0)};
        }
        std::vector<double> zw = {-0.5 * thick, 0.5 * thick};
        std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
        dd4hep::Solid solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
        ns.addSolidNS(ns.prepend(parentName), solid);
        dd4hep::Material matter = ns.material(material);
        dd4hep::Volume glogM = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << solid.name() << " extruded polygon made of "
                                      << matter.name() << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0]
                                      << ":" << scale[0] << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1]
                                      << ":" << scale[1] << " and " << xM.size() << " edges";
        for (unsigned int kk = 0; kk < xM.size(); ++kk)
          edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xM[kk] << ":" << yM[kk];
#endif

        // Then the layers
        std::vector<dd4hep::Volume> glogs(materials.size());
        std::vector<int> copyNumber(materials.size(), 1);
        double zi(-0.5 * thick), thickTot(0.0);
        for (unsigned int l = 0; l < layerType.size(); l++) {
          unsigned int i = layerType[l];
          if (copyNumber[i] == 1) {
            zw[0] = -0.5 * layerThick[i];
            zw[1] = 0.5 * layerThick[i];
            std::string layerName = parentName + layerNames[i];
            solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
            ns.addSolidNS(ns.prepend(layerName), solid);
            dd4hep::Material matter = ns.material(materials[i]);
            glogs[i] = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "DDHGCalPassive: Layer " << i << ":" << l << ":" << solid.name() << " extruded polygon made of "
                << matter.name() << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1] << " and " << xM.size()
                << " edges";
            for (unsigned int kk = 0; kk < xM.size(); ++kk)
              edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xM[kk] << ":" << yM[kk];
#endif
          }
          dd4hep::Position tran0(0, 0, (zi + 0.5 * layerThick[i]));
          glogM.placeVolume(glogs[i], copyNumber[i], tran0);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << glogs[i].name() << " number " << copyNumber[i]
                                        << " positioned in " << glogM.name() << " at (0, 0, "
                                        << cms::convert2mm(zi + 0.5 * layerThick[i]) << ") with no rotation";
#endif
          ++copyNumber[i];
          zi += layerThick[i];
          thickTot += layerThick[i];
        }
        if ((std::abs(thickTot - thick) >= tol) && (!layerType.empty())) {
          if (thickTot > thick) {
            edm::LogError("HGCalGeom") << "Thickness of the partition " << thick << " is smaller than " << thickTot
                                       << ": thickness of all its components **** ERROR ****";
          } else {
            edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick << " does not match with "
                                         << thickTot << " of the components";
          }
        }
      }
    }
  }
};

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  HGCalPassive passiveAlgo(ctxt, e);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalPassive, algorithm)
