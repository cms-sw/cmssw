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
    std::string moduleMaterial = args.value<std::string>("ModuleMaterial");  // Material name for mother volume
    double moduleThick = args.value<double>("ModuleThick");                  // Thickness of the overall module
    int sectors = args.value<int>("Sectors");                                // number of phi sectors (cassettes)
    std::vector<std::string> tagsector;                                      // Tag of the sector (to be added to name)
    tagsector.reserve(sectors);
    for (int k = 0; k < sectors; ++k)
      tagsector.emplace_back("F" + std::to_string(k));
    int position = args.value<int>("Position");  // 0 if -z; 1 if +z
    std::vector<std::string> tagpos;             // Tags for the modules
    std::vector<int> xsignpos;                   // sign of the x-value;
    if (position == 0) {
      tagpos.emplace_back("PN");
      xsignpos.emplace_back(-1);
    } else {
      tagpos.emplace_back("PP");
      xsignpos.emplace_back(1);
    }
    double phi0 = args.value<double>("PhiStart");  // Start phi of the first cassette
    double dphi = (2._pi) / tagsector.size();      // delta phi of the cassette
#ifdef EDM_ML_DEBUG
    std::ostringstream st0, st1;
    for (unsigned int k = 0; k < tagsector.size(); ++k)
      st0 << ": " << tagsector[k];
    for (unsigned int k = 0; k < tagpos.size(); ++k)
      st1 << " " << tagpos[k] << ":" << xsignpos[k];
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << tagpos.size() << " Modules with base name " << parentName
                                  << " made of " << moduleMaterial << " T " << moduleThick << " having " << sectors
                                  << " sectors" << st0.str() << " phi0 " << convertRadToDeg(phi0) << " dphi "
                                  << convertRadToDeg(dphi) << " Tags:" << st1.str();
#endif
    std::vector<std::string> layerNames = args.value<std::vector<std::string>>("LayerNames");  // Names of the layers
    std::vector<std::string> layerMaterials =
        args.value<std::vector<std::string>>("LayerMaterials");                          // Materials of the layers
    std::vector<double> layerThick = args.value<std::vector<double>>("LayerThickness");  // Thickness of layers
    std::vector<int> layerType = args.value<std::vector<int>>("LayerType");              // Layer types
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << layerNames.size() << " types of volumes";
    for (unsigned int i = 0; i < layerNames.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << layerNames[i] << " of thickness " << layerThick[i]
                                    << " filled with " << layerMaterials[i];
    std::ostringstream st2;
    for (unsigned int i = 0; i < layerType.size(); ++i)
      st2 << " [" << i << "] " << layerType[i];
    edm::LogVerbatim("HGCalGeom") << "There are " << layerType.size() << " blocks" << st2.str();
#endif

    std::vector<std::string> absNames =
        args.value<std::vector<std::string>>("AbsorberName");                 // Names of the absorber layers
    std::vector<int> absN = args.value<std::vector<int>>("AbsorberN");        // Number of point in each layer
    std::vector<double> absX = args.value<std::vector<double>>("AbsorberX");  // x coordinates of abs layers
    std::vector<double> absY = args.value<std::vector<double>>("AbsorberY");  // y coordinates of abs layers
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "There are " << absNames.size() << " basic absorber shapes:";
    unsigned int j(0);
    for (unsigned int k = 0; k < absNames.size(); ++k) {
      std::ostringstream st3;
      st3 << absNames[k] << " with " << absN[k] << " points:";
      for (int i = 0; i < absN[k]; ++i)
        st3 << " (" << absX[j + i] << ", " << absY[j + i] << ")";
      j += absN[k];
      edm::LogVerbatim("HGCalGeom") << st3.str();
    }
#endif

    static constexpr double tol = 0.00001;

    // Loop over positions
    for (unsigned int i1 = 0; i1 < tagpos.size(); ++i1) {
      // Loop over sectors
      for (int i2 = 0; i2 < sectors; ++i2) {
        double phi = phi0 + i2 * dphi;
        double cphi = std::cos(phi);
        double sphi = std::sin(phi);
        // Loop over passive volumes
        int j(0);
        for (unsigned i3 = 0; i3 < absNames.size(); ++i3) {
          //First make the mother
          std::string parentname = parentName + absNames[i3] + tagsector[i2] + tagpos[i1];
          std::vector<double> zw = {-0.5 * moduleThick, 0.5 * moduleThick};
          std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
          std::vector<double> xM(absN[i3], 0), yM(absN[i3], 0);
          for (int k = 0; k < absN[i3]; ++k) {
            xM[k] = xsignpos[i1] * (cphi * absX[j + k] + sphi * absY[j + k]);
            yM[k] = -sphi * absX[j + k] + cphi * absY[j + k];
          }
          j += absN[i3];
          dd4hep::Solid solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
          ns.addSolidNS(ns.prepend(parentname), solid);
          dd4hep::Material matter = ns.material(moduleMaterial);
          dd4hep::Volume glogM = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glogM);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << solid.name() << " extruded polygon made of "
                                        << matter.name() << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0]
                                        << ":" << scale[0] << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1]
                                        << ":" << scale[1] << " and " << xM.size() << " edges";
          for (unsigned int kk = 0; kk < xM.size(); ++kk)
            edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xM[kk] << ":" << yM[kk];
#endif
          // Then the layers
          std::vector<dd4hep::Volume> glogs(layerMaterials.size());
          std::vector<int> copyNumber(layerMaterials.size(), 1);
          double zi(-0.5 * moduleThick), thickTot(0.0);
          for (unsigned int l = 0; l < layerType.size(); l++) {
            unsigned int i = layerType[l];
            if (copyNumber[i] == 1) {
              zw[0] = -0.5 * layerThick[i];
              zw[1] = 0.5 * layerThick[i];
              std::string layerName = parentname + layerNames[i];
              solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
              ns.addSolidNS(ns.prepend(layerName), solid);
              matter = ns.material(layerMaterials[i]);
              glogs[i] = dd4hep::Volume(solid.name(), solid, matter);
              ns.addVolumeNS(glogs[i]);
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
            edm::LogVerbatim("HGCalGeom")
                << "DDHGCalPassive: " << glogs[i].name() << " number " << copyNumber[i] << " positioned in "
                << glogM.name() << " at (0,0," << cms::convert2mm(zi + 0.5 * layerThick[i]) << ") with no rotation";
#endif
            ++copyNumber[i];
            zi += layerThick[i];
            thickTot += layerThick[i];
          }
          if ((std::abs(thickTot - moduleThick) >= tol) && (!layerType.empty())) {
            if (thickTot > moduleThick) {
              edm::LogError("HGCalGeom") << "Thickness of the partition " << moduleThick << " is smaller than "
                                         << thickTot << ": thickness of all its components **** ERROR ****";
            } else {
              edm::LogWarning("HGCalGeom") << "Thickness of the partition " << moduleThick << " does not match with "
                                           << thickTot << " of the components";
            }
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
