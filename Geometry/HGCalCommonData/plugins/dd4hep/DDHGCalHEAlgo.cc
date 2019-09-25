///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalHEAlgo.cc
// Description: Geometry factory class for HGCal (Mix)
// Author : Raman Sehgal
// DD4hep code for, HGCalHEAlgo, developed by Sunanda Banerjee
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG
using namespace cms_units::operators;

struct HGCalHEAlgo {
  HGCalHEAlgo() { throw cms::Exception("HGCalGeom") << "Wrong initialization to HGCalHEAlgo"; }
  HGCalHEAlgo(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Creating an instance";
#endif

    dd4hep::Volume mother = ns.volume(args.parentName());
    waferNames = args.value<std::vector<std::string>>("WaferNames");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << waferNames.size() << " wafers";
    for (unsigned int i = 0; i < waferNames.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << waferNames[i];
#endif
    materials = args.value<std::vector<std::string>>("MaterialNames");
    volumeNames = args.value<std::vector<std::string>>("VolumeNames");
    thickness = args.value<std::vector<double>>("Thickness");
    for (unsigned int i = 0; i < materials.size(); ++i) {
      copyNumber.emplace_back(1);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << materials.size() << " types of volumes";
    for (unsigned int i = 0; i < volumeNames.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << volumeNames[i] << " of thickness " << thickness[i]
                                    << " filled with " << materials[i] << " first copy number " << copyNumber[i];
#endif
    layerNumbers = args.value<std::vector<int>>("Layers");
    layerThick = args.value<std::vector<double>>("LayerThick");
    rMixLayer = args.value<std::vector<double>>("LayerRmix");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "There are " << layerNumbers.size() << " blocks";
    for (unsigned int i = 0; i < layerNumbers.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick[i] << " Rmid " << rMixLayer[i]
                                    << " with " << layerNumbers[i] << " layers";
#endif
    layerType = args.value<std::vector<int>>("LayerType");
    layerSense = args.value<std::vector<int>>("LayerSense");
    firstLayer = args.value<int>("FirstLayer");
    absorbMode = args.value<int>("AbsorberMode");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "First Layer " << firstLayer << " and "
                                  << "Absober mode " << absorbMode;
#endif
    layerCenter = args.value<std::vector<int>>("LayerCenter");
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < layerCenter.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "LayerCenter [" << i << "] " << layerCenter[i];
#endif
    if (firstLayer > 0) {
      for (unsigned int i = 0; i < layerType.size(); ++i) {
        if (layerSense[i] > 0) {
          int ii = layerType[i];
          copyNumber[ii] = firstLayer;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "First copy number for layer type " << i << ":" << ii << " with "
                                        << materials[ii] << " changed to " << copyNumber[ii];
#endif
          break;
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "There are " << layerType.size() << " layers";
    for (unsigned int i = 0; i < layerType.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType[i] << " sensitive class "
                                    << layerSense[i];
#endif
    materialsTop = args.value<std::vector<std::string>>("TopMaterialNames");
    namesTop = args.value<std::vector<std::string>>("TopVolumeNames");
    layerThickTop = args.value<std::vector<double>>("TopLayerThickness");
    layerTypeTop = args.value<std::vector<int>>("TopLayerType");
    for (unsigned int i = 0; i < materialsTop.size(); ++i) {
      copyNumberTop.emplace_back(1);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << materialsTop.size() << " types of volumes in the top part";
    for (unsigned int i = 0; i < materialsTop.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << namesTop[i] << " of thickness " << layerThickTop[i]
                                    << " filled with " << materialsTop[i] << " first copy number " << copyNumberTop[i];
    edm::LogVerbatim("HGCalGeom") << "There are " << layerTypeTop.size() << " layers in the top part";
    for (unsigned int i = 0; i < layerTypeTop.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerTypeTop[i];
#endif
    materialsBot = args.value<std::vector<std::string>>("BottomMaterialNames");
    namesBot = args.value<std::vector<std::string>>("BottomVolumeNames");
    layerTypeBot = args.value<std::vector<int>>("BottomLayerType");
    layerSenseBot = args.value<std::vector<int>>("BottomLayerSense");
    layerThickBot = args.value<std::vector<double>>("BottomLayerThickness");
    for (unsigned int i = 0; i < materialsBot.size(); ++i) {
      copyNumberBot.emplace_back(1);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << materialsBot.size() << " types of volumes in the bottom part";
    for (unsigned int i = 0; i < materialsBot.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << namesBot[i] << " of thickness " << layerThickBot[i]
                                    << " filled with " << materialsBot[i] << " first copy number " << copyNumberBot[i];
    edm::LogVerbatim("HGCalGeom") << "There are " << layerTypeBot.size() << " layers in the bottom part";
    for (unsigned int i = 0; i < layerTypeBot.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerTypeBot[i]
                                    << " sensitive class " << layerSenseBot[i];
#endif
    zMinBlock = args.value<double>("zMinBlock");
    rad100to200 = args.value<std::vector<double>>("rad100to200");
    rad200to300 = args.value<std::vector<double>>("rad200to300");
    zMinRadPar = args.value<double>("zMinForRadPar");
    choiceType = args.value<int>("choiceType");
    nCutRadPar = args.value<int>("nCornerCut");
    fracAreaMin = args.value<double>("fracAreaMin");
    waferSize = args.value<double>("waferSize");
    waferSepar = args.value<double>("SensorSeparation");
    sectors = args.value<int>("Sectors");
    alpha = (1._pi) / sectors;
    cosAlpha = cos(alpha);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: zStart " << zMinBlock << " radius for wafer type separation uses "
                                  << rad100to200.size() << " parameters; zmin " << zMinRadPar << " cutoff "
                                  << choiceType << ":" << nCutRadPar << ":" << fracAreaMin << " wafer width "
                                  << waferSize << " separations " << waferSepar << " sectors " << sectors << ":"
                                  << convertRadToDeg(alpha) << ":" << cosAlpha;
    for (unsigned int k = 0; k < rad100to200.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] 100-200 " << rad100to200[k] << " 200-300 " << rad200to300[k];
#endif
    slopeB = args.value<std::vector<double>>("SlopeBottom");
    zFrontB = args.value<std::vector<double>>("ZFrontBottom");
    rMinFront = args.value<std::vector<double>>("RMinFront");
    slopeT = args.value<std::vector<double>>("SlopeTop");
    zFrontT = args.value<std::vector<double>>("ZFrontTop");
    rMaxFront = args.value<std::vector<double>>("RMaxFront");
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < slopeB.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFrontB[i] << " Rmin " << rMinFront[i]
                                    << " Slope " << slopeB[i];
    for (unsigned int i = 0; i < slopeT.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFrontT[i] << " Rmax " << rMaxFront[i]
                                    << " Slope " << slopeT[i];
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: NameSpace " << ns.name();
#endif

    waferType = std::make_unique<HGCalWaferType>(
        rad100to200, rad200to300, (waferSize + waferSepar), zMinRadPar, choiceType, nCutRadPar, fracAreaMin);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalHEAlgo...";
    copies.clear();
#endif

    double zi(zMinBlock);
    int laymin(0);
    const double tol(0.01);
    for (unsigned int i = 0; i < layerNumbers.size(); i++) {
      double zo = zi + layerThick[i];
      double routF = HGCalGeomTools::radius(zi, zFrontT, rMaxFront, slopeT);
      int laymax = laymin + layerNumbers[i];
      double zz = zi;
      double thickTot(0);
      for (int ly = laymin; ly < laymax; ++ly) {
        int ii = layerType[ly];
        int copy = copyNumber[ii];
        double hthick = 0.5 * thickness[ii];
        double rinB = HGCalGeomTools::radius(zo, zFrontB, rMinFront, slopeB);
        zz += hthick;
        thickTot += thickness[ii];

        std::string name = volumeNames[ii] + std::to_string(copy);

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Layer " << ly << ":" << ii << " Front " << zi << ", " << routF
                                      << " Back " << zo << ", " << rinB << " superlayer thickness " << layerThick[i];
#endif

        dd4hep::Material matter = ns.material(materials[ii]);
        dd4hep::Volume glog;

        if (layerSense[ly] < 1) {
          std::vector<double> pgonZ, pgonRin, pgonRout;
          if (layerSense[ly] == 0 || absorbMode == 0) {
            double rmax =
                (std::min(routF, HGCalGeomTools::radius(zz + hthick, zFrontT, rMaxFront, slopeT)) * cosAlpha) - tol;
            pgonZ.emplace_back(-hthick);
            pgonZ.emplace_back(hthick);
            pgonRin.emplace_back(rinB);
            pgonRin.emplace_back(rinB);
            pgonRout.emplace_back(rmax);
            pgonRout.emplace_back(rmax);
          } else {
            HGCalGeomTools::radius(zz - hthick,
                                   zz + hthick,
                                   zFrontB,
                                   rMinFront,
                                   slopeB,
                                   zFrontT,
                                   rMaxFront,
                                   slopeT,
                                   -layerSense[ly],
                                   pgonZ,
                                   pgonRin,
                                   pgonRout);
            for (unsigned int isec = 0; isec < pgonZ.size(); ++isec) {
              pgonZ[isec] -= zz;
              pgonRout[isec] = pgonRout[isec] * cosAlpha - tol;
            }
          }

          dd4hep::Solid solid = dd4hep::Polyhedra(sectors, -alpha, 2. * cms_units::piRadians, pgonZ, pgonRin, pgonRout);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " polyhedra of " << sectors
                                        << " sectors covering " << convertRadToDeg(-alpha) << ":"
                                        << convertRadToDeg(-alpha + 2._pi) << " with " << pgonZ.size() << " sections";
          for (unsigned int k = 0; k < pgonZ.size(); ++k)
            edm::LogVerbatim("HGCalGeom")
                << "[" << k << "] z " << pgonZ[k] << " R " << pgonRin[k] << ":" << pgonRout[k];
#endif
        } else {
          dd4hep::Solid solid = dd4hep::Tube(rinB, routF, hthick, 0.0, 2. * cms_units::piRadians);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);

#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                        << " of dimensions " << rinB << ", " << routF << ", " << hthick
                                        << ", 0.0, 360.0 and positioned in: " << glog.name() << " number " << copy;
#endif
          positionMix(ctxt, e, glog, name, copy, thickness[ii], matter, rinB, rMixLayer[i], routF, zz);
        }

        dd4hep::Position r1(0, 0, zz);
        mother.placeVolume(glog, copy, r1);
        ++copyNumber[ii];
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog.name() << " number " << copy << " positioned in "
                                      << mother.name() << " at " << r1 << " with no rotation";
#endif
        zz += hthick;
      }  // End of loop over layers in a block
      zi = zo;
      laymin = laymax;
      if (std::abs(thickTot - layerThick[i]) < 0.00001) {
      } else if (thickTot > layerThick[i]) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " is smaller than " << thickTot
                                   << ": thickness of all its "
                                   << "components **** ERROR ****";
      } else if (thickTot < layerThick[i]) {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " does not match with "
                                     << thickTot << " of the components";
      }
    }  // End of loop over blocks

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << copies.size() << " different wafer copy numbers";
    int k(0);
    for (std::unordered_set<int>::const_iterator itr = copies.begin(); itr != copies.end(); ++itr, ++k) {
      edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
    }
    copies.clear();
    edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalHEAlgo construction...";
#endif
  }

  void positionMix(cms::DDParsingContext& ctxt,
                   xml_h e,
                   const dd4hep::Volume& glog,
                   const std::string& nameM,
                   int copyM,
                   double thick,
                   const dd4hep::Material& matter,
                   double rin,
                   double rmid,
                   double rout,
                   double zz) {
    cms::DDNamespace ns(ctxt, e, true);

    dd4hep::Volume glog1;
    for (unsigned int ly = 0; ly < layerTypeTop.size(); ++ly) {
      int ii = layerTypeTop[ly];
      copyNumberTop[ii] = copyM;
    }
    for (unsigned int ly = 0; ly < layerTypeBot.size(); ++ly) {
      int ii = layerTypeBot[ly];
      copyNumberBot[ii] = copyM;
    }
    double hthick = 0.5 * thick;
    // Make the top part first
    std::string name = nameM + "Top";

    dd4hep::Solid solid = dd4hep::Tube(rmid, rout, hthick, 0.0, 2. * cms_units::piRadians);
    ns.addSolidNS(ns.prepend(name), solid);
    glog1 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                  << " of dimensions " << rmid << ", " << rout << ", " << hthick << ", 0.0, 360.0";
#endif
    glog.placeVolume(glog1, 1);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog1.name() << " number 1 positioned in " << glog.name()
                                  << " at (0, 0, 0) with no rotation";
#endif
    double thickTot(0), zpos(-hthick);
    for (unsigned int ly = 0; ly < layerTypeTop.size(); ++ly) {
      int ii = layerTypeTop[ly];
      int copy = copyNumberTop[ii];
      double hthickl = 0.5 * layerThickTop[ii];
      thickTot += layerThickTop[ii];
      name = namesTop[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Layer " << ly << ":" << ii << " R " << rmid << ":" << rout
                                    << " Thick " << layerThickTop[ii];
#endif

      dd4hep::Material matter1 = ns.material(materialsTop[ii]);
      solid = dd4hep::Tube(rmid, rout, hthickl, 0.0, 2. * cms_units::piRadians);
      ns.addSolidNS(ns.prepend(name), solid);
      dd4hep::Volume glog2 = dd4hep::Volume(solid.name(), solid, matter1);
      ns.addVolumeNS(glog2);

#ifdef EDM_ML_DEBUG
      double eta1 = -log(tan(0.5 * atan(rmid / zz)));
      double eta2 = -log(tan(0.5 * atan(rout / zz)));
      edm::LogVerbatim("HGCalGeom") << name << " z|rin|rout " << zz << ":" << rmid << ":" << rout << " eta " << eta1
                                    << ":" << eta2;
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter1.name()
                                    << " of dimensions " << rmid << ", " << rout << ", " << hthickl << ", 0.0, 360.0";
#endif
      zpos += hthickl;

      dd4hep::Position r1(0, 0, zpos);
      glog1.placeVolume(glog2, copy, r1);

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Position " << glog2.name() << " number " << copy << " in "
                                    << glog1.name() << " at " << r1 << " with no rotation";
#endif
      ++copyNumberTop[ii];
      zpos += hthickl;
    }
    if (std::abs(thickTot - thick) < 0.00001) {
    } else if (thickTot > thick) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << thick << " is smaller than " << thickTot
                                 << ": thickness of all its components in "
                                 << "the top part **** ERROR ****";
    } else if (thickTot < thick) {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick << " does not match with " << thickTot
                                   << " of the components in top part";
    }

    // Make the bottom part next
    name = nameM + "Bottom";

    solid = dd4hep::Tube(rin, rmid, hthick, 0.0, 2. * cms_units::piRadians);
    ns.addSolidNS(ns.prepend(name), solid);
    glog1 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog1);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                  << " of dimensions " << rin << ", " << rmid << ", " << hthick << ", 0.0, 360.0";
#endif

    glog.placeVolume(glog1, 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog1.name() << " number 1 positioned in " << glog.name()
                                  << " at (0, 0, 0) with no rotation";
#endif
    thickTot = 0;
    zpos = -hthick;
    for (unsigned int ly = 0; ly < layerTypeBot.size(); ++ly) {
      int ii = layerTypeBot[ly];
      int copy = copyNumberBot[ii];
      double hthickl = 0.5 * layerThickBot[ii];
      thickTot += layerThickBot[ii];
      name = namesBot[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Layer " << ly << ":" << ii << " R " << rin << ":" << rmid
                                    << " Thick " << layerThickBot[ii];
#endif

      dd4hep::Material matter1 = ns.material(materialsBot[ii]);
      solid = dd4hep::Tube(rin, rmid, hthickl, 0.0, 2. * cms_units::piRadians);
      ns.addSolidNS(ns.prepend(name), solid);
      dd4hep::Volume glog2 = dd4hep::Volume(solid.name(), solid, matter1);
      ns.addVolumeNS(glog2);

#ifdef EDM_ML_DEBUG
      double eta1 = -log(tan(0.5 * atan(rin / zz)));
      double eta2 = -log(tan(0.5 * atan(rmid / zz)));
      edm::LogVerbatim("HGCalGeom") << name << " z|rin|rout " << zz << ":" << rin << ":" << rmid << " eta " << eta1
                                    << ":" << eta2;
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter1.name()
                                    << " of dimensions " << rin << ", " << rmid << ", " << hthickl << ", 0.0, 360.0";
#endif
      zpos += hthickl;

      dd4hep::Position r1(0, 0, zpos);
      glog1.placeVolume(glog2, copy, r1);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Position " << glog2.name() << " number " << copy << " in "
                                    << glog1.name() << " at " << r1 << " with no rotation";
#endif
      if (layerSenseBot[ly] != 0) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: z " << (zz + zpos) << " Center " << copy << ":"
                                      << (copy - firstLayer) << ":" << layerCenter[copy - firstLayer];
#endif
        positionSensitive(ctxt, e, glog2, rin, rmid, zz + zpos, layerSenseBot[ly], layerCenter[copy - firstLayer]);
      }
      zpos += hthickl;
      ++copyNumberBot[ii];
    }
    if (std::abs(thickTot - thick) < 0.00001) {
    } else if (thickTot > thick) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << thick << " is smaller than " << thickTot
                                 << ": thickness of all its components in "
                                 << "the top part **** ERROR ****";
    } else if (thickTot < thick) {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick << " does not match with " << thickTot
                                   << " of the components in top part";
    }
  }

  void positionSensitive(cms::DDParsingContext& ctxt,
                         xml_h e,
                         const dd4hep::Volume& glog,
                         double rin,
                         double rout,
                         double zpos,
                         int layertype,
                         int layercenter) {
    cms::DDNamespace ns(ctxt, e, true);
    static const double sqrt3 = std::sqrt(3.0);
    double r = 0.5 * (waferSize + waferSepar);
    double R = 2.0 * r / sqrt3;
    double dy = 0.75 * R;
    int N = (int)(0.5 * rout / r) + 2;
    std::pair<double, double> xyoff = geomTools.shiftXY(layercenter, (waferSize + waferSepar));
#ifdef EDM_ML_DEBUG
    int ium(0), ivm(0), iumAll(0), ivmAll(0), kount(0), ntot(0), nin(0);
    std::vector<int> ntype(6, 0);
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog.name() << " rout " << rout << " N " << N
                                  << " for maximum u, v Offset; Shift " << xyoff.first << ":" << xyoff.second
                                  << " WaferSize " << (waferSize + waferSepar);
#endif
    for (int u = -N; u <= N; ++u) {
      int iu = std::abs(u);
      for (int v = -N; v <= N; ++v) {
        int iv = std::abs(v);
        int nr = 2 * v;
        int nc = -2 * u + v;
        double xpos = xyoff.first + nc * r;
        double ypos = xyoff.second + nr * dy;
        std::pair<int, int> corner = HGCalGeomTools::waferCorner(xpos, ypos, r, R, rin, rout, false);
#ifdef EDM_ML_DEBUG
        ++ntot;
#endif
        if (corner.first > 0) {
          int type = waferType->getType(xpos, ypos, zpos);
          int copy = type * 1000000 + iv * 100 + iu;
          if (u < 0)
            copy += 10000;
          if (v < 0)
            copy += 100000;
#ifdef EDM_ML_DEBUG
          if (iu > ium)
            ium = iu;
          if (iv > ivm)
            ivm = iv;
          kount++;
          if (copies.count(copy) == 0)
            copies.insert(copy);
#endif
          if (corner.first == (int)(HGCalParameters::k_CornerSize)) {
#ifdef EDM_ML_DEBUG
            if (iu > iumAll)
              iumAll = iu;
            if (iv > ivmAll)
              ivmAll = iv;
            ++nin;
#endif

            dd4hep::Position tran(xpos, ypos, 0.0);
            if (layertype > 1)
              type += 3;
            glog.placeVolume(ns.volume(waferNames[type]), copy, tran);

#ifdef EDM_ML_DEBUG
            ++ntype[type];
            edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog.name() << " number " << copy << " positioned in "
                                          << glog.name() << " at " << tran << " with no rotation";
#endif
          }
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Maximum # of u " << ium << ":" << iumAll << " # of v " << ivm
                                  << ":" << ivmAll << " and " << nin << ":" << kount << ":" << ntot << " wafers ("
                                  << ntype[0] << ":" << ntype[1] << ":" << ntype[2] << ":" << ntype[3] << ":"
                                  << ntype[4] << ":" << ntype[5] << ") for " << glog.name() << " R " << rin << ":"
                                  << rout;
#endif
  }

  //Required data members to cache the values from XML file
  HGCalGeomTools geomTools;
  std::unique_ptr<HGCalWaferType> waferType;

  std::vector<std::string> waferNames;    // Wafer names
  std::vector<std::string> materials;     // Materials
  std::vector<std::string> volumeNames;   // Names
  std::vector<double> thickness;          // Thickness of the material
  std::vector<int> copyNumber;            // Initial copy numbers
  std::vector<int> layerNumbers;          // Number of layers in a section
  std::vector<double> layerThick;         // Thickness of each section
  std::vector<double> rMixLayer;          // Partition between Si/Sci part
  std::vector<int> layerType;             // Type of the layer
  std::vector<int> layerSense;            // Content of a layer (sensitive?)
  int firstLayer;                         // Copy # of the first sensitive layer
  int absorbMode;                         // Absorber mode
  std::vector<std::string> materialsTop;  // Materials of top layers
  std::vector<std::string> namesTop;      // Names of top layers
  std::vector<double> layerThickTop;      // Thickness of the top sections
  std::vector<int> layerTypeTop;          // Type of the Top layer
  std::vector<int> copyNumberTop;         // Initial copy numbers (top section)
  std::vector<std::string> materialsBot;  // Materials of bottom layers
  std::vector<std::string> namesBot;      // Names of bottom layers
  std::vector<double> layerThickBot;      // Thickness of the bottom sections
  std::vector<int> layerTypeBot;          // Type of the bottom layers
  std::vector<int> copyNumberBot;         // Initial copy numbers (bot section)
  std::vector<int> layerSenseBot;         // Content of bottom layer (sensitive?)
  std::vector<int> layerCenter;           // Centering of the wafers

  double zMinBlock;                 // Starting z-value of the block
  std::vector<double> rad100to200;  // Parameters for 120-200mum trans.
  std::vector<double> rad200to300;  // Parameters for 200-300mum trans.
  double zMinRadPar;                // Minimum z for radius parametriz.
  int choiceType;                   // Type of parametrization to be used
  int nCutRadPar;                   // Cut off threshold for corners
  double fracAreaMin;               // Minimum fractional conatined area
  double waferSize;                 // Width of the wafer
  double waferSepar;                // Sensor separation
  int sectors;                      // Sectors
  std::vector<double> slopeB;       // Slope at the lower R
  std::vector<double> zFrontB;      // Starting Z values for the slopes
  std::vector<double> rMinFront;    // Corresponding rMin's
  std::vector<double> slopeT;       // Slopes at the larger R
  std::vector<double> zFrontT;      // Starting Z values for the slopes
  std::vector<double> rMaxFront;    // Corresponding rMax's
  std::unordered_set<int> copies;   // List of copy #'s
  double alpha, cosAlpha;
};

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  HGCalHEAlgo healgo(ctxt, e);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalHEAlgo, algorithm)
