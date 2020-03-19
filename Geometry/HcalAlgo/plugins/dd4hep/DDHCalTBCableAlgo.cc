#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/DetFactoryHelper.h"

//#define EDM_ML_DEBUG

using namespace geant_units::operators;

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  // Header section
  //     <---- Zout ---->
  //  |  ****************     |
  //  |  *              *     Wstep
  //  W  *              ***** |
  //  |  *                  *
  //  |  ********************
  //     <------ Zin ------->
  //     <------ Zout ------>         Zout = Full sector Z at position
  //  |  ********************         Zin  = Full sector Z at position
  //  |  *                 *
  //  W  *                * Angle = Theta sector
  //  |  *               *  )
  //  |  ****************--------
  //     <------ Zin ------->
  //     <------ Zout ------>         Zin(i)=Zout(i-1)
  //  |  ********************         Zout(i)=Zin(i)+W(i)/tan(Theta(i))
  //  |  *                 *
  //  W  *                *  Theta
  //  |  *               *
  //  |  ****************--------
  //     <--- Zin ------>
  std::string genMat = args.value<std::string>("MaterialName");           //General material
  int nsectors = args.value<int>("NSector");                              //Number of potenital straight edges
  int nsectortot = args.value<int>("NSectorTot");                         //Number of straight edges (actual)
  int nhalf = args.value<int>("NHalf");                                   //Number of half modules
  double rin = args.value<double>("RIn");                                 //(see Figure of hcalbarrel)
  std::vector<double> theta = args.value<std::vector<double> >("Theta");  //  .... (in degrees)
  std::vector<double> rmax = args.value<std::vector<double> >("RMax");    //  ....
  std::vector<double> zoff = args.value<std::vector<double> >("ZOff");    //  ....
  std::string absMat = args.value<std::string>("AbsMatName");             //Absorber material
  double thick = args.value<double>("Thickness");                         //Thickness of absorber
  double width1 = args.value<double>("Width1");                           //Width of absorber type 1
  double length1 = args.value<double>("Length1");                         //Length of absorber type 1
  double width2 = args.value<double>("Width2");                           //Width of absorber type 2
  double length2 = args.value<double>("Length2");                         //Length of absorber type 2
  double gap2 = args.value<double>("Gap2");                               //Gap between abosrbers of type 2
  std::string idName = args.value<std::string>("MotherName");             //Name of the "parent" volume.
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: General material " << genMat << "\tSectors " << nsectors << ", "
                               << nsectortot << "\tHalves " << nhalf << "\tRin " << convertCmToMm(rin);
  for (unsigned int i = 0; i < theta.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t" << i << " Theta " << convertRadToDeg(theta[i]) << " rmax "
                                 << convertCmToMm(rmax[i]) << " zoff " << convertCmToMm(zoff[i]);
  edm::LogVerbatim("HCalGeom") << "\tCable mockup made of " << absMat << "\tThick " << convertCmToMm(thick)
                               << "\tLength and width " << convertCmToMm(length1) << ", " << convertCmToMm(width1)
                               << " and " << convertCmToMm(length2) << ", " << convertCmToMm(width2) << " Gap "
                               << convertCmToMm(gap2);
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Parent " << args.parentName() << " idName " << idName
                               << " NameSpace " << ns.name() << " for solids";
#endif

  double alpha = 1._pi / nsectors;
  double dphi = nsectortot * 2._pi / nsectors;
  double zstep0 = zoff[1] + rmax[1] * tan(theta[1]) + (rin - rmax[1]) * tan(theta[2]);
  double zstep1 = zstep0 + thick / cos(theta[2]);
  double zstep2 = zoff[3];
  double rstep0 = rin + (zstep2 - zstep1) / tan(theta[2]);
  double rstep1 = rin + (zstep1 - zstep0) / tan(theta[2]);

  std::vector<double> pgonZ = {zstep0, zstep1, zstep2, zstep2 + thick / cos(theta[2])};
  std::vector<double> pgonRmin = {rin, rin, rstep0, rmax[2]};
  std::vector<double> pgonRmax = {rin, rstep1, rmax[2], rmax[2]};

  dd4hep::Solid solid = dd4hep::Polyhedra(ns.prepend(idName), nsectortot, -alpha, dphi, pgonZ, pgonRmin, pgonRmax);
  dd4hep::Material matter = ns.material(genMat);
  dd4hep::Volume genlogic(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << solid.name() << " Polyhedra made of " << genMat << " with "
                               << nsectortot << " sectors from " << convertRadToDeg(-alpha) << " to "
                               << convertRadToDeg(-alpha + dphi) << " and with " << pgonZ.size() << " sections";
  for (unsigned int i = 0; i < pgonZ.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[i]) << "\tRmin = " << convertCmToMm(pgonRmin[i])
                                 << "\tRmax = " << convertCmToMm(pgonRmax[i]);
#endif

  dd4hep::Volume parent = ns.volume(args.parentName());
  dd4hep::Rotation3D rot;
  parent.placeVolume(genlogic, 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << genlogic.name() << " number 1 positioned in "
                               << parent.name() << " at (0, 0, 0) with no rotation";
#endif
  if (nhalf != 1) {
    rot = cms::makeRotation3D(90._deg, 180._deg, 90._deg, 90._deg, 180._deg, 0);
    parent.placeVolume(genlogic, 2, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << genlogic.name() << " number 2 positioned in "
                                 << parent.name() << " at (0, 0, 0) with rotation: " << rot;
#endif
  }

  //Construct sector (from -alpha to +alpha)
  std::string name = idName + "Module";
  solid = dd4hep::Polyhedra(ns.prepend(name), 1, -alpha, 2 * alpha, pgonZ, pgonRmin, pgonRmax);
  dd4hep::Volume seclogic(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << solid.name() << " Polyhedra made of " << genMat
                               << " with 1 sector from " << convertRadToDeg(-alpha) << " to " << convertRadToDeg(alpha)
                               << " and with " << pgonZ.size() << " sections";
  for (unsigned int i = 0; i < pgonZ.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[i]) << "\tRmin = " << convertCmToMm(pgonRmin[i])
                                 << "\tRmax = " << convertCmToMm(pgonRmax[i]);
#endif

  for (int ii = 0; ii < nsectortot; ++ii) {
    double phi = ii * 2 * alpha;
    dd4hep::Rotation3D rotation;
    if (phi != 0) {
      rotation = dd4hep::RotationZ(phi);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Creating a new rotation "
                                   << "\t90," << convertRadToDeg(phi) << ",90," << (90 + convertRadToDeg(phi))
                                   << ", 0, 0";
#endif
    }
    genlogic.placeVolume(seclogic, ii + 1, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << seclogic.name() << " number " << ii + 1
                                 << " positioned in " << genlogic.name() << " at (0, 0, 0) with rotation: " << rotation;
#endif
  }

  //Now a trapezoid of air
  double rinl = pgonRmin[0] + thick * sin(theta[2]);
  double routl = pgonRmax[2] - thick * sin(theta[2]);
  double dx1 = rinl * tan(alpha);
  double dx2 = 0.90 * routl * tan(alpha);
  double dy = 0.50 * thick;
  double dz = 0.50 * (routl - rinl);
  name = idName + "Trap";
  solid = dd4hep::Trap(ns.prepend(name), dz, 0, 0, dy, dx1, dx1, 0, dy, dx2, dx2, 0);
  dd4hep::Volume glog(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << solid.name() << " Trap made of " << genMat
                               << " of dimensions " << convertCmToMm(dz) << ", 0, 0, " << convertCmToMm(dy) << ", "
                               << convertCmToMm(dx1) << ", " << convertCmToMm(dx1) << ", 0, " << convertCmToMm(dy)
                               << ", " << convertCmToMm(dx2) << ", " << convertCmToMm(dx2) << ", 0";
#endif

  rot = cms::makeRotation3D(90._deg, 270._deg, (180._deg - theta[2]), 0, (90._deg - theta[2]), 0);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Creating a rotation: \t90, 270, "
                               << (180 - convertRadToDeg(theta[2])) << ", 0, " << (90 - convertRadToDeg(theta[2]))
                               << ", 0";
#endif
  dd4hep::Position r1(0.5 * (rinl + routl), 0, 0.5 * (pgonZ[1] + pgonZ[2]));
  seclogic.placeVolume(glog, 1, dd4hep::Transform3D(rot, r1));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << glog.name() << " number 1 positioned in " << seclogic.name()
                               << " at (" << convertCmToMm(0.5 * (rinl + routl)) << ", 0, "
                               << convertCmToMm(0.5 * (pgonZ[1] + pgonZ[2])) << " with rotation: " << rot;
#endif
  //Now the cable of type 1
  name = idName + "Cable1";
  double phi = atan((dx2 - dx1) / (2 * dz));
  double xmid = 0.5 * (dx1 + dx2) - 1.0;
  solid = dd4hep::Box(ns.prepend(name), 0.5 * width1, 0.5 * thick, 0.5 * length1);
  dd4hep::Material absmatter = ns.material(absMat);
  dd4hep::Volume cablog1(solid.name(), solid, absmatter);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << solid.name() << " Box made of " << absMat << " of dimension "
                               << convertCmToMm(0.5 * width1) << ", " << convertCmToMm(0.5 * thick) << ", "
                               << convertCmToMm(0.5 * length1);
#endif

  dd4hep::Rotation3D rot2 = cms::makeRotation3D((90._deg + phi), 0.0, 90._deg, 90._deg, phi, 0.0);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Creating a rotation \t" << (90 + convertRadToDeg(phi))
                               << ", 0, 90, 90, " << convertRadToDeg(phi) << ", 0";
#endif
  dd4hep::Position r2((xmid - 0.5 * width1 * cos(phi)), 0, 0);
  glog.placeVolume(cablog1, 1, dd4hep::Transform3D(rot2, r2));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << cablog1.name() << " number 1 positioned in " << glog.name()
                               << " at (" << convertCmToMm(xmid - 0.5 * width1 * cos(phi))
                               << ", 0, 0) with rotation: " << rot2;
#endif
  dd4hep::Rotation3D rot3 = cms::makeRotation3D((90._deg - phi), 0, 90._deg, 90._deg, -phi, 0);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Creating a rotation \t" << (90 - convertRadToDeg(phi))
                               << ", 0, 90, 90, " << convertRadToDeg(-phi) << ", 0";
#endif
  dd4hep::Position r3(-(xmid - 0.5 * width1 * cos(phi)), 0, 0);
  glog.placeVolume(cablog1, 2, dd4hep::Transform3D(rot3, r3));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << cablog1.name() << " number 2 positioned in " << glog.name()
                               << " at (" << convertCmToMm(xmid - 0.5 * width1 * cos(phi))
                               << ", 0, 0) with rotation: " << rot3;
#endif
  //Now the cable of type 2
  name = idName + "Cable2";
  solid = dd4hep::Box(ns.prepend(name), 0.5 * width2, 0.5 * thick, 0.5 * length2);
  dd4hep::Volume cablog2(solid.name(), solid, absmatter);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << solid.name() << " Box made of " << absMat << " of dimension "
                               << convertCmToMm(0.5 * width2) << ", " << convertCmToMm(0.5 * thick) << ", "
                               << convertCmToMm(0.5 * length2);
#endif

  glog.placeVolume(cablog2, 1, dd4hep::Position(0.5 * (width2 + gap2), 0, 0));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << cablog2.name() << " number 1 positioned in " << glog.name()
                               << " at (" << convertCmToMm(0.5 * (width2 + gap2)) << ", 0, 0) with no rotation";
#endif
  glog.placeVolume(cablog2, 2, dd4hep::Position(-0.5 * (width2 + gap2), 0, 0));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << cablog2.name() << " number 2 positioned in " << glog.name()
                               << " at " << convertCmToMm(-0.5 * (width2 + gap2)) << ", 0, 0) with no rotation";

  edm::LogVerbatim("HCalGeom") << "<<== End of DDHCalTBCableAlgo construction";
#endif

  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalTBCableAlgo, algorithm);
