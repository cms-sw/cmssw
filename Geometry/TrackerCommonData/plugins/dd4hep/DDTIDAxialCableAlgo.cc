///////////////////////////////////////////////////////////////////////////////
// File: DDTIDAxialCableAlgo.cc
// Description: Create and position TID axial cables at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

namespace {
  long algorithm(dd4hep::Detector &description, cms::DDParsingContext &ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

    LogVerbatim("TIDGeom") << "DDTIDAxialCableAlgo info: Creating an instance...";

    double zBend = args.value<double>("ZBend");                                    // Start z (at bending)........
    double zEnd = args.value<double>("ZEnd");                                      // End   z             ........
    double rMin = args.value<double>("RMin");                                      // Minimum radius      ........
    double rMax = args.value<double>("RMax");                                      // Maximum radius      ........
    double rTop = args.value<double>("RTop");                                      // Maximum radius (top)........
    double width = args.value<double>("Width");                                    // Angular width
    double thick = args.value<double>("Thick");                                    // Thickness
    std::vector<double> angles = args.value<std::vector<double>>("Angles");        // Phi Angles
    std::vector<double> zposWheel = args.value<std::vector<double>>("ZPosWheel");  // Z position of wheels
    std::vector<double> zposRing = args.value<std::vector<double>>("ZPosRing");    // Z position of rings inside wheels

    std::string childName = args.value<std::string>("ChildName");  // Child name
    std::string matIn = args.value<std::string>("MaterialIn");     // Material name (for inner parts)
    std::string matOut = args.value<std::string>("MaterialOut");   // Material name (for outer part)

    LogVerbatim("TIDGeom") << "DDTIDAxialCableAlgo debug: Parameters for creating " << (zposWheel.size() + 2)
                           << " axial cables and positioning" << angles.size() << "copies in Service volume"
                           << "\nzBend " << zBend << "\nzEnd " << zEnd << "\nrMin " << rMin << "\nrMax " << rMax
                           << "\nCable width " << convertRadToDeg(width) << "\nthickness " << thick << " with Angles";

    for (int i = 0; i < (int)(angles.size()); i++)
      LogVerbatim("TIDGeom") << "\n\tangles[" << i << "] = " << convertRadToDeg(angles[i]);

    LogVerbatim("TIDGeom") << "\nWheels " << zposWheel.size() << " at Z";

    for (int i = 0; i < (int)(zposWheel.size()); i++)
      LogVerbatim("TIDGeom") << "\n\tzposWheel[" << i << "] = " << zposWheel[i];

    LogVerbatim("TIDGeom") << "\neach with " << zposRing.size() << " Rings at Z";

    for (int i = 0; i < (int)(zposRing.size()); i++)
      LogVerbatim("TIDGeom") << "\tzposRing[" << i << "] = " << zposRing[i];

    Volume mother = ns.volume(args.parentName());

    LogVerbatim("TIDGeom") << "DDTIDAxialCableAlgo debug: Parent " << mother.name() << "\tChild " << childName
                           << " NameSpace " << ns.name() << "\tMaterial " << matIn << " and " << matOut;

    std::vector<Volume> logs;
    const double thk = thick / zposRing.size();
    double r = rMin;
    double thktot = 0;
    double z;

    // Cables between the wheels
    for (int k = 0; k < (int)(zposWheel.size()); k++) {
      std::vector<double> pconZ, pconRmin, pconRmax;

      for (int i = 0; i < (int)(zposRing.size()); i++) {
        thktot += thk;
        z = zposWheel[k] + zposRing[i] - 0.5 * thk;

        if (i != 0) {
          pconZ.emplace_back(z);
          pconRmin.emplace_back(r);
          pconRmax.emplace_back(rMax);
        }

        r = rMin;
        pconZ.emplace_back(z);
        pconRmin.emplace_back(r);
        pconRmax.emplace_back(rMax);

        z += thk;
        pconZ.emplace_back(z);
        pconRmin.emplace_back(r);
        pconRmax.emplace_back(rMax);

        r = rMax - thktot;
        pconZ.emplace_back(z);
        pconRmin.emplace_back(r);
        pconRmax.emplace_back(rMax);
      }

      if (k >= ((int)(zposWheel.size()) - 1))
        z = zBend;
      else
        z = zposWheel[k + 1] + zposRing[0] - 0.5 * thk;

      pconZ.emplace_back(z);
      pconRmin.emplace_back(r);
      pconRmax.emplace_back(rMax);

      std::string name = childName + std::to_string(k);

      Solid solid = ns.addSolid(name, Polycone(-0.5 * width, width, pconRmin, pconRmax, pconZ));

      LogVerbatim("TIDGeom") << "DDTIDAxialCableAlgo test: " << solid.name() << " Polycone made of " << matIn
                             << " from " << convertRadToDeg(-0.5 * width) << " to " << convertRadToDeg(0.5 * width)
                             << " and with " << pconZ.size() << " sections ";

      for (int i = 0; i < (int)(pconZ.size()); i++)
        LogVerbatim("TIDGeom") << "\t[" << i << "]\tZ = " << pconZ[i] << "\tRmin = " << pconRmin[i]
                               << "\tRmax = " << pconRmax[i];

      logs.emplace_back(Volume(solid.name(), solid, ns.material("materials:Air")));
    }

    // Cable in the vertical part
    std::vector<double> pconZ, pconRmin, pconRmax;
    r = thktot * rMax / rTop;
    z = zBend - thktot;
    LogVerbatim("TIDGeom") << "DDTIDAxialCableAlgo test: Thk " << thk << " Total " << thktot << " rMax " << rMax
                           << " rTop " << rTop << " dR " << r << " z " << z;

    pconZ.emplace_back(z);
    pconRmin.emplace_back(rMax);
    pconRmax.emplace_back(rMax);

    z = zBend - r;
    pconZ.emplace_back(z);
    pconRmin.emplace_back(rMax);
    pconRmax.emplace_back(rTop);

    pconZ.emplace_back(zBend);
    pconRmin.emplace_back(rMax);
    pconRmax.emplace_back(rTop);

    std::string name = childName + std::to_string(zposWheel.size());

    Solid solid = ns.addSolid(name, Polycone(-0.5 * width, width, pconRmin, pconRmax, pconZ));

    LogVerbatim("TIDGeom") << "DDTIDAxialCableAlgo test: " << solid.name() << " Polycone made of " << matIn << " from "
                           << convertRadToDeg(-0.5 * width) << " to " << convertRadToDeg(0.5 * width) << " and with "
                           << pconZ.size() << " sections ";

    for (int i = 0; i < (int)(pconZ.size()); i++)
      LogVerbatim("TIDGeom") << "\t[" << i << "]\tZ = " << pconZ[i] << "\tRmin = " << pconRmin[i]
                             << "\tRmax = " << pconRmax[i];

    logs.emplace_back(Volume(solid.name(), solid, ns.material(matIn)));

    // Cable in the outer part
    name = childName + std::to_string(zposWheel.size() + 1);
    r = rTop - r;
    solid = ns.addSolid(name, Tube(r, rTop, 0.5 * (zEnd - zBend), -0.5 * width, width));

    LogVerbatim("TIDGeom") << "DDTIDAxialCableAlgo test: " << solid.name() << " Tubs made of " << matOut << " from "
                           << convertRadToDeg(-0.5 * width) << " to " << convertRadToDeg(0.5 * width) << " with Rin "
                           << r << " Rout " << rTop << " ZHalf " << 0.5 * (zEnd - zBend);

    logs.emplace_back(Volume(solid.name(), solid, ns.material(matOut)));

    // Position the cables
    constexpr double theta = 90_deg;

    for (int i = 0; i < (int)(angles.size()); i++) {
      double phix = angles[i];
      double phiy = phix + 90_deg;

      Rotation3D rotation = cms::makeRotation3D(theta, phix, theta, phiy, 0., 0.);
      LogVerbatim("TIDGeom") << "DDTIDAxialCableAlgo test: Creating a new "
                             << "rotation: " << convertRadToDeg(theta) << ", " << convertRadToDeg(phix) << ", "
                             << convertRadToDeg(theta) << ", " << convertRadToDeg(phiy) << ", 0, 0";

      for (int k = 0; k < (int)(logs.size()); k++) {
        Position tran = Position(0., 0., 0.);

        if (k == ((int)(logs.size()) - 1))
          tran = Position(0., 0., 0.5 * (zEnd + zBend));

        mother.placeVolume(logs[k], i + 1, Transform3D(rotation, tran));

        LogVerbatim("TIDGeom") << "DDTIDAxialCableAlgo test " << logs[k].name() << " number " << i + 1
                               << " positioned in " << mother.name() << " at " << tran << " with " << rotation;
      }
    }
    return s_executed;
  }
}  // namespace

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTIDAxialCableAlgo, algorithm)
