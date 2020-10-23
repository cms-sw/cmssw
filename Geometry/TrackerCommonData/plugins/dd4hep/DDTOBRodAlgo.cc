#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  string parentName = args.parentName();
  string central = args.str("CentralName");  // Name of the central piece

  double shift = args.dble("Shift");                    // Shift in z
  vector<string> sideRod = args.vecStr("SideRodName");  // Name of the Side Rod
  vector<double> sideRodX = args.vecDble("SideRodX");   // x-positions
  vector<double> sideRodY = args.vecDble("SideRodY");   // y-positions
  vector<double> sideRodZ = args.vecDble("SideRodZ");   // z-positions
  string endRod1 = args.str("EndRod1Name");             // Name of the End Rod of type 1
  vector<double> endRod1Y = args.vecDble("EndRod1Y");   // y-positions
  vector<double> endRod1Z = args.vecDble("EndRod1Z");   // z-positions
  string endRod2 = args.str("EndRod2Name");             // Name of the End Rod of type 2
  double endRod2Y = args.dble("EndRod2Y");              // y-position
  double endRod2Z = args.dble("EndRod2Z");              // z-position

  string cable = args.str("CableName");  // Name of the Mother cable
  double cableZ = args.dble("CableZ");   // z-position

  string clamp = args.str("ClampName");                  // Name of the clamp
  vector<double> clampX = args.vecDble("ClampX");        // x-positions
  vector<double> clampZ = args.vecDble("ClampZ");        // z-positions
  string sideCool = args.str("SideCoolName");            // Name of the Side Cooling Tube
  vector<double> sideCoolX = args.vecDble("SideCoolX");  // x-positions
  vector<double> sideCoolY =
      args.vecDble("SideCoolY");  // y-positions to avoid overlap with the module (be at the same level of EndCool)
  vector<double> sideCoolZ = args.vecDble("SideCoolZ");  // z-positions
  string endCool = args.str("EndCoolName");              // Name of the End Cooling Tube
  string endCoolRot = args.str("EndCoolRot");            // Rotation matrix name for end cool
  double endCoolY = args.dble("EndCoolY");               // y-position to avoid overlap with the module
  double endCoolZ = args.dble("EndCoolZ");               // z-position

  string optFibre = args.str("OptFibreName");            // Name of the Optical Fibre
  vector<double> optFibreX = args.vecDble("optFibreX");  // x-positions
  vector<double> optFibreZ = args.vecDble("optFibreZ");  // z-positions

  string sideClamp1 = args.str("SideClamp1Name");              // Name of the side clamp of type 1
  vector<double> sideClampX = args.vecDble("SideClampX");      // x-positions
  vector<double> sideClamp1DZ = args.vecDble("SideClamp1DZ");  // Delta(z)-positions
  string sideClamp2 = args.str("SideClamp2Name");              // Name of the side clamp of type 2
  vector<double> sideClamp2DZ = args.vecDble("SideClamp2DZ");  // Delta(z)-positions

  string module = args.str("ModuleName");               // Name of the detector modules
  vector<string> moduleRot = args.vecStr("ModuleRot");  // Rotation matrix name for module
  vector<double> moduleY = args.vecDble("ModuleY");     // y-positions
  vector<double> moduleZ = args.vecDble("ModuleZ");     // z-positions
  vector<string> connect = args.vecStr("ICCName");
  ;                                                // Name of the connectors
  vector<double> connectY = args.vecDble("ICCY");  // y-positions
  vector<double> connectZ = args.vecDble("ICCZ");  // z-positions

  string aohName = args.str("AOHName");                  // AOH name
  vector<double> aohCopies = args.vecDble("AOHCopies");  // AOH copies to be positioned on each ICC
  vector<double> aohX = args.vecDble("AOHx");            // AOH translation with respect small-ICC center (X)
  vector<double> aohY = args.vecDble("AOHy");            // AOH translation with respect small-ICC center (Y)
  vector<double> aohZ = args.vecDble("AOHz");            // AOH translation with respect small-ICC center (Z)

  LogDebug("TOBGeom") << "Parent " << parentName << " Central " << central << " NameSpace " << ns.name() << "\tShift "
                      << shift;
  for (int i = 0; i < (int)(sideRod.size()); i++) {
    LogDebug("TOBGeom") << sideRod[i] << " to be positioned " << sideRodX.size() << " times at y = " << sideRodY[i]
                        << " z = " << sideRodZ[i] << " and x";
    for (double j : sideRodX)
      LogDebug("TOBGeom") << "\tsideRodX[" << i << "] = " << j;
  }
  LogDebug("TOBGeom") << endRod1 << " to be "
                      << "positioned " << endRod1Y.size() << " times at";
  for (int i = 0; i < (int)(endRod1Y.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\ty = " << endRod1Y[i] << "\tz = " << endRod1Z[i];
  LogDebug("TOBGeom") << endRod2 << " to be "
                      << "positioned at y = " << endRod2Y << " z = " << endRod2Z;
  LogDebug("TOBGeom") << cable << " to be "
                      << "positioned at z = " << cableZ;
  LogDebug("TOBGeom") << clamp << " to be "
                      << "positioned " << clampX.size() << " times at";
  for (int i = 0; i < (int)(clampX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << clampX[i] << "\tz = " << clampZ[i];
  LogDebug("TOBGeom") << sideCool << " to be "
                      << "positioned " << sideCoolX.size() << " times at";
  for (int i = 0; i < (int)(sideCoolX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << sideCoolX[i] << "\ty = " << sideCoolY[i]
                        << "\tz = " << sideCoolZ[i];
  LogDebug("TOBGeom") << endCool << " to be "
                      << "positioned with " << endCoolRot << " rotation at"
                      << " y = " << endCoolY << " z = " << endCoolZ;
  LogDebug("TOBGeom") << optFibre << " to be "
                      << "positioned " << optFibreX.size() << " times at";
  for (int i = 0; i < (int)(optFibreX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << optFibreX[i] << "\tz = " << optFibreZ[i];
  LogDebug("TOBGeom") << sideClamp1 << " to be "
                      << "positioned " << sideClampX.size() << " times at";
  for (int i = 0; i < (int)(sideClampX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << sideClampX[i] << "\tdz = " << sideClamp1DZ[i];
  LogDebug("TOBGeom") << sideClamp2 << " to be "
                      << "positioned " << sideClampX.size() << " times at";
  for (int i = 0; i < (int)(sideClampX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << sideClampX[i] << "\tdz = " << sideClamp2DZ[i];
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug:\t" << module << " positioned " << moduleRot.size() << " times";
  for (int i = 0; i < (int)(moduleRot.size()); i++)
    LogDebug("TOBGeom") << "\tRotation " << moduleRot[i] << "\ty = " << moduleY[i] << "\tz = " << moduleZ[i];
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug:\t" << connect.size() << " ICC positioned with no rotation";
  for (int i = 0; i < (int)(connect.size()); i++)
    LogDebug("TOBGeom") << "\t" << connect[i] << "\ty = " << connectY[i] << "\tz = " << connectZ[i];
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug:\t" << aohName << " AOH will be positioned on ICC's";
  for (int i = 0; i < (int)(aohCopies.size()); i++)
    LogDebug("TOBGeom") << " copies " << aohCopies[i] << "\tx = " << aohX[i] << "\ty = " << aohY[i]
                        << "\tz = " << aohZ[i];

  const string& centName = central;
  string child;
  const string& rodName = parentName;
  Volume rod = ns.volume(rodName);

  // Side Rods
  for (int i = 0; i < (int)(sideRod.size()); i++) {
    for (int j = 0; j < (int)(sideRodX.size()); j++) {
      Position r(sideRodX[j], sideRodY[i], sideRodZ[i]);
      child = sideRod[i];
      rod.placeVolume(ns.volume(child), j + 1, r);
      LogDebug("TOBGeom") << child << " number " << j + 1 << " positioned in " << rodName << " at " << r
                          << " with no rotation";
    }
  }
  // Clamps
  for (int i = 0; i < (int)(clampX.size()); i++) {
    Position r(clampX[i], 0, shift + clampZ[i]);
    child = clamp;
    rod.placeVolume(ns.volume(child), i + 1, r);
    LogDebug("TOBGeom") << child << " number " << i + 1 << " positioned in " << rodName << " at " << r
                        << " with no rotation";
  }
  // Side Cooling tubes
  for (int i = 0; i < (int)(sideCoolX.size()); i++) {
    Position r(sideCoolX[i], sideCoolY[i], shift + sideCoolZ[i]);
    child = sideCool;
    rod.placeVolume(ns.volume(child), i + 1, r);
    LogDebug("TOBGeom") << child << " number " << i + 1 << " positioned in " << rodName << " at " << r
                        << " with no rotation";
  }
  // Optical Fibres
  for (int i = 0; i < (int)(optFibreX.size()); i++) {
    Position r(optFibreX[i], 0, shift + optFibreZ[i]);
    child = optFibre;
    rod.placeVolume(ns.volume(child), i + 1, r);
    LogDebug("TOBGeom") << child << " number " << i + 1 << " positioned in " << rodName << " at " << r
                        << " with no rotation";
  }

  // Side Clamps
  for (int i = 0; i < (int)(sideClamp1DZ.size()); i++) {
    int j = i / 2;
    Position r(sideClampX[i], moduleY[j], shift + moduleZ[j] + sideClamp1DZ[i]);
    child = sideClamp1;
    rod.placeVolume(ns.volume(child), i + 1, r);
    LogDebug("TOBGeom") << child << " number " << i + 1 << " positioned in " << rodName << " at " << r
                        << " with no rotation";
  }
  for (int i = 0; i < (int)(sideClamp2DZ.size()); i++) {
    int j = i / 2;
    Position r(sideClampX[i], moduleY[j], shift + moduleZ[j] + sideClamp2DZ[i]);
    child = sideClamp2;
    rod.placeVolume(ns.volume(child), i + 1, r);
    LogDebug("TOBGeom") << child << " number " << i + 1 << " positioned in " << rodName << " at " << r
                        << " with no rotation";
  }

  Volume cent = ns.volume(centName);
  // End Rods
  for (int i = 0; i < (int)(endRod1Y.size()); i++) {
    Position r(0, endRod1Y[i], shift + endRod1Z[i]);
    child = endRod1;
    cent.placeVolume(ns.volume(child), i + 1, r);
    LogDebug("TOBGeom") << child << " number " << i + 1 << " positioned in " << centName << " at " << r
                        << " with no rotation";
  }
  Position r1(0, endRod2Y, shift + endRod2Z);
  child = endRod2;
  cent.placeVolume(ns.volume(child), 1, r1);
  LogDebug("TOBGeom") << child << " number 1 "
                      << "positioned in " << centName << " at " << r1 << " with no rotation";

  // End cooling tubes
  Position r2(0, endCoolY, shift + endCoolZ);
  const Rotation3D& rot2 = ns.rotation(endCoolRot);
  child = endCool;
  cent.placeVolume(ns.volume(child), 1, Transform3D(rot2, r2));
  LogDebug("TOBGeom") << child << " number 1 "
                      << "positioned in " << centName << " at " << r2 << " with " << rot2;

  //Mother cable
  Position r3(0, 0, shift + cableZ);
  child = cable;
  cent.placeVolume(ns.volume(child), 1, r3);
  LogDebug("TOBGeom") << child << " number 1 "
                      << "positioned in " << centName << " at " << r3 << " with no rotation";

  //Modules
  for (int i = 0; i < (int)(moduleRot.size()); i++) {
    Position r(0, moduleY[i], shift + moduleZ[i]);
    const Rotation3D& rot = ns.rotation(moduleRot[i]);
    child = module;
    cent.placeVolume(ns.volume(child), i + 1, Transform3D(rot, r));
    LogDebug("TOBGeom") << child << " number " << i + 1 << " positioned in " << centName << " at " << r << " with "
                        << rot;
  }

  //Connectors (ICC, CCUM, ...)
  for (int i = 0; i < (int)(connect.size()); i++) {
    Position r(0, connectY[i], shift + connectZ[i]);
    child = connect[i];
    cent.placeVolume(ns.volume(child), i + 1, r);
    LogDebug("TOBGeom") << child << " number " << i + 1 << " positioned in " << centName << " at " << r
                        << " with no rotation";
  }

  //AOH (only on ICCs)
  int copyNumber = 0;
  for (int i = 0; i < (int)(aohCopies.size()); i++) {
    if (aohCopies[i] != 0) {
      // first copy with (+aohX,+aohZ) translation
      copyNumber++;
      Position r(aohX[i] + 0, aohY[i] + connectY[i], aohZ[i] + shift + connectZ[i]);
      child = aohName;
      cent.placeVolume(ns.volume(child), copyNumber, r);
      LogDebug("TOBGeom") << child << " number " << copyNumber << " positioned in " << centName << " at " << r
                          << " with no rotation";
      // if two copies add a copy with (-aohX,-aohZ) translation
      if (aohCopies[i] == 2) {
        copyNumber++;
        r = Position(-aohX[i] + 0, aohY[i] + connectY[i], -aohZ[i] + shift + connectZ[i]);
        child = aohName;
        cent.placeVolume(ns.volume(child), copyNumber, r);
        LogDebug("TOBGeom") << child << " number " << copyNumber << " positioned in " << centName << " at " << r
                            << " with no rotation";
      }
      // if four copies add 3 copies with (-aohX,+aohZ) (-aohX,-aohZ) (+aohX,+aohZ) and translations
      if (aohCopies[i] == 4) {
        Position rr;
        for (unsigned int j = 1; j < 4; j++) {
          copyNumber++;
          child = aohName;
          switch (j) {
            case 1:
              rr = Position(-aohX[i] + 0, aohY[i] + connectY[i], +aohZ[i] + shift + connectZ[i]);
              cent.placeVolume(ns.volume(child), copyNumber, rr);  // copyNumber
              break;
            case 2:
              rr = Position(-aohX[i] + 0, aohY[i] + connectY[i], -aohZ[i] + shift + connectZ[i]);
              cent.placeVolume(ns.volume(child), copyNumber, rr);  // copyNumber
              break;
            case 3:
              rr = Position(+aohX[i] + 0, aohY[i] + connectY[i], -aohZ[i] + shift + connectZ[i]);
              cent.placeVolume(ns.volume(child), copyNumber, rr);  // copyNumber
              break;
          }
          LogDebug("TOBGeom") << child << " number " << copyNumber << " positioned in " << centName << " at " << rr
                              << " with no rotation";
        }
      }
    }
  }
  LogDebug("TOBGeom") << "<<== End of DDTOBRodAlgo construction ...";
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTOBRodAlgo, algorithm)
