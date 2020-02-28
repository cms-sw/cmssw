#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e, SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  double diskDz = args.dble("DiskDz");                   // Disk  thickness
  double rMax = args.dble("RMax");                       // Maximum radius
  double cableT = args.dble("CableT");                   // Cable thickness
  vector<double> rodRin = args.vecDble("RodRin");        // Radii for inner rods
  vector<double> rodRout = args.vecDble("RodRout");      // Radii for outer rods
  vector<string> cableM = args.vecStr("CableMaterial");  // Materials for cables
  double connW = args.dble("ConnW");                     // Connector width
  double connT = args.dble("ConnT");                     // Connector thickness
  vector<string> connM = args.vecStr("ConnMaterial");    // Materials for connectors
  vector<double> coolR1 = args.vecDble("CoolR1");        // Radii for cooling manifold
  vector<double> coolR2 = args.vecDble("CoolR2");        // Radii for return cooling manifold
  double coolRin = args.dble("CoolRin");                 // Inner radius of cooling manifold
  double coolRout1 = args.dble("CoolRout1");             // Outer radius of cooling manifold
  double coolRout2 = args.dble("CoolRout2");             // Outer radius of cooling fluid in cooling manifold
  double coolStartPhi1 = args.dble("CoolStartPhi1");     // Starting Phi of cooling manifold
  double coolDeltaPhi1 = args.dble("CoolDeltaPhi1");     // Phi Range of cooling manifold
  double coolStartPhi2 = args.dble("CoolStartPhi2");     // Starting Phi of cooling fluid in of cooling manifold
  double coolDeltaPhi2 = args.dble("CoolDeltaPhi2");     // Phi Range of of cooling fluid in cooling manifold
  string coolM1 = args.str("CoolMaterial1");             // Material for cooling manifold
  string coolM2 = args.str("CoolMaterial2");             // Material for cooling fluid
  vector<string> names = args.vecStr("RingName");        // Names of layers

  string parentName = args.parentName();
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Parent " << parentName << " NameSpace " << ns.name();
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Disk Half width " << diskDz << "\tRMax " << rMax
                      << "\tCable Thickness " << cableT << "\tRadii of disk position and cable materials:";
  for (int i = 0; i < (int)(rodRin.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tRin = " << rodRin[i] << "\tRout = " << rodRout[i] << "  " << cableM[i];
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Connector Width = " << connW << "\tThickness = " << connT
                      << "\tMaterials: ";
  for (int i = 0; i < (int)(connM.size()); i++)
    LogDebug("TOBGeom") << "\tconnM[" << i << "] = " << connM[i];
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Cool Manifold Torus Rin = " << coolRin << " Rout = " << coolRout1
                      << "\t Phi start = " << coolStartPhi1 << " Phi Range = " << coolDeltaPhi1
                      << "\t Material = " << coolM1 << "\t Radial positions:";
  for (int i = 0; i < (int)(coolR1.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tR = " << coolR1[i];
  for (int i = 0; i < (int)(coolR2.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tR = " << coolR2[i];
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Cooling Fluid Torus Rin = " << coolRin << " Rout = " << coolRout2
                      << "\t Phi start = " << coolStartPhi2 << " Phi Range = " << coolDeltaPhi2
                      << "\t Material = " << coolM2 << "\t Radial positions:";
  for (int i = 0; i < (int)(coolR1.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tR = " << coolR1[i];
  for (int i = 0; i < (int)(coolR2.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tR = " << coolR2[i];
  for (int i = 0; i < (int)(names.size()); i++)
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: names[" << i << "] = " << names[i];

  Volume disk = ns.volume(parentName);
  // Loop over sub disks
  for (int i = 0; i < (int)(names.size()); i++) {
    Solid solid;
    string name;
    double dz, rin, rout;

    // Cooling Manifolds
    name = "TOBCoolingManifold" + names[i] + "a";
    dz = coolRout1;
    solid = ns.addSolid(name, Torus(coolR1[i], coolRin, coolRout1, coolStartPhi1, coolDeltaPhi1));
    LogDebug("TOBGeom") << name << " Torus made of " << coolM1 << " from " << convertRadToDeg(coolStartPhi1) << " to "
                        << convertRadToDeg((coolStartPhi1 + coolDeltaPhi1)) << " with Rin " << coolRin << " Rout "
                        << coolRout1 << " R torus " << coolR1[i];
    Volume coolManifoldLogic_a = ns.addVolume(Volume(name, solid, ns.material(coolM1)));
    Position r1(0, 0, (dz - diskDz));
    disk.placeVolume(coolManifoldLogic_a, i + 1, r1);  // i+1
    LogDebug("TOBGeom") << name << " number " << i + 1 << " positioned in " << disk.name() << " at " << r1
                        << " with no rotation";

    // Cooling Fluid (in Cooling Manifold)
    name = "TOBCoolingManifoldFluid" + names[i] + "a";
    solid = ns.addSolid(name, Torus(coolR1[i], coolRin, coolRout2, coolStartPhi2, coolDeltaPhi2));
    LogDebug("TOBGeom") << name << " Torus made of " << coolM2 << " from " << convertRadToDeg(coolStartPhi2) << " to "
                        << convertRadToDeg((coolStartPhi2 + coolDeltaPhi2)) << " with Rin " << coolRin << " Rout "
                        << coolRout2 << " R torus " << coolR1[i];
    Volume coolManifoldFluidLogic_a = ns.addVolume(Volume(name, solid, ns.material(coolM2)));
    coolManifoldLogic_a.placeVolume(coolManifoldFluidLogic_a, i + 1);  // i+1
    LogDebug("TOBGeom") << name << " number " << i + 1 << " positioned in " << coolM2
                        << " with no translation and no rotation";

    name = "TOBCoolingManifold" + names[i] + "r";
    dz = coolRout1;
    solid = ns.addSolid(name, Torus(coolR2[i], coolRin, coolRout1, coolStartPhi1, coolDeltaPhi1));
    LogDebug("TOBGeom") << name << " Torus made of " << coolM1 << " from " << convertRadToDeg(coolStartPhi1) << " to "
                        << convertRadToDeg((coolStartPhi1 + coolDeltaPhi1)) << " with Rin " << coolRin << " Rout "
                        << coolRout1 << " R torus " << coolR2[i];
    Volume coolManifoldLogic_r = ns.addVolume(Volume(name, solid, ns.material(coolM1)));
    r1 = Position(0, 0, (dz - diskDz));
    disk.placeVolume(coolManifoldLogic_r, i + 1, r1);  // i+1
    LogDebug("TOBGeom") << name << " number " << i + 1 << " positioned in " << disk.name() << " at " << r1
                        << " with no rotation";

    // Cooling Fluid (in Cooling Manifold)
    name = "TOBCoolingManifoldFluid" + names[i] + "r";
    solid = ns.addSolid(name, Torus(coolR2[i], coolRin, coolRout2, coolStartPhi2, coolDeltaPhi2));
    LogDebug("TOBGeom") << name << " Torus made of " << coolM2 << " from " << convertRadToDeg(coolStartPhi2) << " to "
                        << convertRadToDeg((coolStartPhi2 + coolDeltaPhi2)) << " with Rin " << coolRin << " Rout "
                        << coolRout2 << " R torus " << coolR2[i];
    Volume coolManifoldFluidLogic_r = ns.addVolume(Volume(name, solid, ns.material(coolM2)));
    coolManifoldLogic_r.placeVolume(coolManifoldFluidLogic_r, i + 1);  // i+1
    LogDebug("TOBGeom") << name << " number " << i + 1 << " positioned in " << coolM2
                        << " with no translation and no rotation";

    // Connectors
    name = "TOBConn" + names[i];
    dz = 0.5 * connT;
    rin = 0.5 * (rodRin[i] + rodRout[i]) - 0.5 * connW;
    rout = 0.5 * (rodRin[i] + rodRout[i]) + 0.5 * connW;
    solid = ns.addSolid(name, Tube(rin, rout, dz));
    LogDebug("TOBGeom") << name << " Tubs made of " << connM[i] << " from 0 to " << convertRadToDeg(2_pi)
                        << " with Rin " << rin << " Rout " << rout << " ZHalf " << dz;
    Volume connLogic = ns.addVolume(Volume(name, solid, ns.material(connM[i])));
    Position r2(0, 0, (dz - diskDz));
    disk.placeVolume(connLogic, i + 1, r2);  // i+1
    LogDebug("TOBGeom") << name << " number " << i + 1 << " positioned in " << disk.name() << " at " << r2
                        << " with no rotation";

    // Now the radial cable
    name = "TOBRadServices" + names[i];
    rin = 0.5 * (rodRin[i] + rodRout[i]);
    rout = (i + 1 == (int)(names.size()) ? rMax : 0.5 * (rodRin[i + 1] + rodRout[i + 1]));
    vector<double> pgonZ;
    pgonZ.emplace_back(-0.5 * cableT);
    pgonZ.emplace_back(cableT * (rin / rMax - 0.5));
    pgonZ.emplace_back(0.5 * cableT);
    vector<double> pgonRmin;
    pgonRmin.emplace_back(rin);
    pgonRmin.emplace_back(rin);
    pgonRmin.emplace_back(rin);
    vector<double> pgonRmax;
    pgonRmax.emplace_back(rout);
    pgonRmax.emplace_back(rout);
    pgonRmax.emplace_back(rout);
    solid = ns.addSolid(name, Polycone(0, 2_pi, pgonRmin, pgonRmax, pgonZ));
    LogDebug("TOBGeom") << name << " Polycone made of " << cableM[i] << " from 0 to " << convertRadToDeg(2_pi)
                        << " and with " << pgonZ.size() << " sections";
    for (int ii = 0; ii < (int)(pgonZ.size()); ii++)
      LogDebug("TOBGeom") << "\t[" << ii << "]\tZ = " << pgonZ[ii] << "\tRmin = " << pgonRmin[ii]
                          << "\tRmax = " << pgonRmax[ii];
    Volume cableLogic = ns.addVolume(Volume(name, solid, ns.material(cableM[i])));
    Position r3(0, 0, (diskDz - (i + 0.5) * cableT));
    disk.placeVolume(cableLogic, i + 1, r3);  // i+1
    LogDebug("TOBGeom") << name << " number " << i + 1 << " positioned in " << disk.name() << " at " << r3
                        << " with no rotation";
  }
  LogDebug("TOBGeom") << "<<== End of DDTOBRadCableAlgo construction ...";
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTOBRadCableAlgo, algorithm)
