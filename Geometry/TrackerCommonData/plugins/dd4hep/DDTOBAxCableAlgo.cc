#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  vector<string> sectorNumber = args.vecStr("SectorNumber");          // Id. Number of the sectors
  double sectorRin = args.dble("SectorRin");                          // Inner radius of service sectors
  double sectorRout = args.dble("SectorRout");                        // Outer radius of service sectors
  double sectorDz = args.dble("SectorDz");                            // Sector half-length
  double sectorDeltaPhi_B = args.dble("SectorDeltaPhi_B");            // Sector B phi width [A=C=0.5*(360/sectors)]
  vector<double> sectorStartPhi = args.vecDble("SectorStartPhi");     // Starting phi for the service sectors
  vector<string> sectorMaterial_A = args.vecStr("SectorMaterial_A");  // Material for the A sectors
  vector<string> sectorMaterial_B = args.vecStr("SectorMaterial_B");  // Material for the B sectors
  vector<string> sectorMaterial_C = args.vecStr("SectorMaterial_C");  // Material for the C sectors

  for (int i = 0; i < (int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: sectorNumber[" << i << "] = " << sectorNumber[i];

  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: Axial Service Sectors half-length " << sectorDz << "\tRin "
                      << sectorRin << "\tRout = " << sectorRout << "\tPhi of sectors position:";
  for (int i = 0; i < (int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tPhi = " << sectorStartPhi[i];
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: List of materials for the sectors/3 parts";
  //
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: Sector/3 A";
  for (int i = 0; i < (int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tsectorMaterial_A = " << sectorMaterial_A[i];
  //
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: Sector/3 B";
  for (int i = 0; i < (int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tsectorMaterial_B = " << sectorMaterial_B[i];
  //
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: Sector/3 C";
  for (int i = 0; i < (int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tsectorMaterial_C = " << sectorMaterial_C[i];

  string tubsName = args.parentName();
  Volume tubsVol = ns.volume(tubsName);
  // Loop over sectors (sectorNumber vector)
  for (int i = 0; i < (int)(sectorNumber.size()); i++) {
    Solid solid;
    string name;
    double dz, rin, rout, startphi, widthphi, deltaphi;

    // Axial Services
    // Each sector is divided in 3 parts from phi[i] to phi[i+1]

    widthphi = ((i + 1 == (int)(sectorStartPhi.size())) ? (sectorStartPhi[0] + 2_pi) - sectorStartPhi[i]
                                                        : (sectorStartPhi[i + 1] - sectorStartPhi[i]));
    // First Part: A
    name = "TOBAxService_" + sectorNumber[i] + "A";
    dz = sectorDz;
    rin = sectorRin;
    rout = sectorRout;
    startphi = sectorStartPhi[i];
    deltaphi = 0.5 * (widthphi - sectorDeltaPhi_B);
    solid = ns.addSolid(name, Tube(rin, rout, dz, startphi, startphi + deltaphi));
    LogDebug("TOBGeom") << solid.name() << " Tubs made of " << sectorMaterial_A[i] << " from "
                        << convertRadToDeg(startphi) << " to " << convertRadToDeg((startphi + deltaphi)) << " with Rin "
                        << rin << " Rout " << rout << " ZHalf " << dz;
    Volume sectorLogic = ns.addVolume(Volume(name, solid, ns.material(sectorMaterial_A[i])));
    tubsVol.placeVolume(sectorLogic, i + 1);  // copyNr: i+1
    LogDebug("TOBGeom") << sectorLogic.name() << " number " << i + 1 << " positioned in " << tubsName
                        << " with no translation and no rotation";

    // Second Part: B
    name = "TOBAxService_" + sectorNumber[i] + "B";
    startphi += deltaphi;
    deltaphi = sectorDeltaPhi_B;
    solid = ns.addSolid(name, Tube(rin, rout, dz, startphi, startphi + deltaphi));
    LogDebug("TOBGeom") << solid.name() << " Tubs made of " << sectorMaterial_B[i] << " from "
                        << convertRadToDeg(startphi) << " to " << convertRadToDeg((startphi + deltaphi)) << " with Rin "
                        << rin << " Rout " << rout << " ZHalf " << dz;

    sectorLogic = ns.addVolume(Volume(name, solid, ns.material(sectorMaterial_B[i])));
    tubsVol.placeVolume(sectorLogic, i + 1);  // copyNr: i+1
    LogDebug("TOBGeom") << sectorLogic.name() << " number " << i + 1 << " positioned in " << tubsName
                        << " with no translation and no rotation";

    // Third Part: C
    name = "TOBAxService_" + sectorNumber[i] + "C";
    startphi += deltaphi;
    deltaphi = 0.5 * (widthphi - sectorDeltaPhi_B);
    solid = ns.addSolid(name, Tube(rin, rout, dz, startphi, startphi + deltaphi));
    LogDebug("TOBGeom") << solid.name() << " Tubs made of " << sectorMaterial_C[i] << " from "
                        << convertRadToDeg(startphi) << " to " << convertRadToDeg((startphi + deltaphi)) << " with Rin "
                        << rin << " Rout " << rout << " ZHalf " << dz;
    sectorLogic = ns.addVolume(Volume(name, solid, ns.material(sectorMaterial_C[i])));
    tubsVol.placeVolume(sectorLogic, i + 1);  // copyNr: i+1
    LogDebug("TOBGeom") << sectorLogic.name() << " number " << i + 1 << " positioned in " << tubsName
                        << " with no translation and no rotation";
  }
  LogDebug("TOBGeom") << "<<== End of DDTOBAxCableAlgo construction ...";
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTOBAxCableAlgo, algorithm)
