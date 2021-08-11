#include "DD4hep/DetFactoryHelper.h"
#include <DD4hep/DD4hepUnits.h>
#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace angle_units::operators;

static long algorithm(Detector& /* description */, cms::DDParsingContext& context, xml_h element) {
  using VecDouble = vector<double>;

  cms::DDNamespace ns(context, element, true);
  DDAlgoArguments args(context, element);

  string mother = args.parentName();
  string genMat = args.str("GeneralMaterial");      //General material name
  double detectorTilt = args.dble("DetectorTilt");  //Detector Tilt
  double layerL = args.dble("LayerL");              //Length of the layer

  double radiusLo = args.dble("RadiusLo");          //Radius for detector at lower level
  int stringsLo = args.integer("StringsLo");        //Number of strings      ......
  string detectorLo = args.str("StringDetLoName");  //Detector string name   ......

  double radiusUp = args.dble("RadiusUp");          //Radius for detector at upper level
  int stringsUp = args.integer("StringsUp");        //Number of strings      ......
  string detectorUp = args.str("StringDetUpName");  //Detector string name   ......

  double cylinderT = args.dble("CylinderThickness");      //Cylinder thickness
  double cylinderInR = args.dble("CylinderInnerRadius");  //Cylinder inner radius
  string cylinderMat = args.str("CylinderMaterial");      //Cylinder material
  double MFRingInR = args.dble("MFRingInnerRadius");      //Inner Manifold Ring Inner Radius
  double MFRingOutR = args.dble("MFRingOuterRadius");     //Outer Manifold Ring Outer Radius
  double MFRingT = args.dble("MFRingThickness");          //Manifold Ring Thickness
  double MFRingDz = args.dble("MFRingDeltaz");            //Manifold Ring Half Lenght
  string MFIntRingMat = args.str("MFIntRingMaterial");    //Manifold Ring Material
  string MFExtRingMat = args.str("MFExtRingMaterial");    //Manifold Ring Material

  double supportT = args.dble("SupportThickness");  //Cylinder barrel CF skin thickness

  string centMat = args.str("CentRingMaterial");       //Central rings  material
  VecDouble centRing1par = args.vecDble("CentRing1");  //Central rings parameters
  VecDouble centRing2par = args.vecDble("CentRing2");  //Central rings parameters

  string fillerMat = args.str("FillerMaterial");  //Filler material
  double fillerDz = args.dble("FillerDeltaz");    //Filler Half Length

  string ribMat = args.str("RibMaterial");    //Rib material
  VecDouble ribW = args.vecDble("RibWidth");  //Rib width
  VecDouble ribPhi = args.vecDble("RibPhi");  //Rib Phi position

  VecDouble dohmListFW = args.vecDble("DOHMListFW");  //DOHM/AUX positions in #strings FW
  VecDouble dohmListBW = args.vecDble("DOHMListBW");  //DOHM/AUX positions in #strings BW

  double dohmtoMF = args.dble("DOHMtoMFDist");                   //DOHM Distance to MF
  double dohmCarrierPhiOff = args.dble("DOHMCarrierPhiOffset");  //DOHM Carrier Phi offset wrt horizontal
  string dohmPrimName = args.str("StringDOHMPrimName");          //DOHM Primary Logical Volume name
  string dohmAuxName = args.str("StringDOHMAuxName");            //DOHM Auxiliary Logical Volume name

  string dohmCarrierMaterial = args.str("DOHMCarrierMaterial");  //DOHM Carrier Material
  string dohmCableMaterial = args.str("DOHMCableMaterial");      //DOHM Cable Material
  double dohmPrimL = args.dble("DOHMPRIMLength");                //DOHM PRIMary Length
  string dohmPrimMaterial = args.str("DOHMPRIMMaterial");        //DOHM PRIMary Material
  double dohmAuxL = args.dble("DOHMAUXLength");                  //DOHM AUXiliary Length
  string dohmAuxMaterial = args.str("DOHMAUXMaterial");          //DOHM AUXiliary Material

  string pillarMaterial = args.str("PillarMaterial");  //Pillar Material

  double fwIntPillarDz = args.dble("FWIntPillarDz");  //Internal pillar parameters
  double fwIntPillarDPhi = args.dble("FWIntPillarDPhi");
  VecDouble fwIntPillarZ = args.vecDble("FWIntPillarZ");
  VecDouble fwIntPillarPhi = args.vecDble("FWIntPillarPhi");
  double bwIntPillarDz = args.dble("BWIntPillarDz");
  double bwIntPillarDPhi = args.dble("BWIntPillarDPhi");
  VecDouble bwIntPillarZ = args.vecDble("BWIntPillarZ");
  VecDouble bwIntPillarPhi = args.vecDble("BWIntPillarPhi");

  double fwExtPillarDz = args.dble("FWExtPillarDz");  //External pillar parameters
  double fwExtPillarDPhi = args.dble("FWExtPillarDPhi");
  VecDouble fwExtPillarZ = args.vecDble("FWExtPillarZ");
  VecDouble fwExtPillarPhi = args.vecDble("FWExtPillarPhi");
  double bwExtPillarDz = args.dble("BWExtPillarDz");
  double bwExtPillarDPhi = args.dble("BWExtPillarDPhi");
  VecDouble bwExtPillarZ = args.vecDble("BWExtPillarZ");
  VecDouble bwExtPillarPhi = args.vecDble("BWExtPillarPhi");

  dd4hep::PlacedVolume pv;

  auto LogPosition = [](dd4hep::PlacedVolume pvl) {
    edm::LogVerbatim("TIBGeom") << "DDTIBLayerAlgo: " << pvl.volume()->GetName()
                                << " positioned: " << pvl.motherVol()->GetName() << " " << pvl.position();
  };

  LogDebug("TIBGeom") << "Parent " << mother << " NameSpace " << ns.name() << " General Material " << genMat;
  LogDebug("TIBGeom") << "Lower layer Radius " << radiusLo << " Number " << stringsLo << " String " << detectorLo;
  LogDebug("TIBGeom") << "Upper layer Radius " << radiusUp << " Number " << stringsUp << " String " << detectorUp;
  LogDebug("TIBGeom") << "Cylinder Material/thickness " << cylinderMat << " " << cylinderT << " Rib Material " << ribMat
                      << " at " << ribW.size() << " positions with width/phi";
  for (unsigned int i = 0; i < ribW.size(); i++) {
    LogDebug("TIBGeom") << "\tribW[" << i << "] = " << ribW[i] << "\tribPhi[" << i
                        << "] = " << convertRadToDeg(ribPhi[i]);
  }
  LogDebug("TIBGeom") << "DOHM Primary "
                      << " Material " << dohmPrimMaterial << " Length " << dohmPrimL;
  LogDebug("TIBGeom") << "DOHM Aux     "
                      << " Material " << dohmAuxMaterial << " Length " << dohmAuxL;
  for (double i : dohmListFW) {
    if (i > 0.)
      LogDebug("TIBGeom") << "DOHM Primary at FW Position " << i;
    if (i < 0.)
      LogDebug("TIBGeom") << "DOHM Aux     at FW Position " << -i;
  }
  for (double i : dohmListBW) {
    if (i > 0.)
      LogDebug("TIBGeom") << "DOHM Primary at BW Position " << i;
    if (i < 0.)
      LogDebug("TIBGeom") << "DOHM Aux     at BW Position " << -i;
  }
  LogDebug("TIBGeom") << "FW Internal Pillar [Dz, DPhi] " << fwIntPillarDz << ", " << fwIntPillarDPhi;
  for (unsigned int i = 0; i < fwIntPillarZ.size(); i++) {
    if (fwIntPillarPhi[i] > 0.) {
      LogDebug("TIBGeom") << " at positions [z, phi] " << fwIntPillarZ[i] << " " << fwIntPillarPhi[i];
    }
  }
  LogDebug("TIBGeom") << "BW Internal Pillar [Dz, DPhi] " << bwIntPillarDz << ", " << bwIntPillarDPhi;
  for (unsigned int i = 0; i < bwIntPillarZ.size(); i++) {
    if (bwIntPillarPhi[i] > 0.) {
      LogDebug("TIBGeom") << " at positions [z, phi] " << bwIntPillarZ[i] << " " << bwIntPillarPhi[i];
    }
  }
  LogDebug("TIBGeom") << "FW External Pillar [Dz, DPhi] " << fwExtPillarDz << ", " << fwExtPillarDPhi;
  for (unsigned int i = 0; i < fwExtPillarZ.size(); i++) {
    if (fwExtPillarPhi[i] > 0.) {
      LogDebug("TIBGeom") << " at positions [z, phi] " << fwExtPillarZ[i] << " " << fwExtPillarPhi[i];
    }
  }
  LogDebug("TIBGeom") << "BW External Pillar [Dz, DPhi] " << bwExtPillarDz << ", " << bwExtPillarDPhi;
  for (unsigned int i = 0; i < bwExtPillarZ.size(); i++) {
    if (bwExtPillarPhi[i] > 0.) {
      LogDebug("TIBGeom") << " at positions [z, phi] " << bwExtPillarZ[i] << " " << bwExtPillarPhi[i];
    }
  }

  const string& idName = mother;
  double rmin = MFRingInR;
  double rmax = MFRingOutR;
  Solid solid = ns.addSolidNS(idName, Tube(rmin, rmax, 0.5 * layerL));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << genMat << " from 0 to " << convertRadToDeg(2_pi)
                      << " with Rin " << rmin << " Rout " << rmax << " ZHalf " << 0.5 * layerL;
  Volume layer = ns.addVolumeNS(Volume(idName, solid, ns.material(genMat)));

  //Internal layer first
  double rin = rmin + MFRingT;
  //  double rout = 0.5*(radiusLo+radiusUp-cylinderT);
  double rout = cylinderInR;
  string name = idName + "Down";
  solid = ns.addSolidNS(name, Tube(rin, rout, 0.5 * layerL));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << genMat << " from 0 to " << convertRadToDeg(2_pi)
                      << " with Rin " << rin << " Rout " << rout << " ZHalf " << 0.5 * layerL;
  Volume layerIn = ns.addVolumeNS(Volume(name, solid, ns.material(genMat)));
  pv = layer.placeVolume(layerIn, 1);
  LogPosition(pv);

  double rposdet = radiusLo;
  double dphi = 2_pi / stringsLo;
  Volume detIn = ns.volume(detectorLo);
  for (int n = 0; n < stringsLo; n++) {
    double phi = (n + 0.5) * dphi;
    double phix = phi - detectorTilt + 90_deg;
    double theta = 90_deg;
    double phiy = phix + 90._deg;
    Rotation3D rotation = makeRotation3D(theta, phix, theta, phiy, 0., 0.);
    Position trdet(rposdet * cos(phi), rposdet * sin(phi), 0);
    pv = layerIn.placeVolume(detIn, n + 1, Transform3D(rotation, trdet));
    LogPosition(pv);
  }
  //Now the external layer
  rin = cylinderInR + cylinderT;
  rout = rmax - MFRingT;
  name = idName + "Up";
  solid = ns.addSolidNS(name, Tube(rin, rout, 0.5 * layerL));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << genMat << " from 0 to " << convertRadToDeg(2_pi)
                      << " with Rin " << rin << " Rout " << rout << " ZHalf " << 0.5 * layerL;
  Volume layerOut = ns.addVolumeNS(Volume(name, solid, ns.material(genMat)));
  pv = layer.placeVolume(layerOut, 1);
  LogPosition(pv);

  rposdet = radiusUp;
  dphi = 2_pi / stringsUp;
  Volume detOut = ns.volume(detectorUp);
  for (int n = 0; n < stringsUp; n++) {
    double phi = (n + 0.5) * dphi;
    double phix = phi - detectorTilt - 90_deg;
    double theta = 90_deg;
    double phiy = phix + 90._deg;
    Rotation3D rotation = makeRotation3D(theta, phix, theta, phiy, 0., 0.);
    Position trdet(rposdet * cos(phi), rposdet * sin(phi), 0);
    pv = layerOut.placeVolume(detOut, n + 1, Transform3D(rotation, trdet));
    LogPosition(pv);
  }

  //
  // Inner cylinder, support wall and ribs
  //
  // External skins
  rin = cylinderInR;
  rout = cylinderInR + cylinderT;
  name = idName + "Cylinder";
  solid = ns.addSolidNS(name, Tube(rin, rout, 0.5 * layerL));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << cylinderMat << " from 0 to " << convertRadToDeg(2_pi)
                      << " with Rin " << rin << " Rout " << rout << " ZHalf " << 0.5 * layerL;
  Volume cylinder = ns.addVolumeNS(Volume(name, solid, ns.material(cylinderMat)));
  pv = layer.placeVolume(cylinder, 1);
  LogPosition(pv);

  //
  // inner part of the cylinder
  //
  rin += supportT;
  rout -= supportT;
  name = idName + "CylinderIn";
  solid = ns.addSolidNS(name, Tube(rin, rout, 0.5 * layerL));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << genMat << " from 0 to " << convertRadToDeg(2_pi)
                      << " with Rin " << rin << " Rout " << rout << " ZHalf " << 0.5 * layerL;
  Volume cylinderIn = ns.addVolumeNS(Volume(name, solid, ns.material(genMat)));
  pv = cylinder.placeVolume(cylinderIn, 1);
  LogPosition(pv);

  //
  // Filler Rings
  //
  name = idName + "Filler";
  solid = ns.addSolidNS(name, Tube(rin, rout, fillerDz));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << fillerMat << " from " << 0. << " to "
                      << convertRadToDeg(2_pi) << " with Rin " << rin << " Rout " << rout << " ZHalf " << fillerDz;
  Volume cylinderFiller = ns.addVolumeNS(Volume(name, solid, ns.material(fillerMat)));
  pv = cylinderIn.placeVolume(cylinderFiller, 1, Position(0.0, 0.0, 0.5 * layerL - fillerDz));
  LogPosition(pv);

  pv = cylinderIn.placeVolume(cylinderFiller, 2, Position(0.0, 0.0, -0.5 * layerL + fillerDz));
  LogPosition(pv);

  //
  // Ribs
  //
  Material matrib = ns.material(ribMat);
  for (size_t i = 0; i < ribW.size(); i++) {
    name = idName + "Rib" + std::to_string(i);
    double width = 2. * ribW[i] / (rin + rout);
    double dz = 0.5 * layerL - 2. * fillerDz;
    double _rmi = std::min(rin + 0.5 * dd4hep::mm, rout - 0.5 * dd4hep::mm);
    double _rma = std::max(rin + 0.5 * dd4hep::mm, rout - 0.5 * dd4hep::mm);
    solid = ns.addSolidNS(name, Tube(_rmi, _rma, dz, -0.5 * width, 0.5 * width));
    LogDebug("TIBGeom") << solid.name() << " Tubs made of " << ribMat << " from " << -0.5 * convertRadToDeg(width)
                        << " to " << 0.5 * convertRadToDeg(width) << " with Rin " << rin + 0.5 * dd4hep::mm << " Rout "
                        << rout - 0.5 * dd4hep::mm << " ZHalf " << dz;
    Volume cylinderRib = ns.addVolumeNS(Volume(name, solid, matrib));
    double phix = ribPhi[i];
    double theta = 90_deg;
    double phiy = phix + 90._deg;
    Rotation3D rotation = makeRotation3D(theta, phix, theta, phiy, 0., 0.);
    Position tran(0, 0, 0);
    pv = cylinderIn.placeVolume(cylinderRib, 1, Transform3D(rotation, tran));
    LogPosition(pv);
  }

  //
  //Manifold rings
  //
  // Inner ones first
  rin = MFRingInR;
  rout = rin + MFRingT;
  name = idName + "InnerMFRing";
  solid = ns.addSolidNS(name, Tube(rin, rout, MFRingDz));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << MFIntRingMat << " from 0 to " << convertRadToDeg(2_pi)
                      << " with Rin " << rin << " Rout " << rout << " ZHalf " << MFRingDz;

  Volume inmfr = ns.addVolumeNS(Volume(name, solid, ns.material(MFIntRingMat)));

  pv = layer.placeVolume(inmfr, 1, Position(0.0, 0.0, -0.5 * layerL + MFRingDz));
  LogPosition(pv);

  pv = layer.placeVolume(inmfr, 2, Position(0.0, 0.0, +0.5 * layerL - MFRingDz));
  LogPosition(pv);

  // Outer ones
  rout = MFRingOutR;
  rin = rout - MFRingT;
  name = idName + "OuterMFRing";
  solid = ns.addSolidNS(name, Tube(rin, rout, MFRingDz));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << MFExtRingMat << " from 0 to " << convertRadToDeg(2_pi)
                      << " with Rin " << rin << " Rout " << rout << " ZHalf " << MFRingDz;

  Volume outmfr = ns.addVolumeNS(Volume(name, solid, ns.material(MFExtRingMat)));
  pv = layer.placeVolume(outmfr, 1, Position(0.0, 0.0, -0.5 * layerL + MFRingDz));
  LogPosition(pv);
  pv = layer.placeVolume(outmfr, 2, Position(0.0, 0.0, +0.5 * layerL - MFRingDz));
  LogPosition(pv);

  //
  //Central Support rings
  //
  // Ring 1
  double centZ = centRing1par[0];
  double centDz = 0.5 * centRing1par[1];
  rin = centRing1par[2];
  rout = centRing1par[3];
  name = idName + "CentRing1";
  solid = ns.addSolidNS(name, Tube(rin, rout, centDz));

  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << centMat << " from 0 to " << convertRadToDeg(2_pi)
                      << " with Rin " << rin << " Rout " << rout << " ZHalf " << centDz;

  Volume cent1 = ns.addVolumeNS(Volume(name, solid, ns.material(centMat)));
  pv = layer.placeVolume(cent1, 1, Position(0.0, 0.0, centZ));
  LogPosition(pv);
  // Ring 2
  centZ = centRing2par[0];
  centDz = 0.5 * centRing2par[1];
  rin = centRing2par[2];
  rout = centRing2par[3];
  name = idName + "CentRing2";
  solid = ns.addSolidNS(name, Tube(rin, rout, centDz));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << centMat << " from 0 to " << convertRadToDeg(2_pi)
                      << " with Rin " << rin << " Rout " << rout << " ZHalf " << centDz;

  Volume cent2 = ns.addVolumeNS(Volume(name, solid, ns.material(centMat)));
  pv = layer.placeVolume(cent2, 1, Position(0e0, 0e0, centZ));
  LogPosition(pv);

  //
  ////// DOHM
  //
  // Preparing DOHM Carrier solid
  //
  name = idName + "DOHMCarrier";
  double dohmCarrierRin = MFRingOutR - MFRingT;
  double dohmCarrierRout = MFRingOutR;
  double dohmCarrierDz = 0.5 * (dohmPrimL + dohmtoMF);
  double dohmCarrierZ = 0.5 * layerL - 2. * MFRingDz - dohmCarrierDz;

  solid = ns.addSolidNS(
      name, Tube(dohmCarrierRin, dohmCarrierRout, dohmCarrierDz, dohmCarrierPhiOff, 180._deg - dohmCarrierPhiOff));
  LogDebug("TIBGeom") << solid.name() << " Tubs made of " << dohmCarrierMaterial << " from " << dohmCarrierPhiOff
                      << " to " << 180._deg - dohmCarrierPhiOff << " with Rin " << dohmCarrierRin << " Rout "
                      << MFRingOutR << " ZHalf " << dohmCarrierDz;

  // Define FW and BW carrier logical volume and
  // place DOHM Primary and auxiliary modules inside it
  dphi = 2_pi / stringsUp;

  Rotation3D dohmRotation;
  double dohmR = 0.5 * (dohmCarrierRin + dohmCarrierRout);

  for (int j = 0; j < 4; j++) {
    vector<double> dohmList;
    Position tran;
    string rotstr;
    Rotation3D rotation;
    int dohmCarrierReplica = 0;
    int placeDohm = 0;

    Volume dohmCarrier;

    switch (j) {
      case 0:
        name = idName + "DOHMCarrierFW";
        dohmCarrier = ns.addVolumeNS(Volume(name, solid, ns.material(dohmCarrierMaterial)));
        dohmList = dohmListFW;
        tran = Position(0., 0., dohmCarrierZ);
        rotstr = idName + "FwUp";
        rotation = Rotation3D();
        dohmCarrierReplica = 1;
        placeDohm = 1;
        break;
      case 1:
        name = idName + "DOHMCarrierFW";
        dohmCarrier = ns.volume(name);  // Re-use internally stored DOHMCarrierFW Volume
        dohmList = dohmListFW;
        tran = Position(0., 0., dohmCarrierZ);
        rotstr = idName + "FwDown";
        rotation = makeRotation3D(90._deg, 180._deg, 90._deg, 270._deg, 0., 0.);
        dohmCarrierReplica = 2;
        placeDohm = 0;
        break;
      case 2:
        name = idName + "DOHMCarrierBW";
        dohmCarrier = ns.addVolumeNS(Volume(name, solid, ns.material(dohmCarrierMaterial)));
        dohmList = dohmListBW;
        tran = Position(0., 0., -dohmCarrierZ);
        rotstr = idName + "BwUp";
        rotation = makeRotation3D(90._deg, 180._deg, 90._deg, 90._deg, 180._deg, 0.);
        dohmCarrierReplica = 1;
        placeDohm = 1;
        break;
      case 3:
        name = idName + "DOHMCarrierBW";
        dohmCarrier = ns.volume(name);  // Re-use internally stored DOHMCarrierBW Volume
        dohmList = dohmListBW;
        tran = Position(0., 0., -dohmCarrierZ);
        rotstr = idName + "BwDown";
        rotation = makeRotation3D(90._deg, 0., 90._deg, 270._deg, 180._deg, 0.);
        dohmCarrierReplica = 2;
        placeDohm = 0;
        break;
    }

    int primReplica = 0;
    int auxReplica = 0;

    for (size_t i = 0; i < placeDohm * dohmList.size(); i++) {
      double phi = (std::abs(dohmList[i]) + 0.5 - 1.) * dphi;
      double phix = phi + 90_deg;
      double phideg = convertRadToDeg(phix);
      if (phideg != 0) {
        double theta = 90_deg;
        double phiy = phix + 90._deg;
        dohmRotation = makeRotation3D(theta, phix, theta, phiy, 0., 0.);
      }
      int dohmReplica = 0;
      double dohmZ = 0.;
      Volume dohm;

      if (dohmList[i] < 0.) {
        // Place a Auxiliary DOHM
        dohm = ns.volume(dohmAuxName);
        dohmZ = dohmCarrierDz - 0.5 * dohmAuxL - dohmtoMF;
        primReplica++;
        dohmReplica = primReplica;
      } else {
        // Place a Primary DOHM
        dohm = ns.volume(dohmPrimName);
        dohmZ = dohmCarrierDz - 0.5 * dohmPrimL - dohmtoMF;
        auxReplica++;
        dohmReplica = auxReplica;
      }
      Position dohmTrasl(dohmR * cos(phi), dohmR * sin(phi), dohmZ);
      pv = dohmCarrier.placeVolume(dohm, dohmReplica, Transform3D(dohmRotation, dohmTrasl));
      LogPosition(pv);
    }

    pv = layer.placeVolume(dohmCarrier, dohmCarrierReplica, Transform3D(rotation, tran));
    LogPosition(pv);
  }
  //
  ////// PILLARS
  for (int j = 0; j < 4; j++) {
    vector<double> pillarZ;
    vector<double> pillarPhi;
    double pillarDz = 0, pillarDPhi = 0, pillarRin = 0, pillarRout = 0;

    switch (j) {
      case 0:
        name = idName + "FWIntPillar";
        pillarZ = fwIntPillarZ;
        pillarPhi = fwIntPillarPhi;
        pillarRin = MFRingInR;
        pillarRout = MFRingInR + MFRingT;
        pillarDz = fwIntPillarDz;
        pillarDPhi = fwIntPillarDPhi;
        break;
      case 1:
        name = idName + "BWIntPillar";
        pillarZ = bwIntPillarZ;
        pillarPhi = bwIntPillarPhi;
        pillarRin = MFRingInR;
        pillarRout = MFRingInR + MFRingT;
        pillarDz = bwIntPillarDz;
        pillarDPhi = bwIntPillarDPhi;
        break;
      case 2:
        name = idName + "FWExtPillar";
        pillarZ = fwExtPillarZ;
        pillarPhi = fwExtPillarPhi;
        pillarRin = MFRingOutR - MFRingT;
        pillarRout = MFRingOutR;
        pillarDz = fwExtPillarDz;
        pillarDPhi = fwExtPillarDPhi;
        break;
      case 3:
        name = idName + "BWExtPillar";
        pillarZ = bwExtPillarZ;
        pillarPhi = bwExtPillarPhi;
        pillarRin = MFRingOutR - MFRingT;
        pillarRout = MFRingOutR;
        pillarDz = bwExtPillarDz;
        pillarDPhi = bwExtPillarDPhi;
        break;
    }

    solid = ns.addSolidNS(name, Tube(pillarRin, pillarRout, pillarDz, -pillarDPhi, pillarDPhi));
    Volume Pillar = ns.addVolumeNS(Volume(name, solid, ns.material(pillarMaterial)));
    LogDebug("TIBGeom") << solid.name() << " Tubs made of " << pillarMaterial << " from " << -pillarDPhi << " to "
                        << pillarDPhi << " with Rin " << pillarRin << " Rout " << pillarRout << " ZHalf " << pillarDz;
    Position pillarTran;
    Rotation3D pillarRota;
    int pillarReplica = 0;
    for (unsigned int i = 0; i < pillarZ.size(); i++) {
      if (pillarPhi[i] > 0.) {
        pillarTran = Position(0., 0., pillarZ[i]);
        pillarRota = makeRotation3D(90._deg, pillarPhi[i], 90._deg, 90._deg + pillarPhi[i], 0., 0.);
        pv = layer.placeVolume(Pillar, i, Transform3D(pillarRota, pillarTran));
        LogPosition(pv);
        pillarReplica++;
      }
    }
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTIBLayerAlgo, algorithm)
