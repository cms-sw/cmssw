#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/Printout.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/BenchmarkGrd.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloGeometry/interface/EcalTrapezoidParameters.h"
#include "Math/AxisAngle.h"
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Geometry/Transform3D.h"

//#define EDM_ML_DEBUG

using namespace std;
using namespace cms;
using namespace dd4hep;
using namespace angle_units::operators;

using VecDouble = vector<double>;
using VecStr = vector<string>;
using EcalTrap = EcalTrapezoidParameters;

using Pt3D = HepGeom::Point3D<double>;
using Vec3 = CLHEP::Hep3Vector;
using Rota = CLHEP::HepRotation;
using Ro3D = HepGeom::Rotate3D;
using Tl3D = HepGeom::Translate3D;
using Tf3D = HepGeom::Transform3D;
using RoX3D = HepGeom::RotateX3D;
using RoZ3D = HepGeom::RotateZ3D;

namespace {

  // Barrel volume
  struct Barrel {
    string name;         // Barrel volume name
    string mat;          // Barrel material name
    VecDouble vecZPts;   // Barrel list of z pts
    VecDouble vecRMin;   // Barrel list of rMin pts
    VecDouble vecRMax;   // Barrel list of rMax pts
    VecDouble vecTran;   // Barrel translation
    VecDouble vecRota;   // Barrel rotation
    VecDouble vecRota2;  // 2nd Barrel rotation
    VecDouble vecRota3;  // 3nd Barrel rotation
    double phiLo;        // Barrel phi lo
    double phiHi;        // Barrel phi hi
    double here;         // Barrel presence flag
  };

  // Supermodule volume
  struct Supermodule {
    string name;            // Supermodule volume name
    string mat;             // Supermodule material name
    VecDouble vecZPts;      // Supermodule list of z pts
    VecDouble vecRMin;      // Supermodule list of rMin pts
    VecDouble vecRMax;      // Supermodule list of rMax pts
    VecDouble vecTran;      // Supermodule translation
    VecDouble vecRota;      // Supermodule rotation
    VecDouble vecBTran;     // Base Supermodule translation
    VecDouble vecBRota;     // Base Supermodule rotation
    unsigned int nPerHalf;  // # Supermodules per half detector
    double lowPhi;          // Low   phi value of base supermodule
    double delPhi;          // Delta phi value of base supermodule
    double phiOff;          // Phi offset value supermodule
    VecDouble vecHere;      // Bit saying if a supermodule is present or not
    string cutName;         // Name of cut box
    double cutThick;        // Box thickness
    int cutShow;            // Non-zero means show the box on display (testing only)
    VecDouble vecCutTM;     // Translation for minus phi cut box
    VecDouble vecCutTP;     // Translation for plus  phi cut box
    double cutRM;           // Rotation for minus phi cut box
    double cutRP;           // Rotation for plus  phi cut box
    double expThick;        // Thickness (x) of supermodule expansion box
    double expWide;         // Width     (y) of supermodule expansion box
    double expYOff;         // Offset    (y) of supermodule expansion box
    string sideName;        // Supermodule Side Plate volume name
    string sideMat;         // Supermodule Side Plate material name
    double sideHigh;        // Side plate height
    double sideThick;       // Side plate thickness
    double sideYOffM;       // Side plate Y offset on minus phi side
    double sideYOffP;       // Side plate Y offset on plus  phi side
  };

  struct Crystal {
    VecDouble vecNomCryDimBF;  // Nominal crystal BF
    VecDouble vecNomCryDimCF;  // Nominal crystal CF
    VecDouble vecNomCryDimAR;  // Nominal crystal AR
    VecDouble vecNomCryDimBR;  // Nominal crystal BR
    VecDouble vecNomCryDimCR;  // Nominal crystal CR
    double nomCryDimAF;        // Nominal crystal AF
    double nomCryDimLZ;        // Nominal crystal LZ

    double underAF;  // undershoot of AF
    double underLZ;  // undershoot of LZ
    double underBF;  // undershoot of BF
    double underCF;  // undershoot of CF
    double underAR;  // undershoot of AR
    double underBR;  // undershoot of BR
    double underCR;  // undershoot of CR

    string name;      // string name of crystal volume
    string clrName;   // string name of clearance volume
    string wrapName;  // string name of wrap volume
    string wallName;  // string name of wall volume

    string mat;      // string name of crystal material
    string clrMat;   // string name of clearance material
    string wrapMat;  // string name of wrap material
    string wallMat;  // string name of wall material
  };

  struct Alveolus {
    double wallThAlv;        // alveoli wall thickness
    double wrapThAlv;        // wrapping thickness
    double clrThAlv;         // clearance thickness (nominal)
    VecDouble vecGapAlvEta;  // Extra clearance after each alveoli perp to crystal axis

    double wallFrAlv;  // alveoli wall frontage
    double wrapFrAlv;  // wrapping frontage
    double clrFrAlv;   // clearance frontage (nominal)

    double wallReAlv;  // alveoli wall rearage
    double wrapReAlv;  // wrapping rearage
    double clrReAlv;   // clearance rearage (nominal)

    unsigned int nCryTypes;      // number of crystal shapes
    unsigned int nCryPerAlvEta;  // number of crystals in eta per alveolus
  };
  struct Capsule {
    string name;       // Capsule
    double here;       //
    string mat;        //
    double xSizeHalf;  //
    double ySizeHalf;  //
    double thickHalf;  //
  };

  struct Ceramic {
    string name;       // Ceramic
    string mat;        //
    double xSizeHalf;  //
    double ySizeHalf;  //
    double thickHalf;  //
  };

  struct BulkSilicon {
    string name;       // Bulk Silicon
    string mat;        //
    double xSizeHalf;  //
    double ySizeHalf;  //
    double thickHalf;  //
  };

  struct APD {
    string name;   // APD
    string mat;    //
    double side;   //
    double thick;  //
    double z;      //
    double x1;     //
    double x2;     //

    string atjName;       // After-The-Junction
    string atjMat;        //
    double atjThickHalf;  //

    string sglName;   // APD-Silicone glue
    string sglMat;    //
    double sglThick;  //

    string aglName;   // APD-Glue
    string aglMat;    //
    double aglThick;  //

    string andName;   // APD-Non-Depleted
    string andMat;    //
    double andThick;  //
  };

  struct Web {
    double here;             // here flag
    string plName;           // string name of web plate volume
    string clrName;          // string name of web clearance volume
    string plMat;            // string name of web material
    string clrMat;           // string name of web clearance material
    VecDouble vecWebPlTh;    // Thickness of web plates
    VecDouble vecWebClrTh;   // Thickness of total web clearance
    VecDouble vecWebLength;  // Length of web plate
  };

  struct InnerLayerVolume {
    double here;            // here flag
    string name;            // string name of inner layer volume
    double phiLow;          // low phi of volumes
    double delPhi;          // delta phi of ily
    VecStr vecIlyMat;       // materials of inner layer volumes
    VecDouble vecIlyThick;  // Thicknesses of inner layer volumes

    string pipeName;                 // Cooling pipes
    double pipeHere;                 //
    string pipeMat;                  //
    double pipeODHalf;               //
    double pipeID;                   //
    VecDouble vecIlyPipeLengthHalf;  //
    VecDouble vecIlyPipeType;        //
    VecDouble vecIlyPipePhi;         //
    VecDouble vecIlyPipeZ;           //

    string pTMName;          // PTM
    double pTMHere;          //
    string pTMMat;           //
    double pTMWidthHalf;     //
    double pTMLengthHalf;    //
    double pTMHeightHalf;    //
    VecDouble vecIlyPTMZ;    //
    VecDouble vecIlyPTMPhi;  //

    string fanOutName;          // FanOut
    double fanOutHere;          //
    string fanOutMat;           //
    double fanOutWidthHalf;     //
    double fanOutLengthHalf;    //
    double fanOutHeightHalf;    //
    VecDouble vecIlyFanOutZ;    //
    VecDouble vecIlyFanOutPhi;  //
    string diffName;            // Diffuser
    string diffMat;             //
    double diffOff;             //
    double diffLengthHalf;      //
    string bndlName;            // Fiber bundle
    string bndlMat;             //
    double bndlOff;             //
    double bndlLengthHalf;      //
    string fEMName;             // FEM
    string fEMMat;              //
    double fEMWidthHalf;        //
    double fEMLengthHalf;       //
    double fEMHeightHalf;       //
    VecDouble vecIlyFEMZ;       //
    VecDouble vecIlyFEMPhi;     //
  };

  struct AlveolarWedge {
    string hawRName;           // string name of half-alveolar wedge
    string fawName;            // string name of full-alveolar wedge
    double fawHere;            // here flag
    double hawRHBIG;           // height at big end of half alveolar wedge
    double hawRhsml;           // height at small end of half alveolar wedge
    double hawRCutY;           // x dim of hawR cut box
    double hawRCutZ;           // y dim of hawR cut box
    double hawRCutDelY;        // y offset of hawR cut box from top of HAW
    double hawYOffCry;         // Y offset of crystal wrt HAW at front
    unsigned int nFawPerSupm;  // Number of Full Alv. Wedges per supermodule
    double fawPhiOff;          // Phi offset for FAW placement
    double fawDelPhi;          // Phi delta for FAW placement
    double fawPhiRot;          // Phi rotation of FAW about own axis prior to placement
    double fawRadOff;          // Radial offset for FAW placement
  };

  struct Grid {
    double here;   // here flag
    string name;   // Grid name
    string mat;    // Grid material
    double thick;  // Grid Thickness
  };

  struct Back {
    double xOff;         //
    double yOff;         //
    double here;         // here flag
    string sideName;     //  Back Sides
    double sideHere;     //
    double sideLength;   //
    double sideHeight;   //
    double sideWidth;    //
    double sideYOff1;    //
    double sideYOff2;    //
    double sideAngle;    //
    string sideMat;      //
    string plateName;    // back plate
    double plateHere;    //
    double plateLength;  //
    double plateThick;   //
    double plateWidth;   //
    string plateMat;     //
    string plate2Name;   // back plate2
    double plate2Thick;  //
    string plate2Mat;    //
  };

  struct Grille {
    string name;          // grille
    double here;          //
    double thick;         //
    double width;         //
    double zSpace;        //
    string mat;           //
    VecDouble vecHeight;  //
    VecDouble vecZOff;    //

    string edgeSlotName;    // Slots in Grille
    string edgeSlotMat;     //
    double edgeSlotHere;    //
    double edgeSlotHeight;  //
    double edgeSlotWidth;   //

    string midSlotName;          // Slots in Grille
    string midSlotMat;           //
    double midSlotHere;          //
    double midSlotWidth;         //
    double midSlotXOff;          //
    VecDouble vecMidSlotHeight;  //
  };

  struct BackPipe {
    double here;         // here flag
    string name;         //
    VecDouble vecDiam;   // pipes
    VecDouble vecThick;  // pipes
    string mat;          //
    string waterMat;     //
  };

  struct BackCooling {
    VecStr vecName;    // cooling circuits
    double here;       // here flag
    double barHere;    // here flag
    double barWidth;   //
    double barHeight;  //
    string mat;
    string barName;   // cooling bar
    double barThick;  //
    string barMat;
    string barSSName;   // cooling bar tubing
    double barSSThick;  //
    string barSSMat;
    string barWaName;   // cooling bar water
    double barWaThick;  //
    string barWaMat;
    double vFEHere;  // here flag
    string vFEName;
    string vFEMat;
    string backVFEName;
    string backVFEMat;
    VecDouble vecBackVFELyrThick;  //
    VecStr vecBackVFELyrName;      //
    VecStr vecBackVFELyrMat;       //
    VecDouble vecBackCoolNSec;     //
    VecDouble vecBackCoolSecSep;   //
    VecDouble vecBackCoolNPerSec;  //
  };

  struct BackMisc {
    double here;          // here flag
    VecDouble vecThick;   // misc materials
    VecStr vecName;       //
    VecStr vecMat;        //
    double backCBStdSep;  //
  };

  struct PatchPanel {
    double here;         // here flag
    string name;         //
    VecDouble vecThick;  // patch panel materials
    VecStr vecNames;     //
    VecStr vecMat;       //
  };

  struct BackCoolTank {
    double here;               // here flag
    string name;               // service tank
    double width;              //
    double thick;              //
    string mat;                //
    string waName;             //
    double waWidth;            //
    string waMat;              //
    string backBracketName;    //
    double backBracketHeight;  //
    string backBracketMat;     //
  };

  struct DryAirTube {
    double here;                 // here flag
    string name;                 // dry air tube
    unsigned int mbCoolTubeNum;  //
    double innDiam;              //
    double outDiam;              //
    string mat;                  //
  };

  struct MBCoolTube {
    double here;     // here flag
    string name;     // mothr bd cooling tube
    double innDiam;  //
    double outDiam;  //
    string mat;      //
  };

  struct MBManif {
    double here;     // here flag
    string name;     //mother bd manif
    double innDiam;  //
    double outDiam;  //
    string mat;      //
  };

  struct MBLyr {
    double here;              // here flag
    VecDouble vecMBLyrThick;  // mother bd lyrs
    VecStr vecMBLyrName;      //
    VecStr vecMBLyrMat;       //
  };

  struct Pincer {
    double rodHere;           // here flag
    string rodName;           // pincer rod
    string rodMat;            //
    VecDouble vecRodAzimuth;  //

    string envName;        // pincer envelope
    string envMat;         //
    double envWidthHalf;   //
    double envHeightHalf;  //
    double envLengthHalf;  //
    VecDouble vecEnvZOff;  //

    string blkName;        // pincer block
    string blkMat;         //
    double blkLengthHalf;  //

    string shim1Name;   // pincer shim
    double shimHeight;  //
    string shim2Name;   //
    string shimMat;     //
    double shim1Width;  //
    double shim2Width;  //

    string cutName;    // pincer block
    string cutMat;     //
    double cutWidth;   //
    double cutHeight;  //
  };

  const Rotation3D& myrot(cms::DDNamespace& ns, const string& nam, const CLHEP::HepRotation& r) {
    ns.addRotation(nam, Rotation3D(r.xx(), r.xy(), r.xz(), r.yx(), r.yy(), r.yz(), r.zx(), r.zy(), r.zz()));
    return ns.rotation(ns.prepend(nam));
  }

  Solid mytrap(const std::string& nam, const EcalTrapezoidParameters& t) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << nam << " Trap " << convertRadToDeg(t.theta()) << ":" << convertRadToDeg(t.phi())
                               << ":" << cms::convert2mm(t.h1()) << ":" << cms::convert2mm(t.bl1()) << ":"
                               << cms::convert2mm(t.tl1()) << ":" << convertRadToDeg(t.alp1()) << ":"
                               << cms::convert2mm(t.h2()) << ":" << cms::convert2mm(t.bl2()) << ":"
                               << cms::convert2mm(t.tl2()) << ":" << convertRadToDeg(t.alp2());
#endif
    return Trap(
        nam, t.dz(), t.theta(), t.phi(), t.h1(), t.bl1(), t.tl1(), t.alp1(), t.h2(), t.bl2(), t.tl2(), t.alp2());
  }

  string_view mynamespace(string_view input) {
    string_view v = input;
    auto trim_pos = v.find(':');
    if (trim_pos != v.npos)
      v.remove_suffix(v.size() - (trim_pos + 1));
    return v;
  }
}  // namespace

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  BenchmarkGrd counter("DDEcalBarrelNewAlgo");
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  // TRICK!
  string myns{mynamespace(args.parentName()).data(), mynamespace(args.parentName()).size()};

  // Barrel volume
  // barrel parent volume
  Barrel bar;
  bar.name = myns + args.str("BarName");    // Barrel volume name
  bar.mat = args.str("BarMat");             // Barrel material name
  bar.vecZPts = args.vecDble("BarZPts");    // Barrel list of z pts
  bar.vecRMin = args.vecDble("BarRMin");    // Barrel list of rMin pts
  bar.vecRMax = args.vecDble("BarRMax");    // Barrel list of rMax pts
  bar.vecTran = args.vecDble("BarTran");    // Barrel translation
  bar.vecRota = args.vecDble("BarRota");    // Barrel rotation
  bar.vecRota2 = args.vecDble("BarRota2");  // 2nd Barrel rotation
  bar.vecRota3 = args.vecDble("BarRota3");  // 3rd Barrel rotation
  bar.phiLo = args.dble("BarPhiLo");        // Barrel phi lo
  bar.phiHi = args.dble("BarPhiHi");        // Barrel phi hi
  bar.here = args.dble("BarHere");          // Barrel presence flag

  // Supermodule volume
  Supermodule spm;
  spm.name = ns.prepend(args.str("SpmName"));        // Supermodule volume name
  spm.mat = args.str("SpmMat");                      // Supermodule material name
  spm.vecZPts = args.vecDble("SpmZPts");             // Supermodule list of z pts
  spm.vecRMin = args.vecDble("SpmRMin");             // Supermodule list of rMin pts
  spm.vecRMax = args.vecDble("SpmRMax");             // Supermodule list of rMax pts
  spm.vecTran = args.vecDble("SpmTran");             // Supermodule translation
  spm.vecRota = args.vecDble("SpmRota");             // Supermodule rotation
  spm.vecBTran = args.vecDble("SpmBTran");           // Base Supermodule translation
  spm.vecBRota = args.vecDble("SpmBRota");           // Base Supermodule rotation
  spm.nPerHalf = args.integer("SpmNPerHalf");        // # Supermodules per half detector
  spm.lowPhi = args.dble("SpmLowPhi");               // Low   phi value of base supermodule
  spm.delPhi = args.dble("SpmDelPhi");               // Delta phi value of base supermodule
  spm.phiOff = args.dble("SpmPhiOff");               // Phi offset value supermodule
  spm.vecHere = args.vecDble("SpmHere");             // Bit saying if a supermodule is present or not
  spm.cutName = ns.prepend(args.str("SpmCutName"));  // Name of cut box
  spm.cutThick = args.dble("SpmCutThick");           // Box thickness
  spm.cutShow = args.value<int>("SpmCutShow");       // Non-zero means show the box on display (testing only)
  spm.vecCutTM = args.vecDble("SpmCutTM");           // Translation for minus phi cut box
  spm.vecCutTP = args.vecDble("SpmCutTP");           // Translation for plus  phi cut box
  spm.cutRM = args.dble("SpmCutRM");                 // Rotation for minus phi cut box
  spm.cutRP = args.dble("SpmCutRP");                 // Rotation for plus  phi cut box
  spm.expThick = args.dble("SpmExpThick");           // Thickness (x) of supermodule expansion box
  spm.expWide = args.dble("SpmExpWide");             // Width     (y) of supermodule expansion box
  spm.expYOff = args.dble("SpmExpYOff");             // Offset    (y) of supermodule expansion box
  spm.sideName = myns + args.str("SpmSideName");     // Supermodule Side Plate volume name
  spm.sideMat = args.str("SpmSideMat");              // Supermodule Side Plate material name
  spm.sideHigh = args.dble("SpmSideHigh");           // Side plate height
  spm.sideThick = args.dble("SpmSideThick");         // Side plate thickness
  spm.sideYOffM = args.dble("SpmSideYOffM");         // Side plate Y offset on minus phi side
  spm.sideYOffP = args.dble("SpmSideYOffP");         // Side plate Y offset on plus  phi side

  Crystal cry;
  cry.nomCryDimAF = args.dble("NomCryDimAF");
  cry.nomCryDimLZ = args.dble("NomCryDimLZ");
  cry.vecNomCryDimBF = args.vecDble("NomCryDimBF");
  cry.vecNomCryDimCF = args.vecDble("NomCryDimCF");
  cry.vecNomCryDimAR = args.vecDble("NomCryDimAR");
  cry.vecNomCryDimBR = args.vecDble("NomCryDimBR");
  cry.vecNomCryDimCR = args.vecDble("NomCryDimCR");

  cry.underAF = args.dble("UnderAF");
  cry.underLZ = args.dble("UnderLZ");
  cry.underBF = args.dble("UnderBF");
  cry.underCF = args.dble("UnderCF");
  cry.underAR = args.dble("UnderAR");
  cry.underBR = args.dble("UnderBR");
  cry.underCR = args.dble("UnderCR");

  Alveolus alv;
  alv.wallThAlv = args.dble("WallThAlv");
  alv.wrapThAlv = args.dble("WrapThAlv");
  alv.clrThAlv = args.dble("ClrThAlv");
  alv.vecGapAlvEta = args.vecDble("GapAlvEta");

  alv.wallFrAlv = args.dble("WallFrAlv");
  alv.wrapFrAlv = args.dble("WrapFrAlv");
  alv.clrFrAlv = args.dble("ClrFrAlv");

  alv.wallReAlv = args.dble("WallReAlv");
  alv.wrapReAlv = args.dble("WrapReAlv");
  alv.clrReAlv = args.dble("ClrReAlv");

  alv.nCryTypes = args.integer("NCryTypes");
  alv.nCryPerAlvEta = args.integer("NCryPerAlvEta");

  cry.name = ns.prepend(args.str("CryName"));
  cry.clrName = ns.prepend(args.str("ClrName"));
  cry.wrapName = ns.prepend(args.str("WrapName"));
  cry.wallName = ns.prepend(args.str("WallName"));

  cry.mat = args.str("CryMat");
  cry.clrMat = args.str("ClrMat");
  cry.wrapMat = args.str("WrapMat");
  cry.wallMat = args.str("WallMat");

  Capsule cap;
  cap.name = ns.prepend(args.str("CapName"));
  cap.here = args.dble("CapHere");
  cap.mat = args.str("CapMat");
  cap.xSizeHalf = 0.5 * args.dble("CapXSize");
  cap.ySizeHalf = 0.5 * args.dble("CapYSize");
  cap.thickHalf = 0.5 * args.dble("CapThick");

  Ceramic cer;
  cer.name = ns.prepend(args.str("CerName"));
  cer.mat = args.str("CerMat");
  cer.xSizeHalf = 0.5 * args.dble("CerXSize");
  cer.ySizeHalf = 0.5 * args.dble("CerYSize");
  cer.thickHalf = 0.5 * args.dble("CerThick");

  BulkSilicon bSi;
  bSi.name = ns.prepend(args.str("BSiName"));
  bSi.mat = args.str("BSiMat");
  bSi.xSizeHalf = 0.5 * args.dble("BSiXSize");
  bSi.ySizeHalf = 0.5 * args.dble("BSiYSize");
  bSi.thickHalf = 0.5 * args.dble("BSiThick");

  APD apd;
  apd.name = ns.prepend(args.str("APDName"));
  apd.mat = args.str("APDMat");
  apd.side = args.dble("APDSide");
  apd.thick = args.dble("APDThick");
  apd.z = args.dble("APDZ");
  apd.x1 = args.dble("APDX1");
  apd.x2 = args.dble("APDX2");

  apd.atjName = ns.prepend(args.str("ATJName"));
  apd.atjMat = args.str("ATJMat");
  apd.atjThickHalf = 0.5 * args.dble("ATJThick");

  apd.sglName = ns.prepend(args.str("SGLName"));
  apd.sglMat = args.str("SGLMat");
  apd.sglThick = args.dble("SGLThick");

  apd.aglName = ns.prepend(args.str("AGLName"));
  apd.aglMat = args.str("AGLMat");
  apd.aglThick = args.dble("AGLThick");

  apd.andName = ns.prepend(args.str("ANDName"));
  apd.andMat = args.str("ANDMat");
  apd.andThick = args.dble("ANDThick");

  Web web;
  web.here = args.dble("WebHere");
  web.plName = ns.prepend(args.str("WebPlName"));
  web.clrName = ns.prepend(args.str("WebClrName"));
  web.plMat = args.str("WebPlMat");
  web.clrMat = args.str("WebClrMat");
  web.vecWebPlTh = args.vecDble("WebPlTh");
  web.vecWebClrTh = args.vecDble("WebClrTh");
  web.vecWebLength = args.vecDble("WebLength");

  InnerLayerVolume ily;
  ily.here = args.dble("IlyHere");
  ily.name = myns + args.str("IlyName");
  ily.phiLow = args.dble("IlyPhiLow");
  ily.delPhi = args.dble("IlyDelPhi");
  ily.vecIlyMat = args.vecStr("IlyMat");
  ily.vecIlyThick = args.vecDble("IlyThick");

  ily.pipeName = myns + args.str("IlyPipeName");
  ily.pipeHere = args.dble("IlyPipeHere");
  ily.pipeMat = args.str("IlyPipeMat");
  ily.pipeODHalf = 0.5 * args.dble("IlyPipeOD");
  ily.pipeID = args.dble("IlyPipeID");
  ily.vecIlyPipeLengthHalf = args.vecDble("IlyPipeLength");
  std::transform(ily.vecIlyPipeLengthHalf.begin(),
                 ily.vecIlyPipeLengthHalf.end(),
                 ily.vecIlyPipeLengthHalf.begin(),
                 [](double length) -> double { return 0.5 * length; });

  ily.vecIlyPipeType = args.vecDble("IlyPipeType");
  ily.vecIlyPipePhi = args.vecDble("IlyPipePhi");
  ily.vecIlyPipeZ = args.vecDble("IlyPipeZ");

  ily.pTMName = myns + args.str("IlyPTMName");
  ily.pTMHere = args.dble("IlyPTMHere");
  ily.pTMMat = args.str("IlyPTMMat");
  ily.pTMWidthHalf = 0.5 * args.dble("IlyPTMWidth");
  ily.pTMLengthHalf = 0.5 * args.dble("IlyPTMLength");
  ily.pTMHeightHalf = 0.5 * args.dble("IlyPTMHeight");
  ily.vecIlyPTMZ = args.vecDble("IlyPTMZ");
  ily.vecIlyPTMPhi = args.vecDble("IlyPTMPhi");

  ily.fanOutName = myns + args.str("IlyFanOutName");
  ily.fanOutHere = args.dble("IlyFanOutHere");
  ily.fanOutMat = args.str("IlyFanOutMat");
  ily.fanOutWidthHalf = 0.5 * args.dble("IlyFanOutWidth");
  ily.fanOutLengthHalf = 0.5 * args.dble("IlyFanOutLength");
  ily.fanOutHeightHalf = 0.5 * args.dble("IlyFanOutHeight");
  ily.vecIlyFanOutZ = args.vecDble("IlyFanOutZ");
  ily.vecIlyFanOutPhi = args.vecDble("IlyFanOutPhi");
  ily.diffName = myns + args.str("IlyDiffName");
  ily.diffMat = args.str("IlyDiffMat");
  ily.diffOff = args.dble("IlyDiffOff");
  ily.diffLengthHalf = 0.5 * args.dble("IlyDiffLength");
  ily.bndlName = myns + args.str("IlyBndlName");
  ily.bndlMat = args.str("IlyBndlMat");
  ily.bndlOff = args.dble("IlyBndlOff");
  ily.bndlLengthHalf = 0.5 * args.dble("IlyBndlLength");
  ily.fEMName = myns + args.str("IlyFEMName");
  ily.fEMMat = args.str("IlyFEMMat");
  ily.fEMWidthHalf = 0.5 * args.dble("IlyFEMWidth");
  ily.fEMLengthHalf = 0.5 * args.dble("IlyFEMLength");
  ily.fEMHeightHalf = 0.5 * args.dble("IlyFEMHeight");
  ily.vecIlyFEMZ = args.vecDble("IlyFEMZ");
  ily.vecIlyFEMPhi = args.vecDble("IlyFEMPhi");

  AlveolarWedge alvWedge;
  alvWedge.hawRName = myns + args.str("HawRName");
  alvWedge.fawName = myns + args.str("FawName");
  alvWedge.fawHere = args.dble("FawHere");
  alvWedge.hawRHBIG = args.dble("HawRHBIG");
  alvWedge.hawRhsml = args.dble("HawRhsml");
  alvWedge.hawRCutY = args.dble("HawRCutY");
  alvWedge.hawRCutZ = args.dble("HawRCutZ");
  alvWedge.hawRCutDelY = args.dble("HawRCutDelY");
  alvWedge.hawYOffCry = args.dble("HawYOffCry");

  alvWedge.nFawPerSupm = args.integer("NFawPerSupm");
  alvWedge.fawPhiOff = args.dble("FawPhiOff");
  alvWedge.fawDelPhi = args.dble("FawDelPhi");
  alvWedge.fawPhiRot = args.dble("FawPhiRot");
  alvWedge.fawRadOff = args.dble("FawRadOff");

  Grid grid;
  grid.here = args.dble("GridHere");
  grid.name = myns + args.str("GridName");
  grid.mat = args.str("GridMat");
  grid.thick = args.dble("GridThick");

  Back back;
  back.here = args.dble("BackHere");
  back.xOff = args.dble("BackXOff");
  back.yOff = args.dble("BackYOff");
  back.sideName = myns + args.str("BackSideName");
  back.sideHere = args.dble("BackSideHere");
  back.sideLength = args.dble("BackSideLength");
  back.sideHeight = args.dble("BackSideHeight");
  back.sideWidth = args.dble("BackSideWidth");
  back.sideYOff1 = args.dble("BackSideYOff1");
  back.sideYOff2 = args.dble("BackSideYOff2");
  back.sideAngle = args.dble("BackSideAngle");
  back.sideMat = args.str("BackSideMat");
  back.plateName = myns + args.str("BackPlateName");
  back.plateHere = args.dble("BackPlateHere");
  back.plateLength = args.dble("BackPlateLength");
  back.plateThick = args.dble("BackPlateThick");
  back.plateWidth = args.dble("BackPlateWidth");
  back.plateMat = args.str("BackPlateMat");
  back.plate2Name = myns + args.str("BackPlate2Name");
  back.plate2Thick = args.dble("BackPlate2Thick");
  back.plate2Mat = args.str("BackPlate2Mat");

  Grille grille;
  grille.name = myns + args.str("GrilleName");
  grille.here = args.dble("GrilleHere");
  grille.thick = args.dble("GrilleThick");
  grille.width = args.dble("GrilleWidth");
  grille.zSpace = args.dble("GrilleZSpace");
  grille.mat = args.str("GrilleMat");
  grille.vecHeight = args.vecDble("GrilleHeight");
  grille.vecZOff = args.vecDble("GrilleZOff");

  grille.edgeSlotName = myns + args.str("GrEdgeSlotName");
  grille.edgeSlotMat = args.str("GrEdgeSlotMat");
  grille.edgeSlotHere = args.dble("GrEdgeSlotHere");
  grille.edgeSlotHeight = args.dble("GrEdgeSlotHeight");
  grille.edgeSlotWidth = args.dble("GrEdgeSlotWidth");
  grille.midSlotName = myns + args.str("GrMidSlotName");
  grille.midSlotMat = args.str("GrMidSlotMat");
  grille.midSlotHere = args.dble("GrMidSlotHere");
  grille.midSlotWidth = args.dble("GrMidSlotWidth");
  grille.midSlotXOff = args.dble("GrMidSlotXOff");
  grille.vecMidSlotHeight = args.vecDble("GrMidSlotHeight");

  BackPipe backPipe;
  backPipe.here = args.dble("BackPipeHere");
  backPipe.name = myns + args.str("BackPipeName");
  backPipe.vecDiam = args.vecDble("BackPipeDiam");
  backPipe.vecThick = args.vecDble("BackPipeThick");
  backPipe.mat = args.str("BackPipeMat");
  backPipe.waterMat = args.str("BackPipeWaterMat");

  BackCooling backCool;
  backCool.here = args.dble("BackCoolHere");
  backCool.vecName = args.vecStr("BackCoolName");
  std::transform(backCool.vecName.begin(),
                 backCool.vecName.end(),
                 backCool.vecName.begin(),
                 [&myns](std::string name) -> std::string { return (myns + name); });
  backCool.barHere = args.dble("BackCoolBarHere");
  backCool.barWidth = args.dble("BackCoolBarWidth");
  backCool.barHeight = args.dble("BackCoolBarHeight");
  backCool.mat = args.str("BackCoolMat");
  backCool.barName = myns + args.str("BackCoolBarName");
  backCool.barThick = args.dble("BackCoolBarThick");
  backCool.barMat = args.str("BackCoolBarMat");
  backCool.barSSName = myns + args.str("BackCoolBarSSName");
  backCool.barSSThick = args.dble("BackCoolBarSSThick");
  backCool.barSSMat = args.str("BackCoolBarSSMat");
  backCool.barWaName = myns + args.str("BackCoolBarWaName");
  backCool.barWaThick = args.dble("BackCoolBarWaThick");
  backCool.barWaMat = args.str("BackCoolBarWaMat");
  backCool.vFEHere = args.dble("BackCoolVFEHere");
  backCool.vFEName = myns + args.str("BackCoolVFEName");
  backCool.vFEMat = args.str("BackCoolVFEMat");
  backCool.backVFEName = args.str("BackVFEName");
  backCool.backVFEMat = args.str("BackVFEMat");
  backCool.vecBackVFELyrThick = args.vecDble("BackVFELyrThick");
  backCool.vecBackVFELyrName = args.vecStr("BackVFELyrName");
  std::transform(backCool.vecBackVFELyrName.begin(),
                 backCool.vecBackVFELyrName.end(),
                 backCool.vecBackVFELyrName.begin(),
                 [&myns](std::string name) -> std::string { return (myns + name); });
  backCool.vecBackVFELyrMat = args.vecStr("BackVFELyrMat");
  backCool.vecBackCoolNSec = args.vecDble("BackCoolNSec");
  backCool.vecBackCoolSecSep = args.vecDble("BackCoolSecSep");
  backCool.vecBackCoolNPerSec = args.vecDble("BackCoolNPerSec");

  BackMisc backMisc;
  backMisc.here = args.dble("BackMiscHere");
  backMisc.vecThick = args.vecDble("BackMiscThick");
  backMisc.vecName = args.vecStr("BackMiscName");
  std::transform(backMisc.vecName.begin(),
                 backMisc.vecName.end(),
                 backMisc.vecName.begin(),
                 [&myns](std::string name) -> std::string { return (myns + name); });
  backMisc.vecMat = args.vecStr("BackMiscMat");
  backMisc.backCBStdSep = args.dble("BackCBStdSep");

  PatchPanel patchPanel;
  patchPanel.here = args.dble("PatchPanelHere");
  patchPanel.vecThick = args.vecDble("PatchPanelThick");
  patchPanel.vecNames = args.vecStr("PatchPanelNames");
  std::transform(patchPanel.vecNames.begin(),
                 patchPanel.vecNames.end(),
                 patchPanel.vecNames.begin(),
                 [&myns](std::string name) -> std::string { return (myns + name); });

  patchPanel.vecMat = args.vecStr("PatchPanelMat");
  patchPanel.name = myns + args.str("PatchPanelName");

  BackCoolTank backCoolTank;
  backCoolTank.here = args.dble("BackCoolTankHere");
  backCoolTank.name = myns + args.str("BackCoolTankName");
  backCoolTank.width = args.dble("BackCoolTankWidth");
  backCoolTank.thick = args.dble("BackCoolTankThick");
  backCoolTank.mat = args.str("BackCoolTankMat");
  backCoolTank.waName = myns + args.str("BackCoolTankWaName");
  backCoolTank.waWidth = args.dble("BackCoolTankWaWidth");
  backCoolTank.waMat = args.str("BackCoolTankWaMat");
  backCoolTank.backBracketName = myns + args.str("BackBracketName");
  backCoolTank.backBracketHeight = args.dble("BackBracketHeight");
  backCoolTank.backBracketMat = args.str("BackBracketMat");

  DryAirTube dryAirTube;
  dryAirTube.here = args.dble("DryAirTubeHere");
  dryAirTube.name = args.str("DryAirTubeName");
  dryAirTube.mbCoolTubeNum = args.integer("MBCoolTubeNum");
  dryAirTube.innDiam = args.dble("DryAirTubeInnDiam");
  dryAirTube.outDiam = args.dble("DryAirTubeOutDiam");
  dryAirTube.mat = args.str("DryAirTubeMat");

  MBCoolTube mbCoolTube;
  mbCoolTube.here = args.dble("MBCoolTubeHere");
  mbCoolTube.name = myns + args.str("MBCoolTubeName");
  mbCoolTube.innDiam = args.dble("MBCoolTubeInnDiam");
  mbCoolTube.outDiam = args.dble("MBCoolTubeOutDiam");
  mbCoolTube.mat = args.str("MBCoolTubeMat");

  MBManif mbManif;
  mbManif.here = args.dble("MBManifHere");
  mbManif.name = myns + args.str("MBManifName");
  mbManif.innDiam = args.dble("MBManifInnDiam");
  mbManif.outDiam = args.dble("MBManifOutDiam");
  mbManif.mat = args.str("MBManifMat");

  MBLyr mbLyr;
  mbLyr.here = args.dble("MBLyrHere");
  mbLyr.vecMBLyrThick = args.vecDble("MBLyrThick");
  mbLyr.vecMBLyrName = args.vecStr("MBLyrName");
  std::transform(mbLyr.vecMBLyrName.begin(),
                 mbLyr.vecMBLyrName.end(),
                 mbLyr.vecMBLyrName.begin(),
                 [&myns](std::string name) -> std::string { return (myns + name); });
  mbLyr.vecMBLyrMat = args.vecStr("MBLyrMat");

  Pincer pincer;
  pincer.rodHere = args.dble("PincerRodHere");
  pincer.rodName = myns + args.str("PincerRodName");
  pincer.rodMat = args.str("PincerRodMat");
  pincer.vecRodAzimuth = args.vecDble("PincerRodAzimuth");
  pincer.envName = myns + args.str("PincerEnvName");
  pincer.envMat = args.str("PincerEnvMat");
  pincer.envWidthHalf = 0.5 * args.dble("PincerEnvWidth");
  pincer.envHeightHalf = 0.5 * args.dble("PincerEnvHeight");
  pincer.envLengthHalf = 0.5 * args.dble("PincerEnvLength");
  pincer.vecEnvZOff = args.vecDble("PincerEnvZOff");
  pincer.blkName = myns + args.str("PincerBlkName");
  pincer.blkMat = args.str("PincerBlkMat");
  pincer.blkLengthHalf = 0.5 * args.dble("PincerBlkLength");
  pincer.shim1Name = myns + args.str("PincerShim1Name");
  pincer.shimHeight = args.dble("PincerShimHeight");
  pincer.shim2Name = myns + args.str("PincerShim2Name");
  pincer.shimMat = args.str("PincerShimMat");
  pincer.shim1Width = args.dble("PincerShim1Width");
  pincer.shim2Width = args.dble("PincerShim2Width");
  pincer.cutName = myns + args.str("PincerCutName");
  pincer.cutMat = args.str("PincerCutMat");
  pincer.cutWidth = args.dble("PincerCutWidth");
  pincer.cutHeight = args.dble("PincerCutHeight");

  Volume parentVolume = ns.volume(args.parentName());
  if (bar.here != 0) {
    const unsigned int copyOne(1);
    const unsigned int copyTwo(2);
    // Barrel parent volume----------------------------------------------------------
    Solid barSolid = Polycone(bar.name, bar.phiLo, (bar.phiHi - bar.phiLo), bar.vecRMin, bar.vecRMax, bar.vecZPts);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << bar.name << " PolyCone from " << convertRadToDeg(bar.phiLo) << " to "
                               << convertRadToDeg(bar.phiHi) << " with " << bar.vecZPts.size() << " points";
    for (unsigned int k = 0; k < bar.vecZPts.size(); ++k)
      edm::LogVerbatim("EBGeom") << "[" << k << "] " << cms::convert2mm(bar.vecZPts[k]) << ":"
                                 << cms::convert2mm(bar.vecRMin[k]) << ":" << cms::convert2mm(bar.vecRMax[k]);
#endif
    Position tran(bar.vecTran[0], bar.vecTran[1], bar.vecTran[2]);
    Rotation3D rotation = myrot(ns,
                                bar.name + "Rot",
                                Rota(Vec3(bar.vecRota3[0], bar.vecRota3[1], bar.vecRota3[2]), bar.vecRota3[3]) *
                                    Rota(Vec3(bar.vecRota2[0], bar.vecRota2[1], bar.vecRota2[2]), bar.vecRota2[3]) *
                                    Rota(Vec3(bar.vecRota[0], bar.vecRota[1], bar.vecRota[2]), bar.vecRota[3]));
    Volume barVolume = Volume(bar.name, barSolid, ns.material(bar.mat));
    parentVolume.placeVolume(barVolume, copyOne, Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeomX") << barVolume.name() << ":" << copyOne << " positioned in " << parentVolume.name()
                                << " at (" << cms::convert2mm(tran.x()) << "," << cms::convert2mm(tran.y()) << ","
                                << cms::convert2mm(tran.z()) << ") with rotation";
#endif
    // End Barrel parent volume----------------------------------------------------------

    // Supermodule parent------------------------------------------------------------

    const string spmcut1ddname((0 != spm.cutShow) ? spm.name : (spm.name + "CUT1"));

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << spmcut1ddname << " PolyCone from " << convertRadToDeg(spm.lowPhi) << " to "
                               << convertRadToDeg(spm.lowPhi + spm.delPhi) << " with " << spm.vecZPts.size()
                               << " points";
    for (unsigned int k = 0; k < spm.vecZPts.size(); ++k)
      edm::LogVerbatim("EBGeom") << "[" << k << "] " << cms::convert2mm(spm.vecZPts[k]) << ":"
                                 << cms::convert2mm(spm.vecRMin[k]) << ":" << cms::convert2mm(spm.vecRMax[k]);
#endif

    const unsigned int indx(0.5 * spm.vecRMax.size());

    // Deal with the cut boxes first
    array<double, 3> cutBoxParms{{1.05 * (spm.vecRMax[indx] - spm.vecRMin[indx]) * 0.5,
                                  0.5 * spm.cutThick,
                                  fabs(spm.vecZPts.back() - spm.vecZPts.front()) * 0.5 + 1.0 * dd4hep::mm}};

    // Supermodule side platess
    array<double, 3> sideParms{{0.5 * spm.sideHigh, 0.5 * spm.sideThick, 0.5 * fabs(spm.vecZPts[1] - spm.vecZPts[0])}};
    Solid sideSolid = Box(spm.sideName, sideParms[0], sideParms[1], sideParms[2]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << spm.sideName << " Box " << cms::convert2mm(sideParms[0]) << ":"
                               << cms::convert2mm(sideParms[1]) << ":" << cms::convert2mm(sideParms[2]);
#endif
    Volume sideLog = Volume(spm.sideName, sideSolid, ns.material(spm.sideMat));

    Position sideddtra1;
    Position sideddtra2;
    for (unsigned int icopy(1); icopy <= 2; ++icopy) {
      const std::vector<double>& tvec(1 == icopy ? spm.vecCutTM : spm.vecCutTP);
      double rang(1 == icopy ? spm.cutRM : spm.cutRP);

      const Position tr(tvec[0], tvec[1], tvec[2]);
      RotationZ ro(rang);
      const double ang(1 == icopy ? spm.lowPhi : spm.lowPhi + spm.delPhi);
      RotationZ ro1(ang);
      const Position tr1(
          0.5 * (spm.vecRMax[indx] + spm.vecRMin[indx]), 0, 0.5 * (spm.vecZPts.front() + spm.vecZPts.back()));

      const Tl3D trSide(tvec[0],
                        tvec[1] + (1 == icopy ? 1. : -1.) * (cutBoxParms[1] + sideParms[1]) +
                            (1 == icopy ? spm.sideYOffM : spm.sideYOffP),
                        tvec[2]);
      const RoZ3D roSide(rang);
      const Tf3D sideRot(RoZ3D(1 == icopy ? spm.lowPhi : spm.lowPhi + spm.delPhi) *
                         Tl3D(spm.vecRMin.front() + sideParms[0], 0, spm.vecZPts.front() + sideParms[2]) * trSide *
                         roSide);

      Rotation3D sideddrot(myrot(ns, spm.sideName + std::to_string(icopy), sideRot.getRotation()));
      const Position sideddtra(sideRot.getTranslation());
      1 == icopy ? sideddtra1 = sideddtra : sideddtra2 = sideddtra;
    }

    ns.addAssembly(spm.name);
    ns.assembly(spm.name).placeVolume(
        sideLog, 1, Transform3D(ns.rotation(spm.sideName + std::to_string(1)), sideddtra1));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeomX") << sideLog.name() << ":1 positioned in " << spm.name << " at ("
                                << cms::convert2mm(sideddtra1.x()) << "," << cms::convert2mm(sideddtra1.y()) << ","
                                << cms::convert2mm(sideddtra1.z()) << ") with rotation";
#endif
    ns.assembly(spm.name).placeVolume(
        sideLog, 2, Transform3D(ns.rotation(spm.sideName + std::to_string(2)), sideddtra2));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeomX") << sideLog.name() << ":2 positioned in " << ns.assembly(spm.name).name() << " at ("
                                << cms::convert2mm(sideddtra2.x()) << "," << cms::convert2mm(sideddtra2.y()) << ","
                                << cms::convert2mm(sideddtra2.z()) << ") with rotation";
#endif

    const double dphi(360._deg / (1. * spm.nPerHalf));
    for (unsigned int iphi(0); iphi < 2 * spm.nPerHalf; ++iphi) {
      const double phi(iphi * dphi + spm.phiOff);

      // this base rotation includes the base translation & rotation
      // plus flipping for the negative z hemisphere, plus
      // the phi rotation for this module
      const Tf3D rotaBase(RoZ3D(phi) * (iphi < spm.nPerHalf ? Ro3D() : RoX3D(180._deg)) *
                          Ro3D(spm.vecBRota[3], Vec3(spm.vecBRota[0], spm.vecBRota[1], spm.vecBRota[2])) *
                          Tl3D(Vec3(spm.vecBTran[0], spm.vecBTran[1], spm.vecBTran[2])));

      // here the individual rotations & translations of the supermodule
      // are implemented on top of the overall "base" rotation & translation

      const unsigned int offr(4 * iphi);
      const unsigned int offt(3 * iphi);

      const Ro3D r1(spm.vecRota[offr + 3], Vec3(spm.vecRota[offr + 0], spm.vecRota[offr + 1], spm.vecRota[offr + 2]));

      const Tf3D rotaExtra(r1 * Tl3D(Vec3(spm.vecTran[offt + 0], spm.vecTran[offt + 1], spm.vecTran[offt + 2])));

      const Tf3D both(rotaExtra * rotaBase);

      const Rotation3D rota(myrot(ns, spm.name + std::to_string(convertRadToDeg(phi)), both.getRotation()));

      if (spm.vecHere[iphi] != 0) {
        // convert from CLHEP to Position & etc.
        Position myTran(both.getTranslation().x(), both.getTranslation().y(), both.getTranslation().z());
        barVolume.placeVolume(ns.assembly(spm.name), iphi + 1, Transform3D(rota, myTran));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << ns.assembly(spm.name).name() << ":" << (iphi + 1) << " positioned in "
                                    << barVolume.name() << " at (" << cms::convert2mm(myTran.x()) << ","
                                    << cms::convert2mm(myTran.y()) << "," << cms::convert2mm(myTran.z())
                                    << ") with rotation";
#endif
      }
    }
    // End Supermodule parent------------------------------------------------------------

    // Begin Inner Layer volumes---------------------------------------------------------
    const double ilyLengthHalf(0.5 * (spm.vecZPts[1] - spm.vecZPts[0]));
    double ilyRMin(spm.vecRMin[0]);
    double ilyThick(0);
    for (unsigned int ilyx(0); ilyx != ily.vecIlyThick.size(); ++ilyx) {
      ilyThick += ily.vecIlyThick[ilyx];
    }
    Solid ilySolid = Tube(ily.name,
                          ilyRMin,                   // rmin
                          ilyRMin + ilyThick,        // rmax
                          ilyLengthHalf,             // dz
                          ily.phiLow,                // startPhi
                          ily.phiLow + ily.delPhi);  // startPhi + deltaPhi
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << ily.name << " Tubs " << cms::convert2mm(ilyLengthHalf) << ":"
                               << cms::convert2mm(ilyRMin) << ":" << cms::convert2mm(ilyRMin + ilyThick) << ":"
                               << convertRadToDeg(ily.phiLow) << ":" << convertRadToDeg(ily.delPhi);
#endif
    Volume ilyLog = Volume(ily.name, ilySolid, ns.material(spm.mat));
    ns.assembly(spm.name).placeVolume(ilyLog, copyOne, Position(0, 0, ilyLengthHalf));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeomX") << ilyLog.name() << ":" << copyOne << " positioned in " << ns.assembly(spm.name).name()
                                << " at (0,0," << cms::convert2mm(ilyLengthHalf) << ") with no rotation";
#endif
    Volume ilyPipeLog[200];
    if (0 != ily.pipeHere) {
      for (unsigned int iPipeType(0); iPipeType != ily.vecIlyPipeLengthHalf.size(); ++iPipeType) {
        string pName(ily.pipeName + "_" + std::to_string(iPipeType + 1));

        Solid ilyPipeSolid = Tube(pName,
                                  0,                                    // rmin
                                  ily.pipeODHalf,                       // rmax
                                  ily.vecIlyPipeLengthHalf[iPipeType],  // dz
                                  0._deg,                               // startPhi
                                  360._deg);                            // startPhi + deltaPhi
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << pName << " Tubs " << cms::convert2mm(ily.vecIlyPipeLengthHalf[iPipeType])
                                   << ":0:" << cms::convert2mm(ily.pipeODHalf) << ":0:360";
#endif
        ilyPipeLog[iPipeType] = Volume(pName, ilyPipeSolid, ns.material(ily.pipeMat));

        string pWaName(ily.pipeName + "Wa_" + std::to_string(iPipeType + 1));
        Solid ilyPipeWaSolid = Tube(pWaName,
                                    0,                                    // rmin
                                    0.5 * ily.pipeID,                     // rmax
                                    ily.vecIlyPipeLengthHalf[iPipeType],  // dz
                                    0._deg,                               // startPhi
                                    360._deg);                            // startPhi + deltaPhi
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << pWaName << " Tubs " << cms::convert2mm(ily.vecIlyPipeLengthHalf[iPipeType])
                                   << ":0:" << cms::convert2mm(0.5 * ily.pipeID) << ":0:360";
#endif
        Volume ilyPipeWaLog = Volume(pWaName, ilyPipeWaSolid, ns.material(backPipe.waterMat));
        ilyPipeLog[iPipeType].placeVolume(ilyPipeWaLog, copyOne);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << ilyPipeWaLog.name() << ":" << copyOne << " positioned in "
                                    << ilyPipeLog[iPipeType].name() << " at (0,0,0) with no rotation";
#endif
      }
    }

    Solid ilyPTMSolid = Box(ily.pTMName, ily.pTMHeightHalf, ily.pTMWidthHalf, ily.pTMLengthHalf);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << ily.pTMName << " Box " << cms::convert2mm(ily.pTMHeightHalf) << ":"
                               << cms::convert2mm(ily.pTMWidthHalf) << ":" << cms::convert2mm(ily.pTMLengthHalf);
#endif
    Volume ilyPTMLog = Volume(ily.pTMName, ilyPTMSolid, ns.material(ily.pTMMat));

    Solid ilyFanOutSolid = Box(ily.fanOutName, ily.fanOutHeightHalf, ily.fanOutWidthHalf, ily.fanOutLengthHalf);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << ily.fanOutName << " Box " << cms::convert2mm(ily.fanOutHeightHalf) << ":"
                               << cms::convert2mm(ily.fanOutWidthHalf) << ":" << cms::convert2mm(ily.fanOutLengthHalf);
#endif
    Volume ilyFanOutLog = Volume(ily.fanOutName, ilyFanOutSolid, ns.material(ily.fanOutMat));

    Solid ilyFEMSolid = Box(ily.fEMName, ily.fEMHeightHalf, ily.fEMWidthHalf, ily.fEMLengthHalf);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << ily.fEMName << " Box " << cms::convert2mm(ily.fEMHeightHalf) << ":"
                               << cms::convert2mm(ily.fEMWidthHalf) << ":" << cms::convert2mm(ily.fEMLengthHalf);
#endif
    Volume ilyFEMLog = Volume(ily.fEMName, ilyFEMSolid, ns.material(ily.fEMMat));

    Solid ilyDiffSolid = Box(ily.diffName, ily.fanOutHeightHalf, ily.fanOutWidthHalf, ily.diffLengthHalf);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << ily.diffName << " Box " << cms::convert2mm(ily.fanOutHeightHalf) << ":"
                               << cms::convert2mm(ily.fanOutWidthHalf) << ":" << cms::convert2mm(ily.diffLengthHalf);
#endif
    Volume ilyDiffLog = Volume(ily.diffName, ilyDiffSolid, ns.material(ily.diffMat));

    Solid ilyBndlSolid = Box(ily.bndlName, ily.fanOutHeightHalf, ily.fanOutWidthHalf, ily.bndlLengthHalf);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << ily.bndlName << " Box " << cms::convert2mm(ily.fanOutHeightHalf) << ":"
                               << cms::convert2mm(ily.fanOutWidthHalf) << ":" << cms::convert2mm(ily.bndlLengthHalf);
#endif
    Volume ilyBndlLog = Volume(ily.bndlName, ilyBndlSolid, ns.material(ily.bndlMat));

    ilyFanOutLog.placeVolume(
        ilyDiffLog, copyOne, Position(0, 0, -ily.fanOutLengthHalf + ily.diffLengthHalf + ily.diffOff));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeomX") << ilyDiffLog.name() << ":" << copyOne << " positioned in " << ilyFanOutLog.name()
                                << " at (0,0,"
                                << cms::convert2mm(-ily.fanOutLengthHalf + ily.diffLengthHalf + ily.diffOff)
                                << ") with no rotation";
#endif
    ilyFanOutLog.placeVolume(
        ilyBndlLog, copyOne, Position(0, 0, -ily.fanOutLengthHalf + ily.bndlLengthHalf + ily.bndlOff));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeomX") << ilyBndlLog.name() << ":" << copyOne << " positioned in " << ilyFanOutLog.name()
                                << " at (0,0,"
                                << cms::convert2mm(-ily.fanOutLengthHalf + ily.bndlLengthHalf + ily.bndlOff)
                                << ") with no rotation";
#endif

    Volume xilyLog;
    for (unsigned int iily(0); iily != ily.vecIlyThick.size(); ++iily) {
      const double ilyRMax(ilyRMin + ily.vecIlyThick[iily]);
      string xilyName(ily.name + std::to_string(iily));
      Solid xilySolid = Tube(xilyName, ilyRMin, ilyRMax, ilyLengthHalf, ily.phiLow, ily.phiLow + ily.delPhi);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << xilyName << " Tubs " << cms::convert2mm(ilyLengthHalf) << ":"
                                 << cms::convert2mm(ilyRMin) << ":" << cms::convert2mm(ilyRMax) << ":"
                                 << convertRadToDeg(ily.phiLow) << ":" << convertRadToDeg(ily.delPhi);
#endif
      xilyLog = ns.addVolume(Volume(xilyName, xilySolid, ns.material(ily.vecIlyMat[iily])));
      if (0 != ily.here) {
        ilyLog.placeVolume(xilyLog, copyOne);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << xilyLog.name() << ":" << copyOne << " positioned in " << ilyLog.name()
                                    << " at (0,0,0) with no rotation";
#endif
        unsigned int copyNum[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        if (10 * dd4hep::mm < ily.vecIlyThick[iily] && ily.vecIlyThick.size() != (iily + 1) && 0 != ily.pipeHere) {
          if (0 != ily.pTMHere) {
            unsigned int ptmCopy(0);
            for (unsigned int ilyPTM(0); ilyPTM != ily.vecIlyPTMZ.size(); ++ilyPTM) {
              const double radius(ilyRMax - 1 * dd4hep::mm - ily.pTMHeightHalf);
              const double phi(ily.vecIlyPTMPhi[ilyPTM]);
              const double yy(radius * sin(phi));
              const double xx(radius * cos(phi));
              ++ptmCopy;
              xilyLog.placeVolume(
                  ilyPTMLog,
                  ptmCopy,
                  Transform3D(RotationZ(phi), Position(xx, yy, ily.vecIlyPTMZ[ilyPTM] - ilyLengthHalf)));
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("EBGeomX")
                  << ilyPTMLog.name() << ":" << ptmCopy << " positioned in " << xilyLog.name() << " at ("
                  << cms::convert2mm(xx) << "," << cms::convert2mm(yy) << ","
                  << cms::convert2mm(ily.vecIlyPTMZ[ilyPTM] - ilyLengthHalf) << ") with rotation";
#endif
            }
          }
          if (0 != ily.fanOutHere) {
            unsigned int fanOutCopy(0);
            for (unsigned int ilyFO(0); ilyFO != ily.vecIlyFanOutZ.size(); ++ilyFO) {
              const double radius(ilyRMax - 1 * dd4hep::mm - ily.fanOutHeightHalf);
              const double phi(ily.vecIlyFanOutPhi[ilyFO]);
              const double yy(radius * sin(phi));
              const double xx(radius * cos(phi));
              ++fanOutCopy;
              xilyLog.placeVolume(ilyFanOutLog,
                                  fanOutCopy,
                                  Transform3D(RotationZ(phi) * RotationY(180._deg),
                                              Position(xx, yy, ily.vecIlyFanOutZ[ilyFO] - ilyLengthHalf)));
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("EBGeomX")
                  << ilyFanOutLog.name() << ":" << fanOutCopy << " positioned in " << xilyLog.name() << " at ("
                  << cms::convert2mm(xx) << "," << cms::convert2mm(yy) << ","
                  << cms::convert2mm(ily.vecIlyFanOutZ[ilyFO] - ilyLengthHalf) << ") with rotation";
#endif
            }
            unsigned int femCopy(0);
            for (unsigned int ilyFEM(0); ilyFEM != ily.vecIlyFEMZ.size(); ++ilyFEM) {
              const double radius(ilyRMax - 1 * dd4hep::mm - ily.fEMHeightHalf);
              const double phi(ily.vecIlyFEMPhi[ilyFEM]);
              const double yy(radius * sin(phi));
              const double xx(radius * cos(phi));
              ++femCopy;
              xilyLog.placeVolume(
                  ilyFEMLog,
                  femCopy,
                  Transform3D(RotationZ(phi), Position(xx, yy, ily.vecIlyFEMZ[ilyFEM] - ilyLengthHalf)));
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("EBGeomX")
                  << ilyFEMLog.name() << ":" << femCopy << " positioned in " << xilyLog.name() << " at ("
                  << cms::convert2mm(xx) << "," << cms::convert2mm(yy) << ","
                  << cms::convert2mm(ily.vecIlyFEMZ[ilyFEM] - ilyLengthHalf) << ") with rotation";
#endif
            }
          }
          for (unsigned int iPipe(0); iPipe != ily.vecIlyPipePhi.size(); ++iPipe) {
            const unsigned int type(static_cast<unsigned int>(round(ily.vecIlyPipeType[iPipe])));
            const double zz(-ilyLengthHalf + ily.vecIlyPipeZ[iPipe] + (9 > type ? ily.vecIlyPipeLengthHalf[type] : 0));

            for (unsigned int ly(0); ly != 2; ++ly) {
              const double radius(0 == ly ? ilyRMin + ily.pipeODHalf + 1 * dd4hep::mm
                                          : ilyRMax - ily.pipeODHalf - 1 * dd4hep::mm);
              const double phi(ily.vecIlyPipePhi[iPipe]);
              const double yy(radius * sin(phi));
              const double xx(radius * cos(phi));
              ++copyNum[type];
              if (9 > type) {
                xilyLog.placeVolume(ilyPipeLog[type], copyNum[type], Position(xx, yy, zz));
#ifdef EDM_ML_DEBUG
                edm::LogVerbatim("EBGeomX")
                    << ilyPipeLog[type].name() << ":" << copyNum[type] << " positioned in " << xilyLog.name() << " at ("
                    << cms::convert2mm(xx) << "," << cms::convert2mm(yy) << "," << cms::convert2mm(zz)
                    << ") with no rotation";
#endif
              } else {
                xilyLog.placeVolume(
                    ilyPipeLog[type],
                    copyNum[type],
                    Transform3D(Rotation3D(ROOT::Math::AxisAngle(ROOT::Math::AxisAngle::XYZVector(xx, yy, 0), 90._deg)),
                                Position(xx, yy, zz)));
#ifdef EDM_ML_DEBUG
                edm::LogVerbatim("EBGeomX") << ilyPipeLog[type].name() << ":" << copyNum[type] << " positioned in "
                                            << xilyLog.name() << " at (" << cms::convert2mm(xx) << ","
                                            << cms::convert2mm(yy) << "," << cms::convert2mm(zz) << ") with rotation";
#endif
              }
            }
          }
        }
      }
      ilyRMin = ilyRMax;
    }
    // End Inner Layer volumes---------------------------------------------------------

    string clyrName(ns.prepend("ECLYR"));
    std::vector<double> cri;
    std::vector<double> cro;
    std::vector<double> czz;
    czz.emplace_back(spm.vecZPts[1]);
    cri.emplace_back(spm.vecRMin[0]);
    cro.emplace_back(spm.vecRMin[0] + 25 * dd4hep::mm);
    czz.emplace_back(spm.vecZPts[2]);
    cri.emplace_back(spm.vecRMin[2]);
    cro.emplace_back(spm.vecRMin[2] + 10 * dd4hep::mm);
    Solid clyrSolid = Polycone(clyrName, -9.5_deg, 19_deg, cri, cro, czz);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << clyrName << " PolyCone from -9.5 to 9.5 with " << czz.size() << " points";
    for (unsigned int k = 0; k < czz.size(); ++k)
      edm::LogVerbatim("EBGeom") << "[" << k << "] " << cms::convert2mm(czz[k]) << ":" << cms::convert2mm(cri[k]) << ":"
                                 << cms::convert2mm(cro[k]);
#endif
    Volume clyrLog = Volume(clyrName, clyrSolid, ns.material(ily.vecIlyMat[4]));
    ns.assembly(spm.name).placeVolume(clyrLog, copyOne);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeomX") << clyrLog.name() << ":" << copyOne << " positioned in " << ns.assembly(spm.name).name()
                                << " at (0,0,0) with no rotation";
#endif
    // Begin Alveolar Wedge parent ------------------------------------------------------
    //----------------

    // the next few lines accumulate dimensions appropriate to crystal type 1
    // which we use to set some of the features of the half-alveolar wedge (hawR).

    const double BNom1(cry.vecNomCryDimCR[0]);
    const double bNom1(cry.vecNomCryDimCF[0]);
    const double sWall1(alv.wallThAlv);
    const double fWall1(alv.wallFrAlv);
    const double sWrap1(alv.wrapThAlv);
    const double fWrap1(alv.wrapFrAlv);
    const double sClr1(alv.clrThAlv);
    const double fClr1(alv.clrFrAlv);
    const double LNom1(cry.nomCryDimLZ);
    const double beta1(atan((BNom1 - bNom1) / LNom1));
    const double sinbeta1(sin(beta1));

    const double tana_hawR((BNom1 - bNom1) / LNom1);

    const double H_hawR(alvWedge.hawRHBIG);
    const double h_hawR(alvWedge.hawRhsml);
    const double a_hawR(bNom1 + sClr1 + 2 * sWrap1 + 2 * sWall1 - sinbeta1 * (fClr1 + fWrap1 + fWall1));
    const double B_hawR(a_hawR + H_hawR * tana_hawR);
    const double b_hawR(a_hawR + h_hawR * tana_hawR);
    const double L_hawR(spm.vecZPts[2]);

    const EcalTrap trapHAWR(0.5 * a_hawR,  //double aHalfLengthXNegZLoY , // bl1, A/2
                            0.5 * a_hawR,  //double aHalfLengthXPosZLoY , // bl2, a/2
                            0.5 * b_hawR,  //double aHalfLengthXPosZHiY , // tl2, b/2
                            0.5 * H_hawR,  //double aHalfLengthYNegZ    , // h1, H/2
                            0.5 * h_hawR,  //double aHalfLengthYPosZ    , // h2, h/2
                            0.5 * L_hawR,  //double aHalfLengthZ        , // dz,  L/2
                            90._deg,       //double aAngleAD            , // alfa1
                            0,             //double aCoord15X           , // x15
                            0              //double aCoord15Y             // y15
    );

    string hawRName1(alvWedge.hawRName + "1");

    const double al1_fawR(atan((B_hawR - a_hawR) / H_hawR) + M_PI_2);

    // here is trap for Full Alveolar Wedge
    const EcalTrap trapFAW(a_hawR,        //double aHalfLengthXNegZLoY , // bl1, A/2
                           a_hawR,        //double aHalfLengthXPosZLoY , // bl2, a/2
                           b_hawR,        //double aHalfLengthXPosZHiY , // tl2, b/2
                           0.5 * H_hawR,  //double aHalfLengthYNegZ    , // h1, H/2
                           0.5 * h_hawR,  //double aHalfLengthYPosZ    , // h2, h/2
                           0.5 * L_hawR,  //double aHalfLengthZ        , // dz,  L/2
                           al1_fawR,      //double aAngleAD            , // alfa1
                           0,             //double aCoord15X           , // x15
                           0              //double aCoord15Y             // y15
    );

    const string fawName1(alvWedge.fawName + "1");
    Solid fawSolid1 = mytrap(fawName1, trapFAW);

    const EcalTrap::VertexList vHAW(trapHAWR.vertexList());
    const EcalTrap::VertexList vFAW(trapFAW.vertexList());

    const double hawBoxClr(1 * dd4hep::mm);

    // HAW cut box to cut off back end of wedge
    const string hawCutName(alvWedge.hawRName + "CUTBOX");
    array<double, 3> hawBoxParms{{0.5 * b_hawR + hawBoxClr, 0.5 * alvWedge.hawRCutY, 0.5 * alvWedge.hawRCutZ}};

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << hawCutName << " Box " << cms::convert2mm(hawBoxParms[0]) << ":"
                               << cms::convert2mm(hawBoxParms[1]) << ":" << cms::convert2mm(hawBoxParms[2]);
#endif

    const Pt3D b1(hawBoxParms[0], hawBoxParms[1], hawBoxParms[2]);
    const Pt3D b2(-hawBoxParms[0], hawBoxParms[1], hawBoxParms[2]);
    const Pt3D b3(-hawBoxParms[0], hawBoxParms[1], -hawBoxParms[2]);

    const double zDel(
        sqrt(4 * hawBoxParms[2] * hawBoxParms[2] - (h_hawR - alvWedge.hawRCutDelY) * (h_hawR - alvWedge.hawRCutDelY)));

    const Tf3D hawCutForm(b1,
                          b2,
                          b3,
                          vHAW[2] + Pt3D(hawBoxClr, -alvWedge.hawRCutDelY, 0),
                          vHAW[1] + Pt3D(-hawBoxClr, -alvWedge.hawRCutDelY, 0),
                          Pt3D(vHAW[0].x() - hawBoxClr, vHAW[0].y(), vHAW[0].z() - zDel));

    ns.addAssembly(alvWedge.hawRName);

    // FAW cut box to cut off back end of wedge
    const string fawCutName(alvWedge.fawName + "CUTBOX");
    const array<double, 3> fawBoxParms{{2 * hawBoxParms[0], hawBoxParms[1], hawBoxParms[2]}};
    Solid fawCutBox = Box(fawCutName, fawBoxParms[0], fawBoxParms[1], fawBoxParms[2]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << fawCutName << " Box " << cms::convert2mm(fawBoxParms[0]) << ":"
                               << cms::convert2mm(fawBoxParms[1]) << ":" << cms::convert2mm(fawBoxParms[2]);
#endif

    const Pt3D bb1(fawBoxParms[0], fawBoxParms[1], fawBoxParms[2]);
    const Pt3D bb2(-fawBoxParms[0], fawBoxParms[1], fawBoxParms[2]);
    const Pt3D bb3(-fawBoxParms[0], fawBoxParms[1], -fawBoxParms[2]);

    const Tf3D fawCutForm(bb1,
                          bb2,
                          bb3,
                          vFAW[2] + Pt3D(2 * hawBoxClr, -5 * dd4hep::mm, 0),
                          vFAW[1] + Pt3D(-2 * hawBoxClr, -5 * dd4hep::mm, 0),
                          Pt3D(vFAW[1].x() - 2 * hawBoxClr, vFAW[1].y() - trapFAW.h(), vFAW[1].z() - zDel));

    Solid fawSolid = SubtractionSolid(fawSolid1,
                                      fawCutBox,
                                      Transform3D(myrot(ns, fawCutName + "R", fawCutForm.getRotation()),
                                                  Position(fawCutForm.getTranslation().x(),
                                                           fawCutForm.getTranslation().y(),
                                                           fawCutForm.getTranslation().z())));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeom") << alvWedge.fawName << " Subtraction " << fawSolid1.name() << ":" << fawCutBox.name()
                               << " at (" << cms::convert2mm(fawCutForm.getTranslation().x()) << ","
                               << cms::convert2mm(fawCutForm.getTranslation().y()) << ","
                               << cms::convert2mm(fawCutForm.getTranslation().z()) << ")";
#endif
    Volume fawLog = Volume(alvWedge.fawName, fawSolid, ns.material(spm.mat));

    const Tf3D hawRform(vHAW[3],
                        vHAW[0],
                        vHAW[1],  // HAW inside FAW
                        vFAW[3],
                        0.5 * (vFAW[0] + vFAW[3]),
                        0.5 * (vFAW[1] + vFAW[2]));

    fawLog.placeVolume(
        ns.assembly(alvWedge.hawRName),
        copyOne,
        Transform3D(
            myrot(ns, alvWedge.hawRName + "R", hawRform.getRotation()),
            Position(hawRform.getTranslation().x(), hawRform.getTranslation().y(), hawRform.getTranslation().z())));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeomX") << ns.assembly(alvWedge.hawRName).name() << ":" << copyOne << " positioned in "
                                << fawLog.name() << " at (" << cms::convert2mm(hawRform.getTranslation().x()) << ","
                                << cms::convert2mm(hawRform.getTranslation().y()) << ","
                                << cms::convert2mm(hawRform.getTranslation().z()) << ") with rotation";
#endif

    fawLog.placeVolume(
        ns.assembly(alvWedge.hawRName),
        copyTwo,
        Transform3D(
            Rotation3D(1., 0., 0., 0., 1., 0., 0., 0., -1.) * RotationY(M_PI),  // rotate about Y after refl thru Z
            Position(-hawRform.getTranslation().x(), -hawRform.getTranslation().y(), -hawRform.getTranslation().z())));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EBGeomX") << ns.assembly(alvWedge.hawRName).name() << ":" << copyTwo << " positioned in "
                                << fawLog.name() << " at (" << cms::convert2mm(-hawRform.getTranslation().x()) << ","
                                << cms::convert2mm(-hawRform.getTranslation().y()) << ","
                                << cms::convert2mm(-hawRform.getTranslation().z()) << ") with rotation";
#endif
    for (unsigned int iPhi(1); iPhi <= alvWedge.nFawPerSupm; ++iPhi) {
      const double rPhi(alvWedge.fawPhiOff + (iPhi - 0.5) * alvWedge.fawDelPhi);

      const Tf3D fawform(RoZ3D(rPhi) *
                         Tl3D(alvWedge.fawRadOff + (trapFAW.H() + trapFAW.h()) / 4, 0, 0.5 * trapFAW.L()) *
                         RoZ3D(-90._deg + alvWedge.fawPhiRot));
      if (alvWedge.fawHere) {
        ns.assembly(spm.name).placeVolume(
            fawLog,
            iPhi,
            Transform3D(
                myrot(ns, alvWedge.fawName + "_Rot" + std::to_string(iPhi), fawform.getRotation()),
                Position(fawform.getTranslation().x(), fawform.getTranslation().y(), fawform.getTranslation().z())));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << fawLog.name() << ":" << iPhi << " positioned in " << ns.assembly(spm.name).name()
                                    << " at (" << cms::convert2mm(fawform.getTranslation().x()) << ","
                                    << cms::convert2mm(fawform.getTranslation().y()) << ","
                                    << cms::convert2mm(fawform.getTranslation().z()) << ") with rotation";
#endif
      }
    }

    // End Alveolar Wedge parent ------------------------------------------------------

    // Begin Grid + Tablet insertion

    const double h_Grid(grid.thick);

    const EcalTrap trapGrid(0.5 * (B_hawR - h_Grid * (B_hawR - a_hawR) / H_hawR),  // bl1, A/2
                            0.5 * (b_hawR - h_Grid * (B_hawR - a_hawR) / H_hawR),  // bl2, a/2
                            0.5 * b_hawR,                                          // tl2, b/2
                            0.5 * h_Grid,                                          // h1, H/2
                            0.5 * h_Grid,                                          // h2, h/2
                            0.5 * (L_hawR - 8 * dd4hep::cm),                       // dz,  L/2
                            90._deg,                                               // alfa1
                            0,                                                     // x15
                            H_hawR - h_hawR                                        // y15
    );

    Solid gridSolid = mytrap(grid.name, trapGrid);
    Volume gridLog = Volume(grid.name, gridSolid, ns.material(grid.mat));

    const EcalTrap::VertexList vGrid(trapGrid.vertexList());

    const Tf3D gridForm(vGrid[4],
                        vGrid[5],
                        vGrid[6],  // Grid inside HAW
                        vHAW[5] - Pt3D(0, h_Grid, 0),
                        vHAW[5],
                        vHAW[6]);

    if (0 != grid.here) {
      ns.assembly(alvWedge.hawRName)
          .placeVolume(gridLog,
                       copyOne,
                       Transform3D(myrot(ns, grid.name + "R", gridForm.getRotation()),
                                   Position(gridForm.getTranslation().x(),
                                            gridForm.getTranslation().y(),
                                            gridForm.getTranslation().z())));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeomX") << gridLog.name() << ":" << copyOne << " positioned in "
                                  << ns.assembly(alvWedge.hawRName).name() << " at ("
                                  << cms::convert2mm(gridForm.getTranslation().x()) << ","
                                  << cms::convert2mm(gridForm.getTranslation().y()) << ","
                                  << cms::convert2mm(gridForm.getTranslation().z()) << ") with rotation";
#endif
    }
    // End Grid + Tablet insertion

    // begin filling Wedge with crystal plus supports --------------------------

    const double aNom(cry.nomCryDimAF);
    const double LNom(cry.nomCryDimLZ);

    const double AUnd(cry.underAR);
    const double aUnd(cry.underAF);
    const double bUnd(cry.underCF);
    const double HUnd(cry.underBR);
    const double hUnd(cry.underBF);
    const double LUnd(cry.underLZ);

    const double sWall(alv.wallThAlv);
    const double sWrap(alv.wrapThAlv);
    const double sClr(alv.clrThAlv);

    const double fWall(alv.wallFrAlv);
    const double fWrap(alv.wrapFrAlv);
    const double fClr(alv.clrFrAlv);

    const double rWall(alv.wallReAlv);
    const double rWrap(alv.wrapReAlv);
    const double rClr(alv.clrReAlv);

    // theta is angle in yz plane between z axis & leading edge of crystal
    double theta(90._deg);
    double zee(0 * dd4hep::mm);
    double side(0 * dd4hep::mm);
    double zeta(0._deg);  // increment in theta for last crystal

    for (unsigned int cryType(1); cryType <= alv.nCryTypes; ++cryType) {
      const string sType("_" + std::string(10 > cryType ? "0" : "") + std::to_string(cryType));

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalGeom") << "Crytype=" << cryType;
#endif
      const double ANom(cry.vecNomCryDimAR[cryType - 1]);
      const double BNom(cry.vecNomCryDimCR[cryType - 1]);
      const double bNom(cry.vecNomCryDimCF[cryType - 1]);
      const double HNom(cry.vecNomCryDimBR[cryType - 1]);
      const double hNom(cry.vecNomCryDimBF[cryType - 1]);

      const double alfCry(90._deg + atan((bNom - bUnd - aNom + aUnd) / (hNom - hUnd)));

      const EcalTrap trapCry(0.5 * (ANom - AUnd),        //double aHalfLengthXNegZLoY , // bl1, A/2
                             0.5 * (aNom - aUnd),        //double aHalfLengthXPosZLoY , // bl2, a/2
                             0.5 * (bNom - bUnd),        //double aHalfLengthXPosZHiY , // tl2, b/2
                             0.5 * (HNom - HUnd),        //double aHalfLengthYNegZ    , // h1, H/2
                             0.5 * (hNom - hUnd),        //double aHalfLengthYPosZ    , // h2, h/2
                             0.5 * (LNom - LUnd),        //double aHalfLengthZ        , // dz,  L/2
                             alfCry,                     //double aAngleAD            , // alfa1
                             aNom - aUnd - ANom + AUnd,  //double aCoord15X           , // x15
                             hNom - hUnd - HNom + HUnd   //double aCoord15Y             // y15
      );

      const string cryDDName(cry.name + sType);
      Solid crySolid = mytrap(cryDDName, trapCry);
      Volume cryLog = Volume(cryDDName, crySolid, ns.material(cry.mat));

      //++++++++++++++++++++++++++++++++++  APD ++++++++++++++++++++++++++++++++++
      const unsigned int copyCap(1);
      const string capDDName(cap.name + sType);
      Solid capSolid = Box(capDDName, cap.xSizeHalf, cap.ySizeHalf, cap.thickHalf);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << capDDName << " Box " << cms::convert2mm(cap.xSizeHalf) << ":"
                                 << cms::convert2mm(cap.ySizeHalf) << ":" << cms::convert2mm(cap.thickHalf);
#endif
      Volume capLog = Volume(capDDName, capSolid, ns.material(cap.mat));

      const string sglDDName(apd.sglName + sType);
      Solid sglSolid = Box(sglDDName, cap.xSizeHalf, cap.ySizeHalf, apd.sglThick * 0.5);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << sglDDName << " Box " << cms::convert2mm(cap.xSizeHalf) << ":"
                                 << cms::convert2mm(cap.ySizeHalf) << ":" << cms::convert2mm(apd.sglThick * 0.5);
#endif
      Volume sglLog = Volume(sglDDName, sglSolid, ns.material(apd.sglMat));
      const unsigned int copySGL(1);

      const string cerDDName(cer.name + sType);
      Solid cerSolid = Box(cerDDName, cer.xSizeHalf, cer.ySizeHalf, cer.thickHalf);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << cerDDName << " Box " << cms::convert2mm(cer.xSizeHalf) << ":"
                                 << cms::convert2mm(cer.ySizeHalf) << ":" << cms::convert2mm(cer.thickHalf);
#endif
      Volume cerLog = Volume(cerDDName, cerSolid, ns.material(cer.mat));
      unsigned int copyCER(0);

      const string bsiDDName(bSi.name + sType);
      Solid bsiSolid = Box(bsiDDName, bSi.xSizeHalf, bSi.ySizeHalf, bSi.thickHalf);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << bsiDDName << " Box " << cms::convert2mm(bSi.xSizeHalf) << ":"
                                 << cms::convert2mm(bSi.ySizeHalf) << ":" << cms::convert2mm(bSi.thickHalf);
#endif
      Volume bsiLog = Volume(bsiDDName, bsiSolid, ns.material(bSi.mat));
      const unsigned int copyBSi(1);

      const string atjDDName(apd.atjName + sType);
      Solid atjSolid = Box(atjDDName, 0.5 * apd.side, 0.5 * apd.side, apd.atjThickHalf);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << atjDDName << " Box " << cms::convert2mm(0.5 * apd.side) << ":"
                                 << cms::convert2mm(0.5 * apd.side) << ":" << cms::convert2mm(apd.atjThickHalf);
#endif
      Volume atjLog = Volume(atjDDName, atjSolid, ns.material(apd.atjMat));
      const unsigned int copyATJ(1);

      const string aglDDName(apd.aglName + sType);
      Solid aglSolid = Box(aglDDName, bSi.xSizeHalf, bSi.ySizeHalf, 0.5 * apd.aglThick);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << aglDDName << " Box " << cms::convert2mm(bSi.xSizeHalf) << ":"
                                 << cms::convert2mm(bSi.ySizeHalf) << ":" << cms::convert2mm(0.5 * apd.aglThick);
#endif
      Volume aglLog = Volume(aglDDName, aglSolid, ns.material(apd.aglMat));
      const unsigned int copyAGL(1);

      const string andDDName(apd.andName + sType);
      Solid andSolid = Box(andDDName, 0.5 * apd.side, 0.5 * apd.side, 0.5 * apd.andThick);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << andDDName << " Box " << cms::convert2mm(0.5 * apd.side) << ":"
                                 << cms::convert2mm(0.5 * apd.side) << ":" << cms::convert2mm(0.5 * apd.andThick);
#endif
      Volume andLog = Volume(andDDName, andSolid, ns.material(apd.andMat));
      const unsigned int copyAND(1);

      const string apdDDName(apd.name + sType);
      Solid apdSolid = Box(apdDDName, 0.5 * apd.side, 0.5 * apd.side, 0.5 * apd.thick);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << apdDDName << " Box " << cms::convert2mm(0.5 * apd.side) << ":"
                                 << cms::convert2mm(0.5 * apd.side) << ":" << cms::convert2mm(0.5 * apd.thick);
#endif
      Volume apdLog = Volume(apdDDName, apdSolid, ns.material(apd.mat));
      const unsigned int copyAPD(1);

      //++++++++++++++++++++++++++++++++++ END APD ++++++++++++++++++++++++++++++++++

      const double delta(atan((HNom - hNom) / LNom));
      const double sindelta(sin(delta));

      const double gamma(atan((ANom - aNom) / LNom));
      const double singamma(sin(gamma));

      const double beta(atan((BNom - bNom) / LNom));
      const double sinbeta(sin(beta));

      // Now clearance trap
      const double alfClr(90._deg + atan((bNom - aNom) / (hNom + sClr)));

      const EcalTrap trapClr(0.5 * (ANom + sClr + rClr * singamma),  //double aHalfLengthXNegZLoY , // bl1, A/2
                             0.5 * (aNom + sClr - fClr * singamma),  //double aHalfLengthXPosZLoY , // bl2, a/2
                             0.5 * (bNom + sClr - fClr * sinbeta),   //double aHalfLengthXPosZHiY , // tl2, b/2
                             0.5 * (HNom + sClr + rClr * sindelta),  //double aHalfLengthYNegZ    , // h1,  H/2
                             0.5 * (hNom + sClr - fClr * sindelta),  //double aHalfLengthYPosZ    , // h2,  h/2
                             0.5 * (LNom + fClr + rClr),             // dz,  L/2
                             alfClr,                                 //double aAngleAD            , // alfa1
                             aNom - ANom,                            //double aCoord15X           , // x15
                             hNom - HNom                             //double aCoord15Y             // y15
      );

      const string clrDDName(cry.clrName + sType);
      Solid clrSolid = mytrap(clrDDName, trapClr);
      Volume clrLog = Volume(clrDDName, clrSolid, ns.material(cry.clrMat));

      // Now wrap trap
      const double alfWrap(90._deg + atan((bNom - aNom) / (hNom + sClr + 2. * sWrap)));

      const EcalTrap trapWrap(0.5 * (trapClr.A() + 2. * sWrap + rWrap * singamma),  // bl1, A/2
                              0.5 * (trapClr.a() + 2. * sWrap - fWrap * singamma),  // bl2, a/2
                              0.5 * (trapClr.b() + 2. * sWrap - fWrap * sinbeta),   // tl2, b/2
                              0.5 * (trapClr.H() + 2. * sWrap + rWrap * sindelta),  // h1,  H/2
                              0.5 * (trapClr.h() + 2. * sWrap - fWrap * sindelta),  // h2,  h/2
                              0.5 * (trapClr.L() + fWrap + rWrap),                  // dz,  L/2
                              alfWrap,  //double aAngleAD            , // alfa1
                              aNom - ANom - (cryType > 9 ? 0 : 0.020 * dd4hep::mm),
                              hNom - HNom  //double aCoord15Y             // y15
      );

      const string wrapDDName(cry.wrapName + sType);
      Solid wrapSolid = mytrap(wrapDDName, trapWrap);
      Volume wrapLog = Volume(wrapDDName, wrapSolid, ns.material(cry.wrapMat));

      // Now wall trap

      const double alfWall(90._deg + atan((bNom - aNom) / (hNom + sClr + 2. * sWrap + 2. * sWall)));

      const EcalTrap trapWall(0.5 * (trapWrap.A() + 2 * sWall + rWall * singamma),  // A/2
                              0.5 * (trapWrap.a() + 2 * sWall - fWall * singamma),  // a/2
                              0.5 * (trapWrap.b() + 2 * sWall - fWall * sinbeta),   // b/2
                              0.5 * (trapWrap.H() + 2 * sWall + rWall * sindelta),  // H/2
                              0.5 * (trapWrap.h() + 2 * sWall - fWall * sindelta),  // h/2
                              0.5 * (trapWrap.L() + fWall + rWall),                 // L/2
                              alfWall,                                              // alfa1
                              aNom - ANom - (cryType < 10 ? 0.150 * dd4hep::mm : 0.100 * dd4hep::mm),
                              hNom - HNom  // y15
      );

      const string wallDDName(cry.wallName + sType);
      Solid wallSolid = mytrap(wallDDName, trapWall);
      Volume wallLog = Volume(wallDDName, wallSolid, ns.material(cry.wallMat));

      // Now for placement of cry within clr
      const Vec3 cryToClr(0., 0., 0.5 * (rClr - fClr));
      clrLog.placeVolume(cryLog, copyOne, Position(0, 0, 0.5 * (rClr - fClr)));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeomX") << cryLog.name() << ":" << copyOne << " positioned in " << clrLog.name()
                                  << " at (0,0," << cms::convert2mm(0.5 * (rClr - fClr)) << ") with no rotation";
#endif
      if (0 != cap.here) {
        bsiLog.placeVolume(aglLog, copyAGL, Position(0, 0, -0.5 * apd.aglThick + bSi.thickHalf));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << aglLog.name() << ":" << copyAGL << " positioned in " << bsiLog.name()
                                    << " at (0,0," << cms::convert2mm(-0.5 * apd.aglThick + bSi.thickHalf)
                                    << ") with no rotation";
#endif
        bsiLog.placeVolume(andLog, copyAND, Position(0, 0, -0.5 * apd.andThick - apd.aglThick + bSi.thickHalf));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << andLog.name() << ":" << copyAND << " positioned in " << bsiLog.name()
                                    << " at (0,0,"
                                    << cms::convert2mm(-0.5 * apd.andThick - apd.aglThick + bSi.thickHalf)
                                    << ") with no rotation";
#endif
        bsiLog.placeVolume(
            apdLog, copyAPD, Position(0, 0, -0.5 * apd.thick - apd.andThick - apd.aglThick + bSi.thickHalf));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << apdLog.name() << ":" << copyAPD << " positioned in " << bsiLog.name()
                                    << " at (0,0,"
                                    << cms::convert2mm(-0.5 * apd.thick - apd.andThick - apd.aglThick + bSi.thickHalf)
                                    << ") with no rotation";
#endif
        bsiLog.placeVolume(atjLog,
                           copyATJ,
                           Position(0, 0, -apd.atjThickHalf - apd.thick - apd.andThick - apd.aglThick + bSi.thickHalf));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << atjLog.name() << ":" << copyATJ << " positioned in " << bsiLog.name()
                                    << " at (0,0,"
                                    << cms::convert2mm(-apd.atjThickHalf - apd.thick - apd.andThick - apd.aglThick +
                                                       bSi.thickHalf)
                                    << ") with no rotation";
#endif
        cerLog.placeVolume(bsiLog, copyBSi, Position(0, 0, -bSi.thickHalf + cer.thickHalf));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << bsiLog.name() << ":" << copyBSi << " positioned in " << cerLog.name()
                                    << " at (0,0," << cms::convert2mm(-bSi.thickHalf + cer.thickHalf)
                                    << ") with no rotation";
#endif
        capLog.placeVolume(sglLog, copySGL, Position(0, 0, -0.5 * apd.sglThick + cap.thickHalf));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << sglLog.name() << ":" << copySGL << " positioned in " << capLog.name()
                                    << " at (0,0," << cms::convert2mm(-0.5 * apd.sglThick + cap.thickHalf)
                                    << ") with no rotation";
#endif

        for (unsigned int ijkl(0); ijkl != 2; ++ijkl) {
          capLog.placeVolume(cerLog,
                             ++copyCER,
                             Position(trapCry.bl1() - (0 == ijkl ? apd.x1 : apd.x2),
                                      trapCry.h1() - apd.z,
                                      -apd.sglThick - cer.thickHalf + cap.thickHalf));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << cerLog.name() << ":" << copyCER << " positioned in " << capLog.name()
                                      << " at (" << cms::convert2mm(trapCry.bl1() - (0 == ijkl ? apd.x1 : apd.x2))
                                      << "," << cms::convert2mm(trapCry.h1() - apd.z) << ","
                                      << cms::convert2mm(-apd.sglThick - cer.thickHalf + cap.thickHalf)
                                      << ") with no rotation";
#endif
        }
        clrLog.placeVolume(capLog, copyCap, Position(0, 0, -trapCry.dz() - cap.thickHalf + 0.5 * (rClr - fClr)));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << capLog.name() << ":" << copyCap << " positioned in " << clrLog.name()
                                    << " at (0,0,"
                                    << cms::convert2mm(-trapCry.dz() - cap.thickHalf + 0.5 * (rClr - fClr))
                                    << ") with no rotation";
#endif
      }

      const Vec3 clrToWrap(0, 0, 0.5 * (rWrap - fWrap));
      wrapLog.placeVolume(clrLog, copyOne, Position(0, 0, 0.5 * (rWrap - fWrap)));  //SAME as cryToWrap
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeomX") << clrLog.name() << ":" << copyOne << " positioned in " << wrapLog.name()
                                  << " at (0,0," << cms::convert2mm(0.5 * (rWrap - fWrap)) << ") with no rotation";
#endif

      // Now for placement of clr within wall
      const Vec3 wrapToWall1(0, 0, 0.5 * (rWall - fWall));
      const Vec3 wrapToWall(Vec3((cryType > 9 ? 0 : 0.005 * dd4hep::mm), 0, 0) + wrapToWall1);
      wallLog.placeVolume(
          wrapLog,
          copyOne,
          Position(Vec3((cryType > 9 ? 0 : 0.005 * dd4hep::mm), 0, 0) + wrapToWall1));  //SAME as wrapToWall
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeomX") << wrapLog.name() << ":" << copyOne << " positioned in " << wallLog.name() << " at ("
                                  << cms::convert2mm(wrapToWall.x()) << "," << cms::convert2mm(wrapToWall.y()) << ","
                                  << cms::convert2mm(wrapToWall.z()) << ") with no rotation";
#endif

      const EcalTrap::VertexList vWall(trapWall.vertexList());
      const EcalTrap::VertexList vCry(trapCry.vertexList());

      const double sidePrime(0.5 * (trapWall.a() - trapCry.a()));
      const double frontPrime(fWall + fWrap + fClr + 0.5 * LUnd);

      // define web plates with clearance ===========================================

      if (1 == cryType)  // first web plate: inside clearance volume
      {
        const unsigned int iWeb(0);
        const Pt3D corner(vHAW[4] + Pt3D(0, alvWedge.hawYOffCry, 0));
        const unsigned int copyOne(1);
        const double LWebx(web.vecWebLength[iWeb]);
        const double BWebx(trapWall.b() + (trapWall.B() - trapWall.b()) * LWebx / trapWall.L());

        const double thick(web.vecWebPlTh[iWeb] + web.vecWebClrTh[iWeb]);
        const EcalTrap trapWebClr(0.5 * BWebx,           // A/2
                                  0.5 * trapWall.b(),    // a/2
                                  0.5 * trapWall.b(),    // b/2
                                  0.5 * thick,           // H/2
                                  0.5 * thick,           // h/2
                                  0.5 * LWebx,           // L/2
                                  90._deg,               // alfa1
                                  trapWall.b() - BWebx,  // x15
                                  0                      // y15
        );
        std::string webClrName(web.clrName + std::to_string(iWeb));
        Solid webClrSolid = mytrap(webClrName, trapWebClr);
        Volume webClrLog = Volume(webClrName, webClrSolid, ns.material(web.clrMat));

        const EcalTrap trapWebPl(0.5 * trapWebClr.A(),             // A/2
                                 0.5 * trapWebClr.a(),             // a/2
                                 0.5 * trapWebClr.b(),             // b/2
                                 0.5 * web.vecWebPlTh[iWeb],       // H/2
                                 0.5 * web.vecWebPlTh[iWeb],       // h/2
                                 0.5 * trapWebClr.L(),             // L/2
                                 90._deg,                          // alfa1
                                 trapWebClr.b() - trapWebClr.B(),  // x15
                                 0                                 // y15
        );
        std::string webPlName(web.plName + std::to_string(iWeb));
        Solid webPlSolid = mytrap(webPlName, trapWebPl);
        Volume webPlLog = Volume(webPlName, webPlSolid, ns.material(web.plMat));

        webClrLog.placeVolume(webPlLog, copyOne);  // place plate inside clearance volume
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << webPlLog.name() << ":" << copyOne << " positioned in " << webClrName
                                    << " at (0,0,0) with no rotation";
#endif
        const EcalTrap::VertexList vWeb(trapWebClr.vertexList());

        zee += trapWebClr.h() / sin(theta);

        const double beta(theta + delta);

        const double zWeb(zee - frontPrime * cos(beta) + sidePrime * sin(beta));
        const double yWeb(frontPrime * sin(beta) + sidePrime * cos(beta));

        const Pt3D wedge3(corner + Pt3D(0, -yWeb, zWeb));
        const Pt3D wedge2(wedge3 + Pt3D(0, trapWebClr.h() * cos(theta), -trapWebClr.h() * sin(theta)));
        const Pt3D wedge1(wedge3 + Pt3D(trapWebClr.a(), 0, 0));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EcalGeom") << "trap1=" << vWeb[0] << ", trap2=" << vWeb[2] << ", trap3=" << vWeb[3];
        edm::LogVerbatim("EcalGeom") << "wedge1=" << wedge1 << ", wedge2=" << wedge2 << ", wedge3=" << wedge3;
#endif
        const Tf3D tForm(vWeb[0], vWeb[2], vWeb[3], wedge1, wedge2, wedge3);

        if (0 != web.here) {
          ns.assembly(alvWedge.hawRName)
              .placeVolume(
                  webClrLog,
                  copyOne,
                  Transform3D(
                      myrot(ns, webClrName + std::to_string(iWeb), tForm.getRotation()),
                      Position(tForm.getTranslation().x(), tForm.getTranslation().y(), tForm.getTranslation().z())));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << webClrLog.name() << ":" << copyOne << " positioned in "
                                      << ns.assembly(alvWedge.hawRName).name() << " at ("
                                      << cms::convert2mm(tForm.getTranslation().x()) << ","
                                      << cms::convert2mm(tForm.getTranslation().y()) << ","
                                      << cms::convert2mm(tForm.getTranslation().z()) << ") with rotation";
#endif
        }
        zee += alv.vecGapAlvEta[0];
      }

      for (unsigned int etaAlv(1); etaAlv <= alv.nCryPerAlvEta; ++etaAlv) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EcalGeom") << "theta=" << convertRadToDeg(theta) << ", sidePrime=" << sidePrime
                                     << ", frontPrime=" << frontPrime << ",  zeta=" << zeta << ", delta=" << delta
                                     << ",  zee=" << zee;
#endif

        zee += 0.075 * dd4hep::mm + (side * cos(zeta) + trapWall.h() - sidePrime) / sin(theta);

        // make transform for placing enclosed crystal

        const Pt3D trap2(vCry[2] + cryToClr + clrToWrap + wrapToWall);

        const Pt3D trap3(trap2 + Pt3D(0, -trapCry.h(), 0));
        const Pt3D trap1(trap3 + Pt3D(-trapCry.a(), 0, 0));

        const Pt3D wedge3(vHAW[4] + Pt3D(sidePrime, alvWedge.hawYOffCry, zee));
        const Pt3D wedge2(wedge3 + Pt3D(0, trapCry.h() * cos(theta), -trapCry.h() * sin(theta)));
        const Pt3D wedge1(wedge3 + Pt3D(trapCry.a(), 0, 0));

        const Tf3D tForm1(trap1, trap2, trap3, wedge1, wedge2, wedge3);

        const double xx(0.050 * dd4hep::mm);

        const Tf3D tForm(HepGeom::Translate3D(xx, 0, 0) * tForm1);
        ns.assembly(alvWedge.hawRName)
            .placeVolume(
                wallLog,
                etaAlv,
                Transform3D(
                    myrot(ns, wallDDName + "_" + std::to_string(etaAlv), tForm.getRotation()),
                    Position(tForm.getTranslation().x(), tForm.getTranslation().y(), tForm.getTranslation().z())));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << wallLog.name() << ":" << etaAlv << " positioned in "
                                    << ns.assembly(alvWedge.hawRName).name() << " at ("
                                    << cms::convert2mm(tForm.getTranslation().x()) << ","
                                    << cms::convert2mm(tForm.getTranslation().y()) << ","
                                    << cms::convert2mm(tForm.getTranslation().z()) << ") with rotation";
#endif
        theta -= delta;
        side = sidePrime;
        zeta = delta;
      }
      if (5 == cryType || 9 == cryType || 13 == cryType || 17 == cryType) {  // web plates
        zee += 0.5 * alv.vecGapAlvEta[cryType] / sin(theta);

        const unsigned int iWeb(cryType / 4);
        const Pt3D corner(vHAW[4] + Pt3D(0, alvWedge.hawYOffCry, 0));
        const unsigned int copyOne(1);
        const double LWebx(web.vecWebLength[iWeb]);
        const double BWebx(trapWall.a() + (trapWall.A() - trapWall.a()) * LWebx / trapWall.L());

        const double thick(web.vecWebPlTh[iWeb] + web.vecWebClrTh[iWeb]);
        const EcalTrap trapWebClr(0.5 * BWebx,           // A/2
                                  0.5 * trapWall.a(),    // a/2
                                  0.5 * trapWall.a(),    // b/2
                                  0.5 * thick,           // H/2
                                  0.5 * thick,           // h/2
                                  0.5 * LWebx,           // L/2
                                  90._deg,               // alfa1
                                  trapWall.a() - BWebx,  // x15
                                  0                      // y15
        );
        std::string webClrName(web.clrName + std::to_string(iWeb));
        Solid webClrSolid = mytrap(webClrName, trapWebClr);
        Volume webClrLog = Volume(webClrName, webClrSolid, ns.material(web.clrMat));

        const EcalTrap trapWebPl(0.5 * trapWebClr.A(),             // A/2
                                 0.5 * trapWebClr.a(),             // a/2
                                 0.5 * trapWebClr.b(),             // b/2
                                 0.5 * web.vecWebPlTh[iWeb],       // H/2
                                 0.5 * web.vecWebPlTh[iWeb],       // h/2
                                 0.5 * trapWebClr.L(),             // L/2
                                 90._deg,                          // alfa1
                                 trapWebClr.b() - trapWebClr.B(),  // x15
                                 0                                 // y15
        );
        std::string webPlName(web.plName + std::to_string(iWeb));
        Solid webPlSolid = mytrap(webPlName, trapWebPl);
        Volume webPlLog = Volume(webPlName, webPlSolid, ns.material(web.plMat));

        webClrLog.placeVolume(webPlLog, copyOne);  // place plate inside clearance volume
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << webPlLog.name() << ":" << copyOne << " positioned in " << webClrName
                                    << " at (0,0,0) with no rotation";
#endif
        const EcalTrap::VertexList vWeb(trapWebClr.vertexList());

        zee += trapWebClr.h() / sin(theta);

        const double beta(theta + delta);

        const double zWeb(zee - frontPrime * cos(beta) + sidePrime * sin(beta));
        const double yWeb(frontPrime * sin(beta) + sidePrime * cos(beta));

        const Pt3D wedge3(corner + Pt3D(0, -yWeb, zWeb));
        const Pt3D wedge2(wedge3 + Pt3D(0, trapWebClr.h() * cos(theta), -trapWebClr.h() * sin(theta)));
        const Pt3D wedge1(wedge3 + Pt3D(trapWebClr.a(), 0, 0));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EcalGeom") << "trap1=" << vWeb[0] << ", trap2=" << vWeb[2] << ", trap3=" << vWeb[3];
        edm::LogVerbatim("EcalGeom") << "wedge1=" << wedge1 << ", wedge2=" << wedge2 << ", wedge3=" << wedge3;
#endif
        const Tf3D tForm(vWeb[0], vWeb[2], vWeb[3], wedge1, wedge2, wedge3);

        if (0 != web.here) {
          ns.assembly(alvWedge.hawRName)
              .placeVolume(
                  webClrLog,
                  copyOne,
                  Transform3D(
                      myrot(ns, webClrName + std::to_string(iWeb), tForm.getRotation()),
                      Position(tForm.getTranslation().x(), tForm.getTranslation().y(), tForm.getTranslation().z())));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << webClrLog.name() << ":" << copyOne << " positioned in "
                                      << ns.assembly(alvWedge.hawRName).name() << " at ("
                                      << cms::convert2mm(tForm.getTranslation().x()) << ","
                                      << cms::convert2mm(tForm.getTranslation().y()) << ","
                                      << cms::convert2mm(tForm.getTranslation().z()) << ") with rotation";
#endif
        }

        zee += 0.5 * alv.vecGapAlvEta[cryType] / sin(theta);
      } else {
        if (17 != cryType)
          zee += alv.vecGapAlvEta[cryType] / sin(theta);
      }
    }
    // END   filling Wedge with crystal plus supports --------------------------

    //------------------------------------------------------------------------
    //------------------------------------------------------------------------
    //------------------------------------------------------------------------
    //------------------------------------------------------------------------
    //**************** Material at outer radius of supermodule ***************
    //------------------------------------------------------------------------
    //------------------------------------------------------------------------
    //------------------------------------------------------------------------
    //------------------------------------------------------------------------

    if (0 != back.here) {
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     Begin Back Cover Plate     !!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      const Position outtra(back.xOff + 0.5 * back.sideHeight, back.yOff, 0.5 * back.sideLength);

      const double realBPthick(back.plateThick + back.plate2Thick);
      array<double, 3> backPlateParms{{0.5 * back.plateWidth, 0.5 * realBPthick, 0.5 * back.plateLength}};
      Solid backPlateSolid = Box(back.plateName, backPlateParms[0], backPlateParms[1], backPlateParms[2]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << back.plateName << " Box " << cms::convert2mm(backPlateParms[0]) << ":"
                                 << cms::convert2mm(backPlateParms[1]) << ":" << cms::convert2mm(backPlateParms[2]);
#endif
      Volume backPlateLog = Volume(back.plateName, backPlateSolid, ns.material(back.plateMat));

      const Position backPlateTra(
          0.5 * back.sideHeight + backPlateParms[1], 0 * dd4hep::mm, backPlateParms[2] - 0.5 * back.sideLength);

      Solid backPlate2Solid =
          Box(back.plate2Name, 0.5 * back.plateWidth, 0.5 * back.plate2Thick, 0.5 * back.plateLength);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << back.plate2Name << " Box " << cms::convert2mm(0.5 * back.plateWidth) << ":"
                                 << cms::convert2mm(0.5 * back.plate2Thick) << ":"
                                 << cms::convert2mm(0.5 * back.plateLength);
#endif
      Volume backPlate2Log = Volume(back.plate2Name, backPlate2Solid, ns.material(back.plate2Mat));

      const Position backPlate2Tra(0, -backPlateParms[1] + back.plate2Thick * 0.5, 0);
      if (0 != back.plateHere) {
        backPlateLog.placeVolume(backPlate2Log, copyOne, Transform3D(backPlate2Tra));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << backPlate2Log.name() << ":" << copyOne << " positioned in "
                                    << backPlateLog.name() << " at (" << cms::convert2mm(backPlate2Tra.x()) << ","
                                    << cms::convert2mm(backPlate2Tra.y()) << "," << cms::convert2mm(backPlate2Tra.z())
                                    << ") with no rotation";
#endif
        ns.assembly(spm.name).placeVolume(
            backPlateLog,
            copyOne,
            Transform3D(myrot(ns, back.plateName + "Rot5", CLHEP::HepRotationZ(270._deg)), outtra + backPlateTra));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << backPlateLog.name() << ":" << copyOne << " positioned in "
                                    << ns.assembly(spm.name).name() << " at ("
                                    << cms::convert2mm((outtra + backPlateTra).x()) << ","
                                    << cms::convert2mm((outtra + backPlateTra).y()) << ","
                                    << cms::convert2mm((outtra + backPlateTra).z()) << ") with rotation";
#endif
      }
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     End Back Cover Plate       !!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     Begin Back Side Plates    !!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      const EcalTrap trapBS(back.sideWidth * 0.5,   //double aHalfLengthXNegZLoY , // bl1, A/2
                            back.sideWidth * 0.5,   //double aHalfLengthXPosZLoY , // bl2, a/2
                            back.sideWidth / 4.,    //double aHalfLengthXPosZHiY , // tl2, b/2
                            back.sideHeight * 0.5,  //double aHalfLengthYNegZ    , // h1, H/2
                            back.sideHeight * 0.5,  //double aHalfLengthYPosZ    , // h2, h/2
                            back.sideLength * 0.5,  //double aHalfLengthZ        , // dz,  L/2
                            back.sideAngle,         //double aAngleAD            , // alfa1
                            0,                      //double aCoord15X           , // x15
                            0                       //double aCoord15Y             // y15
      );

      Solid backSideSolid = mytrap(back.sideName, trapBS);
      Volume backSideLog = Volume(back.sideName, backSideSolid, ns.material(back.sideMat));

      const Position backSideTra1(0 * dd4hep::mm, back.plateWidth * 0.5 + back.sideYOff1, 1 * dd4hep::mm);
      if (0 != back.sideHere) {
        ns.assembly(spm.name).placeVolume(
            backSideLog,
            copyOne,
            Transform3D(myrot(ns, back.sideName + "Rot8", CLHEP::HepRotationX(180._deg) * CLHEP::HepRotationZ(90._deg)),
                        outtra + backSideTra1));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << backSideLog.name() << ":" << copyOne << " positioned in "
                                    << ns.assembly(spm.name).name() << " at ("
                                    << cms::convert2mm((outtra + backSideTra1).x()) << ","
                                    << cms::convert2mm((outtra + backSideTra1).y()) << ","
                                    << cms::convert2mm((outtra + backSideTra1).z()) << ") with rotation";
#endif
        const Position backSideTra2(0 * dd4hep::mm, -back.plateWidth * 0.5 + back.sideYOff2, 1 * dd4hep::mm);
        ns.assembly(spm.name).placeVolume(
            backSideLog,
            copyTwo,
            Transform3D(myrot(ns, back.sideName + "Rot9", CLHEP::HepRotationZ(90._deg)), outtra + backSideTra2));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << backSideLog.name() << ":" << copyTwo << " positioned in "
                                    << ns.assembly(spm.name).name() << " at ("
                                    << cms::convert2mm((outtra + backSideTra2).x()) << ","
                                    << cms::convert2mm((outtra + backSideTra2).y()) << ","
                                    << cms::convert2mm((outtra + backSideTra2).z()) << ") with rotation";
#endif
      }
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     End Back Side Plates       !!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      //=====================
      const double backCoolWidth(backCool.barWidth + 2. * backCoolTank.width);

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     Begin Mother Board Cooling Manifold Setup !!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      const double manifCut(2 * dd4hep::mm);

      Solid mBManifSolid = Tube(0, mbManif.outDiam * 0.5, backCoolWidth * 0.5 - manifCut, 0._deg, 360._deg);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << mbManif.name << " Tubs " << cms::convert2mm(backCoolWidth * 0.5 - manifCut)
                                 << ":0:" << cms::convert2mm(mbManif.outDiam * 0.5) << ":0:360";
#endif
      Volume mBManifLog = Volume(mbManif.name, mBManifSolid, ns.material(mbManif.mat));

      const string mBManifWaName(mbManif.name + "Wa");
      Solid mBManifWaSolid = Tube(0, mbManif.innDiam * 0.5, backCoolWidth * 0.5 - manifCut, 0._deg, 360._deg);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << mBManifWaName << " Tubs " << cms::convert2mm(backCoolWidth * 0.5 - manifCut)
                                 << ":0:" << cms::convert2mm(mbManif.innDiam * 0.5) << ":0:360";
#endif
      Volume mBManifWaLog(mBManifWaName, mBManifWaSolid, ns.material(backPipe.waterMat));
      mBManifLog.placeVolume(mBManifWaLog, copyOne);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeomX") << mBManifWaLog.name() << ":" << copyOne << " positioned in " << mBManifLog.name()
                                  << " at (0,0,0) with no rotation";
#endif

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     End Mother Board Cooling Manifold Setup   !!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //=====================

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     Begin Loop over Grilles & MB Cooling Manifold !!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      const double deltaY(-5 * dd4hep::mm);

      Solid grEdgeSlotSolid =
          Box(grille.edgeSlotName, grille.edgeSlotHeight * 0.5, grille.edgeSlotWidth * 0.5, grille.thick * 0.5);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << grille.edgeSlotName << " Box " << cms::convert2mm(grille.edgeSlotHeight * 0.5)
                                 << ":" << cms::convert2mm(grille.edgeSlotWidth * 0.5) << ":"
                                 << cms::convert2mm(grille.thick * 0.5);
#endif
      Volume grEdgeSlotLog = Volume(grille.edgeSlotName, grEdgeSlotSolid, ns.material(grille.edgeSlotMat));

      unsigned int edgeSlotCopy(0);
      unsigned int midSlotCopy(0);

      Volume grMidSlotLog[4];

      for (unsigned int iGr(0); iGr != grille.vecHeight.size(); ++iGr) {
        string gName(grille.name + std::to_string(iGr));
        Solid grilleSolid = Box(gName, grille.vecHeight[iGr] * 0.5, backCoolWidth * 0.5, grille.thick * 0.5);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << gName << " Box " << cms::convert2mm(grille.vecHeight[iGr] * 0.5) << ":"
                                   << cms::convert2mm(backCoolWidth * 0.5) << ":"
                                   << cms::convert2mm(grille.thick * 0.5);
#endif
        Volume grilleLog = Volume(gName, grilleSolid, ns.material(grille.mat));

        const Position grilleTra(-realBPthick * 0.5 - grille.vecHeight[iGr] * 0.5,
                                 deltaY,
                                 grille.vecZOff[iGr] + grille.thick * 0.5 - back.sideLength * 0.5);
        const Position gTra(outtra + backPlateTra + grilleTra);

        if (0 != grille.midSlotHere && 0 != iGr) {
          if (0 == (iGr - 1) % 2) {
            string mName(grille.midSlotName + std::to_string(iGr * 0.5));
            Solid grMidSlotSolid =
                Box(mName, grille.vecMidSlotHeight[(iGr - 1) / 2] * 0.5, grille.midSlotWidth * 0.5, grille.thick * 0.5);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("EBGeom") << mName << " Box "
                                       << cms::convert2mm(grille.vecMidSlotHeight[(iGr - 1) / 2] * 0.5) << ":"
                                       << cms::convert2mm(grille.midSlotWidth * 0.5) << ":"
                                       << cms::convert2mm(grille.thick * 0.5);
#endif
            grMidSlotLog[(iGr - 1) / 2] = Volume(mName, grMidSlotSolid, ns.material(grille.midSlotMat));
          }
          grilleLog.placeVolume(
              grMidSlotLog[(iGr - 1) / 2],
              ++midSlotCopy,
              Transform3D(Position(
                  grille.vecHeight[iGr] * 0.5 - grille.vecMidSlotHeight[(iGr - 1) / 2] * 0.5, +grille.midSlotXOff, 0)));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << grMidSlotLog[(iGr - 1) / 2].name() << ":" << midSlotCopy << " positioned in "
                                      << grilleLog.name() << " at ("
                                      << cms::convert2mm(grille.vecHeight[iGr] * 0.5 -
                                                         grille.vecMidSlotHeight[(iGr - 1) / 2] * 0.5)
                                      << "," << cms::convert2mm(grille.midSlotXOff) << ",0) with no rotation";
#endif
          grilleLog.placeVolume(
              grMidSlotLog[(iGr - 1) / 2],
              ++midSlotCopy,
              Transform3D(Position(
                  grille.vecHeight[iGr] * 0.5 - grille.vecMidSlotHeight[(iGr - 1) / 2] * 0.5, -grille.midSlotXOff, 0)));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << grMidSlotLog[(iGr - 1) / 2].name() << ":" << midSlotCopy << " positioned in "
                                      << grilleLog.name() << " at ("
                                      << cms::convert2mm(grille.vecHeight[iGr] * 0.5 -
                                                         grille.vecMidSlotHeight[(iGr - 1) / 2] * 0.5)
                                      << "," << cms::convert2mm(-grille.midSlotXOff) << ",0) with no rotation";
#endif
        }

        if (0 != grille.edgeSlotHere && 0 != iGr) {
          grilleLog.placeVolume(grEdgeSlotLog,
                                ++edgeSlotCopy,
                                Transform3D(Position(grille.vecHeight[iGr] * 0.5 - grille.edgeSlotHeight * 0.5,
                                                     backCoolWidth * 0.5 - grille.edgeSlotWidth * 0.5,
                                                     0)));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << grEdgeSlotLog.name() << ":" << edgeSlotCopy << " positioned in "
                                      << grilleLog.name() << " at ("
                                      << cms::convert2mm(grille.vecHeight[iGr] * 0.5 - grille.edgeSlotHeight * 0.5)
                                      << "," << cms::convert2mm(backCoolWidth * 0.5 - grille.edgeSlotWidth * 0.5)
                                      << ",0) with no rotation";
#endif
          grilleLog.placeVolume(grEdgeSlotLog,
                                ++edgeSlotCopy,
                                Transform3D(Position(grille.vecHeight[iGr] * 0.5 - grille.edgeSlotHeight * 0.5,
                                                     -backCoolWidth * 0.5 + grille.edgeSlotWidth * 0.5,
                                                     0)));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << grEdgeSlotLog.name() << ":" << edgeSlotCopy << " positioned in "
                                      << grilleLog.name() << " at ("
                                      << cms::convert2mm(grille.vecHeight[iGr] * 0.5 - grille.edgeSlotHeight * 0.5)
                                      << "," << cms::convert2mm(-backCoolWidth * 0.5 + grille.edgeSlotWidth * 0.5)
                                      << ",0) with no rotation";
#endif
        }
        if (0 != grille.here) {
          ns.assembly(spm.name).placeVolume(grilleLog, iGr, Transform3D(gTra));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << grilleLog.name() << ":" << iGr << " positioned in "
                                      << ns.assembly(spm.name).name() << " at (" << cms::convert2mm(gTra.x()) << ","
                                      << cms::convert2mm(gTra.y()) << "," << cms::convert2mm(gTra.z())
                                      << ") with no rotation";
#endif
        }

        if ((0 != iGr % 2) && (0 != mbManif.here)) {
          ns.assembly(spm.name).placeVolume(
              mBManifLog,
              iGr,
              Transform3D(myrot(ns, mbManif.name + "R1", CLHEP::HepRotationX(90._deg)),
                          gTra - Position(-mbManif.outDiam * 0.5 + grille.vecHeight[iGr] * 0.5,
                                          manifCut,
                                          grille.thick * 0.5 + 3 * mbManif.outDiam * 0.5)));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << mBManifLog.name() << ":" << iGr << " positioned in "
                                      << ns.assembly(spm.name).name() << " at ("
                                      << cms::convert2mm(gTra.x() + mbManif.outDiam * 0.5 - grille.vecHeight[iGr] * 0.5)
                                      << "," << cms::convert2mm(gTra.y() - manifCut) << ","
                                      << cms::convert2mm(gTra.z() - grille.thick * 0.5 - 3 * mbManif.outDiam * 0.5)
                                      << ") with rotation";
#endif
          ns.assembly(spm.name).placeVolume(
              mBManifLog,
              iGr - 1,
              Transform3D(myrot(ns, mbManif.name + "R2", CLHEP::HepRotationX(90._deg)),
                          gTra - Position(-3 * mbManif.outDiam * 0.5 + grille.vecHeight[iGr] * 0.5,
                                          manifCut,
                                          grille.thick * 0.5 + 3 * mbManif.outDiam * 0.5)));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << mBManifLog.name() << ":" << (iGr - 1) << " positioned in "
                                      << ns.assembly(spm.name).name() << " at ("
                                      << cms::convert2mm(gTra.x() + 3 * mbManif.outDiam * 0.5 -
                                                         grille.vecHeight[iGr] * 0.5)
                                      << "," << cms::convert2mm(gTra.y() - manifCut) << ","
                                      << cms::convert2mm(gTra.z() - grille.thick * 0.5 - 3 * mbManif.outDiam * 0.5)
                                      << ") with rotation";
#endif
        }
      }

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     End Loop over Grilles & MB Cooling Manifold   !!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     Begin Cooling Bar Setup    !!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      Solid backCoolBarSolid =
          Box(backCool.barName, backCool.barHeight * 0.5, backCool.barWidth * 0.5, backCool.barThick * 0.5);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << backCool.barName << " Box " << cms::convert2mm(backCool.barHeight * 0.5) << ":"
                                 << cms::convert2mm(backCool.barWidth * 0.5) << ":"
                                 << cms::convert2mm(backCool.barThick * 0.5);
#endif
      Volume backCoolBarLog = Volume(backCool.barName, backCoolBarSolid, ns.material(backCool.barMat));

      Solid backCoolBarSSSolid =
          Box(backCool.barSSName, backCool.barHeight * 0.5, backCool.barWidth * 0.5, backCool.barSSThick * 0.5);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << backCool.barSSName << " Box " << cms::convert2mm(backCool.barHeight * 0.5) << ":"
                                 << cms::convert2mm(backCool.barWidth * 0.5) << ":"
                                 << cms::convert2mm(backCool.barSSThick * 0.5);
#endif
      Volume backCoolBarSSLog = Volume(backCool.barSSName, backCoolBarSSSolid, ns.material(backCool.barSSMat));
      const Position backCoolBarSSTra(0, 0, 0);
      backCoolBarLog.placeVolume(backCoolBarSSLog, copyOne, Transform3D(backCoolBarSSTra));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeomX") << backCoolBarSSLog.name() << ":" << copyOne << " positioned in "
                                  << backCoolBarLog.name() << " at (" << cms::convert2mm(backCoolBarSSTra.x()) << ","
                                  << cms::convert2mm(backCoolBarSSTra.y()) << ","
                                  << cms::convert2mm(backCoolBarSSTra.z()) << ") with no rotation";
#endif

      Solid backCoolBarWaSolid =
          Box(backCool.barWaName, backCool.barHeight * 0.5, backCool.barWidth * 0.5, backCool.barWaThick * 0.5);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << backCool.barWaName << " Box " << cms::convert2mm(backCool.barHeight * 0.5) << ":"
                                 << cms::convert2mm(backCool.barWidth * 0.5) << ":"
                                 << cms::convert2mm(backCool.barWaThick * 0.5);
#endif
      Volume backCoolBarWaLog = Volume(backCool.barWaName, backCoolBarWaSolid, ns.material(backCool.barWaMat));
      const Position backCoolBarWaTra(0, 0, 0);
      backCoolBarSSLog.placeVolume(backCoolBarWaLog, copyOne, Transform3D(backCoolBarWaTra));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeomX") << backCoolBarWaLog.name() << ":" << copyOne << " positioned in "
                                  << backCoolBarSSLog.name() << " at (" << cms::convert2mm(backCoolBarWaTra.x()) << ","
                                  << cms::convert2mm(backCoolBarWaTra.y()) << ","
                                  << cms::convert2mm(backCoolBarWaTra.z()) << ") with no rotation";
#endif

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     End Cooling Bar Setup      !!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     Begin VFE Card Setup       !!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      double thickVFE(0);
      for (unsigned int iLyr(0); iLyr != backCool.vecBackVFELyrThick.size(); ++iLyr) {
        thickVFE += backCool.vecBackVFELyrThick[iLyr];
      }
      Solid backVFESolid =
          Box((myns + backCool.backVFEName), backCool.barHeight * 0.5, backCool.barWidth * 0.5, thickVFE * 0.5);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << (myns + backCool.backVFEName) << " Box "
                                 << cms::convert2mm(backCool.barHeight * 0.5) << ":"
                                 << cms::convert2mm(backCool.barWidth * 0.5) << ":" << cms::convert2mm(thickVFE * 0.5);
#endif
      Volume backVFELog =
          ns.addVolume(Volume(myns + backCool.backVFEName, backVFESolid, ns.material(backCool.backVFEMat)));
      Position offTra(0, 0, -thickVFE * 0.5);
      for (unsigned int iLyr(0); iLyr != backCool.vecBackVFELyrThick.size(); ++iLyr) {
        Solid backVFELyrSolid = Box(backCool.vecBackVFELyrName[iLyr],
                                    backCool.barHeight * 0.5,
                                    backCool.barWidth * 0.5,
                                    backCool.vecBackVFELyrThick[iLyr] * 0.5);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << backCool.vecBackVFELyrName[iLyr] << " Box "
                                   << cms::convert2mm(backCool.barHeight * 0.5) << ":"
                                   << cms::convert2mm(backCool.barWidth * 0.5) << ":"
                                   << cms::convert2mm(backCool.vecBackVFELyrThick[iLyr] * 0.5);
#endif
        Volume backVFELyrLog =
            Volume(backCool.vecBackVFELyrName[iLyr], backVFELyrSolid, ns.material(backCool.vecBackVFELyrMat[iLyr]));
        const Position backVFELyrTra(0, 0, backCool.vecBackVFELyrThick[iLyr] * 0.5);
        backVFELog.placeVolume(backVFELyrLog, copyOne, Transform3D(backVFELyrTra + offTra));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << backVFELyrLog.name() << ":" << copyOne << " positioned in " << backVFELog.name()
                                    << " at (" << cms::convert2mm((backVFELyrTra + offTra).x()) << ","
                                    << cms::convert2mm((backVFELyrTra + offTra).y()) << ","
                                    << cms::convert2mm((backVFELyrTra + offTra).z()) << ") with no rotation";
#endif
        offTra += 2 * backVFELyrTra;
      }

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     End VFE Card Setup         !!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     Begin Cooling Bar + VFE Setup  !!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      const double halfZCoolVFE(thickVFE + backCool.barThick * 0.5);
      Solid backCoolVFESolid = Box(backCool.barHeight * 0.5, backCool.barWidth * 0.5, halfZCoolVFE);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << backCool.vFEName << " Box " << cms::convert2mm(backCool.barHeight * 0.5) << ":"
                                 << cms::convert2mm(backCool.barWidth * 0.5) << ":" << cms::convert2mm(halfZCoolVFE);
#endif
      Volume backCoolVFELog = ns.addVolume(Volume(backCool.vFEName, backCoolVFESolid, ns.material(backCool.vFEMat)));
      if (0 != backCool.barHere) {
        backCoolVFELog.placeVolume(backCoolBarLog, copyOne, Transform3D());
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << backCoolBarLog.name() << ":" << copyOne << " positioned in "
                                    << backCoolVFELog.name() << " at (0,0,0) with no rotation";
#endif
      }
      if (0 != backCool.vFEHere) {
        backCoolVFELog.placeVolume(
            backVFELog, copyOne, Transform3D(Position(0, 0, backCool.barThick * 0.5 + thickVFE * 0.5)));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << backVFELog.name() << ":" << copyOne << " positioned in " << backCoolVFELog.name()
                                    << " at (0,0," << cms::convert2mm(backCool.barThick * 0.5 + thickVFE * 0.5)
                                    << ") with no rotation";
#endif
      }
      backCoolVFELog.placeVolume(backVFELog,
                                 copyTwo,
                                 Transform3D(myrot(ns, backCool.vFEName + "Flip", CLHEP::HepRotationX(180._deg)),
                                             Position(0, 0, -backCool.barThick * 0.5 - thickVFE * 0.5)));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeomX") << backVFELog.name() << ":" << copyTwo << " positioned in " << backCoolVFELog.name()
                                  << " at (0,0," << cms::convert2mm(-backCool.barThick * 0.5 - thickVFE * 0.5)
                                  << ") with rotation";
#endif

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!     End Cooling Bar + VFE Setup    !!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!! Begin Placement of Readout & Cooling by Module  !!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      unsigned int iCVFECopy(1);
      unsigned int iSep(0);
      unsigned int iNSec(0);
      const unsigned int nMisc(backMisc.vecThick.size() / 4);
      for (unsigned int iMod(0); iMod != 4; ++iMod) {
        const double pipeLength(grille.vecZOff[2 * iMod + 1] - grille.vecZOff[2 * iMod] - grille.thick -
                                3 * dd4hep::mm);
        const double pipeZPos(grille.vecZOff[2 * iMod + 1] - pipeLength * 0.5 - 1.5 * dd4hep::mm);

        // accumulate total height of parent volume
        double backCoolHeight(backCool.barHeight + mbCoolTube.outDiam);
        for (unsigned int iMisc(0); iMisc != nMisc; ++iMisc) {
          backCoolHeight += backMisc.vecThick[iMod * nMisc + iMisc];
        }
        double bottomThick(mbCoolTube.outDiam);
        for (unsigned int iMB(0); iMB != mbLyr.vecMBLyrThick.size(); ++iMB) {
          backCoolHeight += mbLyr.vecMBLyrThick[iMB];
          bottomThick += mbLyr.vecMBLyrThick[iMB];
        }

        const double halfZBCool((pipeLength - 2 * mbManif.outDiam - grille.zSpace) * 0.5);
        Solid backCoolSolid = Box(backCool.vecName[iMod], backCoolHeight * 0.5, backCoolWidth * 0.5, halfZBCool);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << backCool.vecName[iMod] << " Box " << cms::convert2mm(backCoolHeight * 0.5) << ":"
                                   << cms::convert2mm(backCoolWidth * 0.5) << ":" << cms::convert2mm(halfZBCool);
#endif
        Volume backCoolLog = Volume(backCool.vecName[iMod], backCoolSolid, ns.material(spm.mat));

        const Position bCoolTra(
            -realBPthick * 0.5 + backCoolHeight * 0.5 - grille.vecHeight[2 * iMod],
            deltaY,
            grille.vecZOff[2 * iMod] + grille.thick + grille.zSpace + halfZBCool - back.sideLength * 0.5);
        if (0 != backCool.here) {
          ns.assembly(spm.name).placeVolume(backCoolLog, iMod + 1, outtra + backPlateTra + bCoolTra);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << backCoolLog.name() << ":" << (iMod + 1) << " positioned in "
                                      << ns.assembly(spm.name).name() << " at ("
                                      << cms::convert2mm((outtra + backPlateTra + bCoolTra).x()) << ","
                                      << cms::convert2mm((outtra + backPlateTra + bCoolTra).y()) << ","
                                      << cms::convert2mm((outtra + backPlateTra + bCoolTra).z())
                                      << ") with no rotation";
#endif
        }

        //===
        const double backCoolTankHeight(backCool.barHeight);  // - backBracketHeight() ) ;
        const double halfZTank(halfZBCool - 5 * dd4hep::cm);

        string bTankName(backCoolTank.name + std::to_string(iMod + 1));
        Solid backCoolTankSolid = Box(bTankName, backCoolTankHeight * 0.5, backCoolTank.width * 0.5, halfZTank);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << bTankName << " Box " << cms::convert2mm(backCoolTankHeight * 0.5) << ":"
                                   << cms::convert2mm(backCoolTank.width * 0.5) << ":" << cms::convert2mm(halfZTank);
#endif
        Volume backCoolTankLog = Volume(bTankName, backCoolTankSolid, ns.material(backCoolTank.mat));
        if (0 != backCoolTank.here) {
          backCoolLog.placeVolume(backCoolTankLog,
                                  copyOne,
                                  Transform3D(Rotation3D(),
                                              Position(-backCoolHeight * 0.5 + backCoolTankHeight * 0.5 + bottomThick,
                                                       backCool.barWidth * 0.5 + backCoolTank.width * 0.5,
                                                       0)));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << backCoolTankLog.name() << ":" << copyOne << " positioned in "
                                      << backCoolLog.name() << " at ("
                                      << cms::convert2mm(-backCoolHeight * 0.5 + backCoolTankHeight * 0.5 + bottomThick)
                                      << "," << cms::convert2mm(backCool.barWidth * 0.5 + backCoolTank.width * 0.5)
                                      << ",0) with no rotation";
#endif
        }

        string bTankWaName(backCoolTank.waName + std::to_string(iMod + 1));
        Solid backCoolTankWaSolid = Box(bTankWaName,
                                        backCoolTankHeight * 0.5 - backCoolTank.thick * 0.5,
                                        backCoolTank.waWidth * 0.5,
                                        halfZTank - backCoolTank.thick * 0.5);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << bTankWaName << " Box "
                                   << cms::convert2mm(backCoolTankHeight * 0.5 - backCoolTank.thick * 0.5) << ":"
                                   << cms::convert2mm(backCoolTank.waWidth * 0.5) << ":"
                                   << cms::convert2mm(halfZTank - backCoolTank.thick * 0.5);
#endif
        Volume backCoolTankWaLog = Volume(bTankWaName, backCoolTankWaSolid, ns.material(backCoolTank.waMat));
        backCoolTankLog.placeVolume(backCoolTankWaLog, copyOne, Transform3D(Rotation3D(), Position(0, 0, 0)));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << backCoolTankWaLog.name() << ":" << copyOne << " positioned in "
                                    << backCoolTankLog.name() << " at (0,0,0) with no rotation";
#endif

        string bBracketName(backCoolTank.backBracketName + std::to_string(iMod + 1));
        Solid backBracketSolid =
            Box(bBracketName, backCoolTank.backBracketHeight * 0.5, backCoolTank.width * 0.5, halfZTank);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << bBracketName << " Box " << cms::convert2mm(backCoolTank.backBracketHeight * 0.5)
                                   << ":" << cms::convert2mm(backCoolTank.width * 0.5) << ":"
                                   << cms::convert2mm(halfZTank);
#endif
        Volume backBracketLog = Volume(bBracketName, backBracketSolid, ns.material(backCoolTank.backBracketMat));
        if (0 != backCoolTank.here) {
          backCoolLog.placeVolume(backBracketLog,
                                  copyOne,
                                  Transform3D(Rotation3D(),
                                              Position(backCool.barHeight - backCoolHeight * 0.5 -
                                                           backCoolTank.backBracketHeight * 0.5 + bottomThick,
                                                       -backCool.barWidth * 0.5 - backCoolTank.width * 0.5,
                                                       0)));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << backBracketLog.name() << ":" << copyOne << " positioned in "
                                      << backCoolLog.name() << " at ("
                                      << cms::convert2mm(backCool.barHeight - backCoolHeight * 0.5 -
                                                         backCoolTank.backBracketHeight * 0.5 + bottomThick)
                                      << "," << cms::convert2mm(-backCool.barWidth * 0.5 - backCoolTank.width * 0.5)
                                      << ",0) with no rotation";
#endif
        }
        //===

        Position bSumTra(backCool.barHeight - backCoolHeight * 0.5 + bottomThick, 0, 0);
        for (unsigned int j(0); j != nMisc; ++j) {  // loop over miscellaneous layers
          Solid bSolid = Box(backMisc.vecName[iMod * nMisc + j],
                             backMisc.vecThick[iMod * nMisc + j] * 0.5,
                             backCool.barWidth * 0.5 + backCoolTank.width,
                             halfZBCool);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeom") << backMisc.vecName[iMod * nMisc + j] << " Box "
                                     << cms::convert2mm(backMisc.vecThick[iMod * nMisc + j] * 0.5) << ":"
                                     << cms::convert2mm(backCool.barWidth * 0.5 + backCoolTank.width) << ":"
                                     << cms::convert2mm(halfZBCool);
#endif

          Volume bLog =
              Volume(backMisc.vecName[iMod * nMisc + j], bSolid, ns.material(backMisc.vecMat[iMod * nMisc + j]));

          const Position bTra(backMisc.vecThick[iMod * nMisc + j] * 0.5, 0 * dd4hep::mm, 0 * dd4hep::mm);

          if (0 != backMisc.here) {
            backCoolLog.placeVolume(bLog, copyOne, Transform3D(Rotation3D(), bSumTra + bTra));
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("EBGeomX") << bLog.name() << ":" << copyOne << " positioned in " << backCoolLog.name()
                                        << " at (" << cms::convert2mm((bSumTra + bTra).x()) << ","
                                        << cms::convert2mm((bSumTra + bTra).y()) << ","
                                        << cms::convert2mm((bSumTra + bTra).z()) << ") with no rotation";
#endif
          }

          bSumTra += 2 * bTra;
        }

        const double bHalfWidth(backCool.barWidth * 0.5 + backCoolTank.width);

        if (0 != mbLyr.here) {
          Position mTra(-backCoolHeight * 0.5 + mbCoolTube.outDiam, 0, 0);
          for (unsigned int j(0); j != mbLyr.vecMBLyrThick.size(); ++j)  // loop over MB layers
          {
            Solid mSolid = Box(mbLyr.vecMBLyrThick[j] * 0.5, bHalfWidth, halfZBCool);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("EBGeom") << (mbLyr.vecMBLyrName[j] + "_" + std::to_string(iMod + 1)) << " Box "
                                       << cms::convert2mm(mbLyr.vecMBLyrThick[j] * 0.5) << ":"
                                       << cms::convert2mm(bHalfWidth) << ":" << cms::convert2mm(halfZBCool);
#endif
            Volume mLog = Volume(
                mbLyr.vecMBLyrName[j] + "_" + std::to_string(iMod + 1), mSolid, ns.material(mbLyr.vecMBLyrMat[j]));

            mTra += Position(mbLyr.vecMBLyrThick[j] * 0.5, 0 * dd4hep::mm, 0 * dd4hep::mm);
            backCoolLog.placeVolume(mLog, copyOne, Transform3D(Rotation3D(), mTra));
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("EBGeomX") << mLog.name() << ":" << copyOne << " positioned in " << backCoolLog.name()
                                        << " at (" << cms::convert2mm(mTra.x()) << "," << cms::convert2mm(mTra.y())
                                        << "," << cms::convert2mm(mTra.z()) << ") with no rotation";
#endif
            mTra += Position(mbLyr.vecMBLyrThick[j] * 0.5, 0 * dd4hep::mm, 0 * dd4hep::mm);
          }
        }

        if (0 != mbCoolTube.here) {
          const string mBName(mbCoolTube.name + "_" + std::to_string(iMod + 1));

          Solid mBCoolTubeSolid = Tube(0, mbCoolTube.outDiam * 0.5, halfZBCool, 0._deg, 360._deg);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeom") << mBName << " Tubs " << cms::convert2mm(halfZBCool)
                                     << ":0:" << cms::convert2mm(mbCoolTube.outDiam * 0.5) << ":0:360";
#endif
          Volume mBLog = Volume(mBName, mBCoolTubeSolid, ns.material(mbCoolTube.mat));

          const string mBWaName(mbCoolTube.name + "Wa_" + std::to_string(iMod + 1));
          Solid mBCoolTubeWaSolid = Tube(mBWaName, 0, mbCoolTube.innDiam * 0.5, halfZBCool, 0._deg, 360._deg);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeom") << mBWaName << " Tubs " << cms::convert2mm(halfZBCool)
                                     << ":0:" << cms::convert2mm(mbCoolTube.innDiam * 0.5) << ":0:360";
#endif
          Volume mBWaLog = Volume(mBWaName, mBCoolTubeWaSolid, ns.material(backPipe.waterMat));
          mBLog.placeVolume(mBWaLog, copyOne);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << mBWaLog.name() << ":" << copyOne << " positioned in " << mBLog.name()
                                      << " at (0,0,0) with no rotation";
#endif

          for (unsigned int j(0); j != dryAirTube.mbCoolTubeNum; ++j)  // loop over all MB cooling circuits
          {
            backCoolLog.placeVolume(mBLog,
                                    2 * j + 1,
                                    Transform3D(Rotation3D(),
                                                Position(-backCoolHeight * 0.5 + mbCoolTube.outDiam * 0.5,
                                                         -bHalfWidth + (j + 1) * bHalfWidth / 5,
                                                         0)));
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("EBGeomX") << mBLog.name() << ":" << (2 * j + 1) << " positioned in " << backCoolLog.name()
                                        << " at (" << cms::convert2mm(-backCoolHeight * 0.5 + mbCoolTube.outDiam * 0.5)
                                        << "," << cms::convert2mm(-bHalfWidth + (j + 1) * bHalfWidth / 5)
                                        << ",0) with no rotation";
#endif
          }
        }

        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!! Begin Back Water Pipes   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (0 != backPipe.here && 0 != iMod) {
          string bPipeName(backPipe.name + "_" + std::to_string(iMod + 1));
          string bInnerName(backPipe.name + "_H2O_" + std::to_string(iMod + 1));

          Solid backPipeSolid =
              Tube(bPipeName, 0 * dd4hep::mm, backPipe.vecDiam[iMod] * 0.5, pipeLength * 0.5, 0._deg, 360._deg);
          Solid backInnerSolid = Tube(bInnerName,
                                      0 * dd4hep::mm,
                                      backPipe.vecDiam[iMod] * 0.5 - backPipe.vecThick[iMod],
                                      pipeLength * 0.5,
                                      0._deg,
                                      360._deg);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeom") << bPipeName << " Tubs " << cms::convert2mm(pipeLength * 0.5)
                                     << ":0:" << cms::convert2mm(backPipe.vecDiam[iMod] * 0.5) << ":0:360";
          edm::LogVerbatim("EBGeom") << bInnerName << " Tubs " << cms::convert2mm(pipeLength * 0.5)
                                     << ":0:" << cms::convert2mm(backPipe.vecDiam[iMod] * 0.5 - backPipe.vecThick[iMod])
                                     << ":0:360";
#endif

          Volume backPipeLog = Volume(bPipeName, backPipeSolid, ns.material(backPipe.mat));
          Volume backInnerLog = Volume(bInnerName, backInnerSolid, ns.material(backPipe.waterMat));

          const Position bPipeTra1(back.xOff + back.sideHeight - 0.7 * backPipe.vecDiam[iMod],
                                   back.yOff + back.plateWidth * 0.5 - back.sideWidth - 0.7 * backPipe.vecDiam[iMod],
                                   pipeZPos);

          ns.assembly(spm.name).placeVolume(backPipeLog, copyOne, Transform3D(Rotation3D(), bPipeTra1));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << backPipeLog.name() << ":" << copyOne << " positioned in "
                                      << ns.assembly(spm.name).name() << " at (" << cms::convert2mm(bPipeTra1.x())
                                      << "," << cms::convert2mm(bPipeTra1.y()) << "," << cms::convert2mm(bPipeTra1.z())
                                      << ") with no rotation";
#endif
          const Position bPipeTra2(bPipeTra1.x(),
                                   back.yOff - back.plateWidth * 0.5 + back.sideWidth + backPipe.vecDiam[iMod],
                                   bPipeTra1.z());

          ns.assembly(spm.name).placeVolume(backPipeLog, copyTwo, Transform3D(Rotation3D(), bPipeTra2));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << backPipeLog.name() << ":" << copyTwo << " positioned in "
                                      << ns.assembly(spm.name).name() << " at (" << cms::convert2mm(bPipeTra2.x())
                                      << "," << cms::convert2mm(bPipeTra2.y()) << "," << cms::convert2mm(bPipeTra2.z())
                                      << ") with no rotation";
#endif

          backPipeLog.placeVolume(backInnerLog, copyOne, Transform3D(Rotation3D(), Position()));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << backInnerLog.name() << ":" << copyOne << " positioned in "
                                      << backPipeLog.name() << " at (0,0,0) with no rotation";
#endif
        }
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!! End Back Water Pipes   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        //=================================================

        if (0 != dryAirTube.here) {
          string dryAirTubName(dryAirTube.name + std::to_string(iMod + 1));

          Solid dryAirTubeSolid = Tube(
              dryAirTubName, dryAirTube.innDiam * 0.5, dryAirTube.outDiam * 0.5, pipeLength * 0.5, 0._deg, 360._deg);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeom") << dryAirTubName << " Tubs " << cms::convert2mm(pipeLength * 0.5) << ":"
                                     << cms::convert2mm(dryAirTube.innDiam * 0.5) << ":"
                                     << cms::convert2mm(dryAirTube.outDiam * 0.5) << ":0:360";
#endif
          Volume dryAirTubeLog = Volume((myns + dryAirTubName), dryAirTubeSolid, ns.material(dryAirTube.mat));

          const Position dryAirTubeTra1(back.xOff + back.sideHeight - 0.7 * dryAirTube.outDiam - backPipe.vecDiam[iMod],
                                        back.yOff + back.plateWidth * 0.5 - back.sideWidth - 1.2 * dryAirTube.outDiam,
                                        pipeZPos);

          ns.assembly(spm.name).placeVolume(dryAirTubeLog, copyOne, Transform3D(Rotation3D(), dryAirTubeTra1));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << dryAirTubeLog.name() << ":" << copyOne << " positioned in "
                                      << ns.assembly(spm.name).name() << " at (" << cms::convert2mm(dryAirTubeTra1.x())
                                      << "," << cms::convert2mm(dryAirTubeTra1.y()) << ","
                                      << cms::convert2mm(dryAirTubeTra1.z()) << ") with no rotation";
#endif

          const Position dryAirTubeTra2(dryAirTubeTra1.x(),
                                        back.yOff - back.plateWidth * 0.5 + back.sideWidth + 0.7 * dryAirTube.outDiam,
                                        dryAirTubeTra1.z());

          ns.assembly(spm.name).placeVolume(dryAirTubeLog, copyTwo, Transform3D(Rotation3D(), dryAirTubeTra2));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << dryAirTubeLog.name() << ":" << copyTwo << " positioned in "
                                      << ns.assembly(spm.name).name() << " at (" << cms::convert2mm(dryAirTubeTra2.x())
                                      << "," << cms::convert2mm(dryAirTubeTra2.y()) << ","
                                      << cms::convert2mm(dryAirTubeTra2.z()) << ") with no rotation";
#endif
        }
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!! Begin Placement of Cooling + VFE Cards          !!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Position cTra(backCool.barHeight * 0.5 - backCoolHeight * 0.5 + bottomThick, 0, -halfZTank + halfZCoolVFE);
        const unsigned int numSec(static_cast<unsigned int>(backCool.vecBackCoolNSec[iMod]));
        for (unsigned int jSec(0); jSec != numSec; ++jSec) {
          const unsigned int nMax(static_cast<unsigned int>(backCool.vecBackCoolNPerSec[iNSec++]));
          for (unsigned int iBar(0); iBar != nMax; ++iBar) {
            backCoolLog.placeVolume(backCoolVFELog, iCVFECopy++, cTra);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("EBGeomX") << backCoolVFELog.name() << ":" << iCVFECopy << " positioned in "
                                        << backCoolLog.name() << " at (" << cms::convert2mm(cTra.x()) << ","
                                        << cms::convert2mm(cTra.y()) << "," << cms::convert2mm(cTra.z())
                                        << ") with no rotation";
#endif
            cTra += Position(0, 0, backMisc.backCBStdSep);
          }
          cTra -= Position(0, 0, backMisc.backCBStdSep);  // backspace to previous
          if (jSec != numSec - 1)
            cTra += Position(0, 0, backCool.vecBackCoolSecSep[iSep++]);  // now take atypical step
        }
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!! End Placement of Cooling + VFE Cards            !!!!!!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      }

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!! End Placement of Readout & Cooling by Module    !!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!! Begin Patch Panel   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      double patchHeight(0);
      for (unsigned int iPatch(0); iPatch != patchPanel.vecThick.size(); ++iPatch) {
        patchHeight += patchPanel.vecThick[iPatch];
      }

      array<double, 3> patchParms{{patchHeight * 0.5,
                                   backCool.barWidth * 0.5,
                                   (spm.vecZPts.back() - grille.vecZOff.back() - grille.thick) / 2}};
      Solid patchSolid = Box(patchParms[0], patchParms[1], patchParms[2]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EBGeom") << patchPanel.name << " Box " << cms::convert2mm(patchParms[0]) << ":"
                                 << cms::convert2mm(patchParms[1]) << ":" << cms::convert2mm(patchParms[2]);
#endif
      Volume patchLog = Volume(patchPanel.name, patchSolid, ns.material(spm.mat));

      const Position patchTra(
          back.xOff + 4 * dd4hep::mm, 0 * dd4hep::mm, grille.vecZOff.back() + grille.thick + patchParms[2]);
      if (0 != patchPanel.here) {
        ns.assembly(spm.name).placeVolume(patchLog, copyOne, patchTra);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << patchLog.name() << ":" << copyOne << " positioned in " << spm.name << " at ("
                                    << cms::convert2mm(patchTra.x()) << "," << cms::convert2mm(patchTra.y()) << ","
                                    << cms::convert2mm(patchTra.z()) << ") with no rotation";
#endif
      }

      Position pTra(-patchParms[0], 0, 0);

      for (unsigned int j(0); j != patchPanel.vecNames.size(); ++j) {
        Solid pSolid = Box(patchPanel.vecThick[j] * 0.5, patchParms[1], patchParms[2]);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << patchPanel.vecNames[j] << " Box " << cms::convert2mm(patchPanel.vecThick[j] * 0.5)
                                   << ":" << cms::convert2mm(patchParms[1]) << ":" << cms::convert2mm(patchParms[2]);
#endif
        Volume pLog = Volume(patchPanel.vecNames[j], pSolid, ns.material(patchPanel.vecMat[j]));

        pTra += Position(patchPanel.vecThick[j] * 0.5, 0 * dd4hep::mm, 0 * dd4hep::mm);
        patchLog.placeVolume(pLog, copyOne, pTra);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << pLog.name() << ":" << copyOne << " positioned in " << patchLog.name() << " at ("
                                    << cms::convert2mm(pTra.x()) << "," << cms::convert2mm(pTra.y()) << ","
                                    << cms::convert2mm(pTra.z()) << ") with no rotation";
#endif

        pTra += Position(patchPanel.vecThick[j] * 0.5, 0 * dd4hep::mm, 0 * dd4hep::mm);
      }
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!! End Patch Panel     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!! Begin Pincers       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      if (0 != pincer.rodHere) {
        // Make hierarchy of rods, envelopes, blocks, shims, and cutouts

        Solid rodSolid = Box(pincer.rodName, pincer.envWidthHalf, pincer.envHeightHalf, ilyLengthHalf);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << pincer.rodName << " Box " << cms::convert2mm(pincer.envWidthHalf) << ":"
                                   << cms::convert2mm(pincer.envHeightHalf) << ":" << cms::convert2mm(ilyLengthHalf);
#endif
        Volume rodLog = Volume(pincer.rodName, rodSolid, ns.material(pincer.rodMat));

        array<double, 3> envParms{{pincer.envWidthHalf, pincer.envHeightHalf, pincer.envLengthHalf}};
        Solid envSolid = Box(pincer.envName, envParms[0], envParms[1], envParms[2]);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << pincer.envName << " Box " << cms::convert2mm(envParms[0]) << ":"
                                   << cms::convert2mm(envParms[1]) << ":" << cms::convert2mm(envParms[2]);
#endif
        Volume envLog = Volume(pincer.envName, envSolid, ns.material(pincer.envMat));

        array<double, 3> blkParms{{pincer.envWidthHalf, pincer.envHeightHalf, pincer.blkLengthHalf}};
        Solid blkSolid = Box(pincer.blkName, blkParms[0], blkParms[1], blkParms[2]);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << pincer.blkName << " Box " << cms::convert2mm(blkParms[0]) << ":"
                                   << cms::convert2mm(blkParms[1]) << ":" << cms::convert2mm(blkParms[2]);
#endif
        Volume blkLog = Volume(pincer.blkName, blkSolid, ns.material(pincer.blkMat));

        envLog.placeVolume(blkLog, copyOne, Position(0, 0, pincer.envLengthHalf - pincer.blkLengthHalf));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << blkLog.name() << ":" << copyOne << " positioned in " << envLog.name()
                                    << " at (0,0," << cms::convert2mm(pincer.envLengthHalf - pincer.blkLengthHalf)
                                    << ") with no rotation";
#endif

        array<double, 3> cutParms{{pincer.cutWidth * 0.5, pincer.cutHeight * 0.5, pincer.blkLengthHalf}};
        Solid cutSolid = Box(pincer.cutName, cutParms[0], cutParms[1], cutParms[2]);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << pincer.cutName << " Box " << cms::convert2mm(cutParms[0]) << ":"
                                   << cms::convert2mm(cutParms[1]) << ":" << cms::convert2mm(cutParms[2]);
#endif
        Volume cutLog = Volume(pincer.cutName, cutSolid, ns.material(pincer.cutMat));
        blkLog.placeVolume(
            cutLog,
            copyOne,
            Position(
                +blkParms[0] - cutParms[0] - pincer.shim1Width + pincer.shim2Width, -blkParms[1] + cutParms[1], 0));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << cutLog.name() << ":" << copyOne << " positioned in " << blkLog.name() << " at ("
                                    << cms::convert2mm(+blkParms[0] - cutParms[0] - pincer.shim1Width +
                                                       pincer.shim2Width)
                                    << "," << cms::convert2mm(-blkParms[1] + cutParms[1]) << ",0) with no rotation";
#endif
        array<double, 3> shim2Parms{{pincer.shim2Width * 0.5, pincer.shimHeight * 0.5, pincer.blkLengthHalf}};
        Solid shim2Solid = Box(pincer.shim2Name, shim2Parms[0], shim2Parms[1], shim2Parms[2]);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << pincer.shim2Name << " Box " << cms::convert2mm(shim2Parms[0]) << ":"
                                   << cms::convert2mm(shim2Parms[1]) << ":" << cms::convert2mm(shim2Parms[2]);
#endif
        Volume shim2Log = Volume(pincer.shim2Name, shim2Solid, ns.material(pincer.shimMat));
        cutLog.placeVolume(shim2Log, copyOne, Position(+cutParms[0] - shim2Parms[0], -cutParms[1] + shim2Parms[1], 0));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << shim2Log.name() << ":" << copyOne << " positioned in " << cutLog.name()
                                    << " at (" << cms::convert2mm(cutParms[0] - shim2Parms[0]) << ","
                                    << cms::convert2mm(-cutParms[1] + shim2Parms[1]) << ",0) with no rotation";
#endif

        array<double, 3> shim1Parms{
            {pincer.shim1Width * 0.5, pincer.shimHeight * 0.5, pincer.envLengthHalf - pincer.blkLengthHalf}};
        Solid shim1Solid = Box(pincer.shim1Name, shim1Parms[0], shim1Parms[1], shim1Parms[2]);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeom") << pincer.shim1Name << " Box " << cms::convert2mm(shim1Parms[0]) << ":"
                                   << cms::convert2mm(shim1Parms[1]) << ":" << cms::convert2mm(shim1Parms[2]);
#endif
        Volume shim1Log = Volume(pincer.shim1Name, shim1Solid, ns.material(pincer.shimMat));
        envLog.placeVolume(
            shim1Log,
            copyOne,
            Position(+envParms[0] - shim1Parms[0], -envParms[1] + shim1Parms[1], -envParms[2] + shim1Parms[2]));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EBGeomX") << shim1Log.name() << ":" << copyOne << " positioned in " << envLog.name()
                                    << " at (" << cms::convert2mm(envParms[0] - shim1Parms[0]) << ","
                                    << cms::convert2mm(-envParms[1] + shim1Parms[1]) << ","
                                    << cms::convert2mm(-envParms[2] + shim1Parms[2]) << ") with no rotation";
#endif

        for (unsigned int iEnv(0); iEnv != pincer.vecEnvZOff.size(); ++iEnv) {
          rodLog.placeVolume(
              envLog, 1 + iEnv, Position(0, 0, -ilyLengthHalf + pincer.vecEnvZOff[iEnv] - pincer.envLengthHalf));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << envLog.name() << ":" << (1 + iEnv) << " positioned in " << rodLog.name()
                                      << " at (0,0,"
                                      << cms::convert2mm(-ilyLengthHalf + pincer.vecEnvZOff[iEnv] -
                                                         pincer.envLengthHalf)
                                      << ") with no rotation";
#endif
        }

        // Place the rods
        const double radius(ilyRMin - pincer.envHeightHalf - 1 * dd4hep::mm);
        const string xilyName(ily.name + std::to_string(ily.vecIlyMat.size() - 1));

        for (unsigned int iRod(0); iRod != pincer.vecRodAzimuth.size(); ++iRod) {
          const Position rodTra(radius * cos(pincer.vecRodAzimuth[iRod]), radius * sin(pincer.vecRodAzimuth[iRod]), 0);
          xilyLog.placeVolume(rodLog,
                              1 + iRod,
                              Transform3D(myrot(ns,
                                                pincer.rodName + std::to_string(iRod),
                                                CLHEP::HepRotationZ(90._deg + pincer.vecRodAzimuth[iRod])),
                                          rodTra));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EBGeomX") << rodLog.name() << ":" << (1 + iRod) << " positioned in " << xilyLog.name()
                                      << " at (" << cms::convert2mm(rodTra.x()) << "," << cms::convert2mm(rodTra.y())
                                      << "," << cms::convert2mm(rodTra.z()) << ") with rotation";
#endif
        }
      }
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!! End   Pincers       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    }
  }

  return 1;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_ecal_DDEcalBarrelNewAlgo, algorithm)
