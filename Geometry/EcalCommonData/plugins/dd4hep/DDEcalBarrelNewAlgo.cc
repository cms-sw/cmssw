#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace geant_units::operators;

using VecDouble = vector<double>;
using VecStr = vector<string>;

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
    VecDouble vecRota3;  // 2nd Barrel rotation
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
    string name;   // Capsule
    double here;   //
    string mat;    //
    double xSize;  //
    double ySize;  //
    double thick;  //
  };

  struct Ceramic {
    string name;   // Ceramic
    string mat;    //
    double xSize;  //
    double ySize;  //
    double thick;  //
  };

  struct BulkSilicon {
    string name;   // Bulk Silicon
    string mat;    //
    double xSize;  //
    double ySize;  //
    double thick;  //
  };

  struct APD {
    string name;   // APD
    string mat;    //
    double side;   //
    double thick;  //
    double z;      //
    double x1;     //
    double x2;     //

    string atjName;   // After-The-Junction
    string atjMat;    //
    double atjThick;  //

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

    string pipeName;             // Cooling pipes
    double pipeHere;             //
    string pipeMat;              //
    double pipeOD;               //
    double pipeID;               //
    VecDouble vecIlyPipeLength;  //
    VecDouble vecIlyPipeType;    //
    VecDouble vecIlyPipePhi;     //
    VecDouble vecIlyPipeZ;       //

    string pTMName;          // PTM
    double pTMHere;          //
    string pTMMat;           //
    double pTMWidth;         //
    double pTMLength;        //
    double pTMHeight;        //
    VecDouble vecIlyPTMZ;    //
    VecDouble vecIlyPTMPhi;  //

    string fanOutName;          // FanOut
    double fanOutHere;          //
    string fanOutMat;           //
    double fanOutWidth;         //
    double fanOutLength;        //
    double fanOutHeight;        //
    VecDouble vecIlyFanOutZ;    //
    VecDouble vecIlyFanOutPhi;  //
    string diffName;            // Diffuser
    string diffMat;             //
    double diffOff;             //
    double diffLength;          //
    string bndlName;            // Fiber bundle
    string bndlMat;             //
    double bndlOff;             //
    double bndlLength;          //
    string fEMName;             // FEM
    string fEMMat;              //
    double fEMWidth;            //
    double fEMLength;           //
    double fEMHeight;           //
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
    double envWidth;       //
    double envHeight;      //
    double envLength;      //
    VecDouble vecEnvZOff;  //

    string blkName;    // pincer block
    string blkMat;     //
    double blkLength;  //

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
}  // namespace

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  // Barrel volume
  // barrel parent volume
  Barrel bar;
  bar.name = args.str("BarName");           // Barrel volume name
  bar.mat = args.str("BarMat");             // Barrel material name
  bar.vecZPts = args.vecDble("BarZPts");    // Barrel list of z pts
  bar.vecRMin = args.vecDble("BarRMin");    // Barrel list of rMin pts
  bar.vecRMax = args.vecDble("BarRMax");    // Barrel list of rMax pts
  bar.vecTran = args.vecDble("BarTran");    // Barrel translation
  bar.vecRota = args.vecDble("BarRota");    // Barrel rotation
  bar.vecRota2 = args.vecDble("BarRota2");  // 2nd Barrel rotation
  bar.vecRota3 = args.vecDble("BarRota3");  // 2nd Barrel rotation
  bar.phiLo = args.dble("BarPhiLo");        // Barrel phi lo
  bar.phiHi = args.dble("BarPhiHi");        // Barrel phi hi
  bar.here = args.dble("BarHere");          // Barrel presence flag

  // Supermodule volume
  Supermodule spm;
  spm.name = args.str("SpmName");               // Supermodule volume name
  spm.mat = args.str("SpmMat");                 // Supermodule material name
  spm.vecZPts = args.vecDble("SpmZPts");        // Supermodule list of z pts
  spm.vecRMin = args.vecDble("SpmRMin");        // Supermodule list of rMin pts
  spm.vecRMax = args.vecDble("SpmRMax");        // Supermodule list of rMax pts
  spm.vecTran = args.vecDble("SpmTran");        // Supermodule translation
  spm.vecRota = args.vecDble("SpmRota");        // Supermodule rotation
  spm.vecBTran = args.vecDble("SpmBTran");      // Base Supermodule translation
  spm.vecBRota = args.vecDble("SpmBRota");      // Base Supermodule rotation
  spm.nPerHalf = args.integer("SpmNPerHalf");   // # Supermodules per half detector
  spm.lowPhi = args.dble("SpmLowPhi");          // Low   phi value of base supermodule
  spm.delPhi = args.dble("SpmDelPhi");          // Delta phi value of base supermodule
  spm.phiOff = args.dble("SpmPhiOff");          // Phi offset value supermodule
  spm.vecHere = args.vecDble("SpmHere");        // Bit saying if a supermodule is present or not
  spm.cutName = args.str("SpmCutName");         // Name of cut box
  spm.cutThick = args.dble("SpmCutThick");      // Box thickness
  spm.cutShow = args.value<int>("SpmCutShow");  // Non-zero means show the box on display (testing only)
  spm.vecCutTM = args.vecDble("SpmCutTM");      // Translation for minus phi cut box
  spm.vecCutTP = args.vecDble("SpmCutTP");      // Translation for plus  phi cut box
  spm.cutRM = args.dble("SpmCutRM");            // Rotation for minus phi cut box
  spm.cutRP = args.dble("SpmCutRP");            // Rotation for plus  phi cut box
  spm.expThick = args.dble("SpmExpThick");      // Thickness (x) of supermodule expansion box
  spm.expWide = args.dble("SpmExpWide");        // Width     (y) of supermodule expansion box
  spm.expYOff = args.dble("SpmExpYOff");        // Offset    (y) of supermodule expansion box
  spm.sideName = args.str("SpmSideName");       // Supermodule Side Plate volume name
  spm.sideMat = args.str("SpmSideMat");         // Supermodule Side Plate material name
  spm.sideHigh = args.dble("SpmSideHigh");      // Side plate height
  spm.sideThick = args.dble("SpmSideThick");    // Side plate thickness
  spm.sideYOffM = args.dble("SpmSideYOffM");    // Side plate Y offset on minus phi side
  spm.sideYOffP = args.dble("SpmSideYOffP");    // Side plate Y offset on plus  phi side

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

  cry.name = args.str("CryName");
  cry.clrName = args.str("ClrName");
  cry.wrapName = args.str("WrapName");
  cry.wallName = args.str("WallName");

  cry.mat = args.str("CryMat");
  cry.clrMat = args.str("ClrMat");
  cry.wrapMat = args.str("WrapMat");
  cry.wallMat = args.str("WallMat");

  Capsule cap;
  cap.name = args.str("CapName");
  cap.here = args.dble("CapHere");
  cap.mat = args.str("CapMat");
  cap.xSize = args.dble("CapXSize");
  cap.ySize = args.dble("CapYSize");
  cap.thick = args.dble("CapThick");

  Ceramic cer;
  cer.name = args.str("CerName");
  cer.mat = args.str("CerMat");
  cer.xSize = args.dble("CerXSize");
  cer.ySize = args.dble("CerYSize");
  cer.thick = args.dble("CerThick");

  BulkSilicon bSi;
  bSi.name = args.str("BSiName");
  bSi.mat = args.str("BSiMat");
  bSi.xSize = args.dble("BSiXSize");
  bSi.ySize = args.dble("BSiYSize");
  bSi.thick = args.dble("BSiThick");

  APD apd;
  apd.name = args.str("APDName");
  apd.mat = args.str("APDMat");
  apd.side = args.dble("APDSide");
  apd.thick = args.dble("APDThick");
  apd.z = args.dble("APDZ");
  apd.x1 = args.dble("APDX1");
  apd.x2 = args.dble("APDX2");

  apd.atjName = args.str("ATJName");
  apd.atjMat = args.str("ATJMat");
  apd.atjThick = args.dble("ATJThick");

  apd.sglName = args.str("SGLName");
  apd.sglMat = args.str("SGLMat");
  apd.sglThick = args.dble("SGLThick");

  apd.aglName = args.str("AGLName");
  apd.aglMat = args.str("AGLMat");
  apd.aglThick = args.dble("AGLThick");

  apd.andName = args.str("ANDName");
  apd.andMat = args.str("ANDMat");
  apd.andThick = args.dble("ANDThick");

  Web web;
  web.here = args.dble("WebHere");
  web.plName = args.str("WebPlName");
  web.clrName = args.str("WebClrName");
  web.plMat = args.str("WebPlMat");
  web.clrMat = args.str("WebClrMat");
  web.vecWebPlTh = args.vecDble("WebPlTh");
  web.vecWebClrTh = args.vecDble("WebClrTh");
  web.vecWebLength = args.vecDble("WebLength");

  InnerLayerVolume ily;
  ily.here = args.dble("IlyHere");
  ily.name = args.str("IlyName");
  ily.phiLow = args.dble("IlyPhiLow");
  ily.delPhi = args.dble("IlyDelPhi");
  ily.vecIlyMat = args.vecStr("IlyMat");
  ily.vecIlyThick = args.vecDble("IlyThick");

  ily.pipeName = args.str("IlyPipeName");
  ily.pipeHere = args.dble("IlyPipeHere");
  ily.pipeMat = args.str("IlyPipeMat");
  ily.pipeOD = args.dble("IlyPipeOD");
  ily.pipeID = args.dble("IlyPipeID");
  ily.vecIlyPipeLength = args.vecDble("IlyPipeLength");
  ily.vecIlyPipeType = args.vecDble("IlyPipeType");
  ily.vecIlyPipePhi = args.vecDble("IlyPipePhi");
  ily.vecIlyPipeZ = args.vecDble("IlyPipeZ");

  ily.pTMName = args.str("IlyPTMName");
  ily.pTMHere = args.dble("IlyPTMHere");
  ily.pTMMat = args.str("IlyPTMMat");
  ily.pTMWidth = args.dble("IlyPTMWidth");
  ily.pTMLength = args.dble("IlyPTMLength");
  ily.pTMHeight = args.dble("IlyPTMHeight");
  ily.vecIlyPTMZ = args.vecDble("IlyPTMZ");
  ily.vecIlyPTMPhi = args.vecDble("IlyPTMPhi");

  ily.fanOutName = args.str("IlyFanOutName");
  ily.fanOutHere = args.dble("IlyFanOutHere");
  ily.fanOutMat = args.str("IlyFanOutMat");
  ily.fanOutWidth = args.dble("IlyFanOutWidth");
  ily.fanOutLength = args.dble("IlyFanOutLength");
  ily.fanOutHeight = args.dble("IlyFanOutHeight");
  ily.vecIlyFanOutZ = args.vecDble("IlyFanOutZ");
  ily.vecIlyFanOutPhi = args.vecDble("IlyFanOutPhi");
  ily.diffName = args.str("IlyDiffName");
  ily.diffMat = args.str("IlyDiffMat");
  ily.diffOff = args.dble("IlyDiffOff");
  ily.diffLength = args.dble("IlyDiffLength");
  ily.bndlName = args.str("IlyBndlName");
  ily.bndlMat = args.str("IlyBndlMat");
  ily.bndlOff = args.dble("IlyBndlOff");
  ily.bndlLength = args.dble("IlyBndlLength");
  ily.fEMName = args.str("IlyFEMName");
  ily.fEMMat = args.str("IlyFEMMat");
  ily.fEMWidth = args.dble("IlyFEMWidth");
  ily.fEMLength = args.dble("IlyFEMLength");
  ily.fEMHeight = args.dble("IlyFEMHeight");
  ily.vecIlyFEMZ = args.vecDble("IlyFEMZ");
  ily.vecIlyFEMPhi = args.vecDble("IlyFEMPhi");

  AlveolarWedge alvWedge;
  alvWedge.hawRName = args.str("HawRName");
  alvWedge.fawName = args.str("FawName");
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
  grid.name = args.str("GridName");
  grid.mat = args.str("GridMat");
  grid.thick = args.dble("GridThick");

  Back back;
  back.here = args.dble("BackHere");
  back.xOff = args.dble("BackXOff");
  back.yOff = args.dble("BackYOff");
  back.sideName = args.str("BackSideName");
  back.sideHere = args.dble("BackSideHere");
  back.sideLength = args.dble("BackSideLength");
  back.sideHeight = args.dble("BackSideHeight");
  back.sideWidth = args.dble("BackSideWidth");
  back.sideYOff1 = args.dble("BackSideYOff1");
  back.sideYOff2 = args.dble("BackSideYOff2");
  back.sideAngle = args.dble("BackSideAngle");
  back.sideMat = args.str("BackSideMat");
  back.plateName = args.str("BackPlateName");
  back.plateHere = args.dble("BackPlateHere");
  back.plateLength = args.dble("BackPlateLength");
  back.plateThick = args.dble("BackPlateThick");
  back.plateWidth = args.dble("BackPlateWidth");
  back.plateMat = args.str("BackPlateMat");
  back.plate2Name = args.str("BackPlate2Name");
  back.plate2Thick = args.dble("BackPlate2Thick");
  back.plate2Mat = args.str("BackPlate2Mat");

  Grille grille;
  grille.name = args.str("GrilleName");
  grille.here = args.dble("GrilleHere");
  grille.thick = args.dble("GrilleThick");
  grille.width = args.dble("GrilleWidth");
  grille.zSpace = args.dble("GrilleZSpace");
  grille.mat = args.str("GrilleMat");
  grille.vecHeight = args.vecDble("GrilleHeight");
  grille.vecZOff = args.vecDble("GrilleZOff");

  grille.edgeSlotName = args.str("GrEdgeSlotName");
  grille.edgeSlotMat = args.str("GrEdgeSlotMat");
  grille.edgeSlotHere = args.dble("GrEdgeSlotHere");
  grille.edgeSlotHeight = args.dble("GrEdgeSlotHeight");
  grille.edgeSlotWidth = args.dble("GrEdgeSlotWidth");
  grille.midSlotName = args.str("GrMidSlotName");
  grille.midSlotMat = args.str("GrMidSlotMat");
  grille.midSlotHere = args.dble("GrMidSlotHere");
  grille.midSlotWidth = args.dble("GrMidSlotWidth");
  grille.midSlotXOff = args.dble("GrMidSlotXOff");
  grille.vecMidSlotHeight = args.vecDble("GrMidSlotHeight");

  BackPipe backPipe;
  backPipe.here = args.dble("BackPipeHere");
  backPipe.name = args.str("BackPipeName");
  backPipe.vecDiam = args.vecDble("BackPipeDiam");
  backPipe.vecThick = args.vecDble("BackPipeThick");
  backPipe.mat = args.str("BackPipeMat");
  backPipe.waterMat = args.str("BackPipeWaterMat");

  BackCooling backCool;
  backCool.here = args.dble("BackCoolHere");
  backCool.vecName = args.vecStr("BackCoolName");
  backCool.barHere = args.dble("BackCoolBarHere");
  backCool.barWidth = args.dble("BackCoolBarWidth");
  backCool.barHeight = args.dble("BackCoolBarHeight");
  backCool.mat = args.str("BackCoolMat");
  backCool.barName = args.str("BackCoolBarName");
  backCool.barThick = args.dble("BackCoolBarThick");
  backCool.barMat = args.str("BackCoolBarMat");
  backCool.barSSName = args.str("BackCoolBarSSName");
  backCool.barSSThick = args.dble("BackCoolBarSSThick");
  backCool.barSSMat = args.str("BackCoolBarSSMat");
  backCool.barWaName = args.str("BackCoolBarWaName");
  backCool.barWaThick = args.dble("BackCoolBarWaThick");
  backCool.barWaMat = args.str("BackCoolBarWaMat");
  backCool.vFEHere = args.dble("BackCoolVFEHere");
  backCool.vFEName = args.str("BackCoolVFEName");
  backCool.vFEMat = args.str("BackCoolVFEMat");
  backCool.backVFEName = args.str("BackVFEName");
  backCool.backVFEMat = args.str("BackVFEMat");
  backCool.vecBackVFELyrThick = args.vecDble("BackVFELyrThick");
  backCool.vecBackVFELyrName = args.vecStr("BackVFELyrName");
  backCool.vecBackVFELyrMat = args.vecStr("BackVFELyrMat");
  backCool.vecBackCoolNSec = args.vecDble("BackCoolNSec");
  backCool.vecBackCoolSecSep = args.vecDble("BackCoolSecSep");
  backCool.vecBackCoolNPerSec = args.vecDble("BackCoolNPerSec");

  BackMisc backMisc;
  backMisc.here = args.dble("BackMiscHere");
  backMisc.vecThick = args.vecDble("BackMiscThick");
  backMisc.vecName = args.vecStr("BackMiscName");
  backMisc.vecMat = args.vecStr("BackMiscMat");
  backMisc.backCBStdSep = args.dble("BackCBStdSep");

  PatchPanel patchPanel;
  patchPanel.here = args.dble("PatchPanelHere");
  patchPanel.vecThick = args.vecDble("PatchPanelThick");
  patchPanel.vecNames = args.vecStr("PatchPanelNames");
  patchPanel.vecMat = args.vecStr("PatchPanelMat");
  patchPanel.name = args.str("PatchPanelName");

  BackCoolTank backCoolTank;
  backCoolTank.here = args.dble("BackCoolTankHere");
  backCoolTank.name = args.str("BackCoolTankName");
  backCoolTank.width = args.dble("BackCoolTankWidth");
  backCoolTank.thick = args.dble("BackCoolTankThick");
  backCoolTank.mat = args.str("BackCoolTankMat");
  backCoolTank.waName = args.str("BackCoolTankWaName");
  backCoolTank.waWidth = args.dble("BackCoolTankWaWidth");
  backCoolTank.waMat = args.str("BackCoolTankWaMat");
  backCoolTank.backBracketName = args.str("BackBracketName");
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
  mbCoolTube.name = args.str("MBCoolTubeName");
  mbCoolTube.innDiam = args.dble("MBCoolTubeInnDiam");
  mbCoolTube.outDiam = args.dble("MBCoolTubeOutDiam");
  mbCoolTube.mat = args.str("MBCoolTubeMat");

  MBManif mbManif;
  mbManif.here = args.dble("MBManifHere");
  mbManif.name = args.str("MBManifName");
  mbManif.innDiam = args.dble("MBManifInnDiam");
  mbManif.outDiam = args.dble("MBManifOutDiam");
  mbManif.mat = args.str("MBManifMat");

  MBLyr mbLyr;
  mbLyr.here = args.dble("MBLyrHere");
  mbLyr.vecMBLyrThick = args.vecDble("MBLyrThick");
  mbLyr.vecMBLyrName = args.vecStr("MBLyrName");
  mbLyr.vecMBLyrMat = args.vecStr("MBLyrMat");

  Pincer pincer;
  pincer.rodHere = args.dble("PincerRodHere");
  pincer.rodName = args.str("PincerRodName");
  pincer.rodMat = args.str("PincerRodMat");
  pincer.vecRodAzimuth = args.vecDble("PincerRodAzimuth");
  pincer.envName = args.str("PincerEnvName");
  pincer.envMat = args.str("PincerEnvMat");
  pincer.envWidth = args.dble("PincerEnvWidth");
  pincer.envHeight = args.dble("PincerEnvHeight");
  pincer.envLength = args.dble("PincerEnvLength");
  pincer.vecEnvZOff = args.vecDble("PincerEnvZOff");
  pincer.blkName = args.str("PincerBlkName");
  pincer.blkMat = args.str("PincerBlkMat");
  pincer.blkLength = args.dble("PincerBlkLength");
  pincer.shim1Name = args.str("PincerShim1Name");
  pincer.shimHeight = args.dble("PincerShimHeight");
  pincer.shim2Name = args.str("PincerShim2Name");
  pincer.shimMat = args.str("PincerShimMat");
  pincer.shim1Width = args.dble("PincerShim1Width");
  pincer.shim2Width = args.dble("PincerShim2Width");
  pincer.cutName = args.str("PincerCutName");
  pincer.cutMat = args.str("PincerCutMat");
  pincer.cutWidth = args.dble("PincerCutWidth");
  pincer.cutHeight = args.dble("PincerCutHeight");

  return 1;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_ecal_DDEcalBarrelNewAlgo, algorithm)
