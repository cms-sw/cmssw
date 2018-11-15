
//////////////////////////////////////////////////////////////////////////////
// File: DDEcalBarrelAlgo.cc
// Description: Geometry factory class for Ecal Barrel
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "Geometry/EcalCommonData/interface/DDEcalBarrelAlgo.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>

namespace std{} using namespace std;

DDEcalBarrelAlgo::DDEcalBarrelAlgo() :
  m_idNameSpace  ( "" ),
  m_BarName      ( "" ),
  m_BarMat       ( "" ),
  m_vecBarZPts   (    ),
  m_vecBarRMin   (    ),
  m_vecBarRMax   (    ),
  m_vecBarTran   (    ),
  m_vecBarRota   (    ),
  m_vecBarRota2  (    ),
  m_vecBarRota3  (    ),
  m_BarPhiLo     (0),
  m_BarPhiHi     (0),
  m_BarHere      (0),
  m_SpmName      (""),
  m_SpmMat       (""),
  m_vecSpmZPts   (),
  m_vecSpmRMin   (),
  m_vecSpmRMax   (),
  m_vecSpmTran   (),
  m_vecSpmRota   (),
  m_vecSpmBTran  (),
  m_vecSpmBRota  (),
  m_SpmNPerHalf  (0),
  m_SpmLowPhi    (0),
  m_SpmDelPhi    (0),
  m_SpmPhiOff    (0),
  m_vecSpmHere   (),
  m_SpmCutName   (""),
  m_SpmCutThick  (0),
  m_SpmCutShow   (0),
  m_vecSpmCutTM  (),
  m_vecSpmCutTP  (),
  m_SpmCutRM     (0),
  m_SpmCutRP     (0),
  m_SpmExpThick  (0),
  m_SpmExpWide   (0),
  m_SpmExpYOff   (0),
  m_SpmSideName  (""),
  m_SpmSideMat   (""),
  m_SpmSideHigh  (0),
  m_SpmSideThick (0),
  m_SpmSideYOffM (0),
  m_SpmSideYOffP (0),
  m_NomCryDimAF  (0),    
  m_NomCryDimLZ  (0),  
  m_vecNomCryDimBF (),
  m_vecNomCryDimCF (),
  m_vecNomCryDimAR (),
  m_vecNomCryDimBR (),
  m_vecNomCryDimCR (),
  m_UnderAF      (0),  
  m_UnderLZ      (0),
  m_UnderBF      (0),
  m_UnderCF      (0),
  m_UnderAR      (0),
  m_UnderBR      (0),
  m_UnderCR      (0),
  m_WallThAlv    (0),
  m_WrapThAlv    (0),
  m_ClrThAlv     (0),
  m_vecGapAlvEta (),
  m_WallFrAlv    (0),
  m_WrapFrAlv    (0),
  m_ClrFrAlv     (0),
  m_WallReAlv    (0),
  m_WrapReAlv    (0),
  m_ClrReAlv     (0),
  m_NCryTypes    (0),
  m_NCryPerAlvEta (0),
  m_CryName      (""),
  m_ClrName      (""),
  m_WrapName     (""),
  m_WallName     (""),
  m_CryMat       (""),
  m_ClrMat       (""),
  m_WrapMat      (""),
  m_WallMat      (""),

  m_APDName      (""),
  m_APDHere      (0),
  m_APDMat       (""),
  m_APDSide      (0),
  m_APDThick     (0),
  m_APDZ         (0),
  m_APDX1        (0),
  m_APDX2        (0),

  m_WebHere      (0),
  m_WebPlName    (""),    
  m_WebClrName   (""),    
  m_WebPlMat     (""),
  m_WebClrMat    (""),
  m_vecWebPlTh   (),
  m_vecWebClrTh  (),
  m_vecWebLength (),
  m_IlyHere      (0),
  m_IlyName      (),
  m_IlyPhiLow    (0),
  m_IlyDelPhi    (0),
  m_vecIlyMat    (),
  m_vecIlyThick  (),
  m_IlyPipeName      (""),
  m_IlyPipeHere      (0),
  m_IlyPipeMat       (""),
  m_IlyPipeOD        (0),
  m_IlyPipeID        (0),
  m_vecIlyPipeLength (),
  m_vecIlyPipeType   (),
  m_vecIlyPipePhi    (),
  m_vecIlyPipeZ      (),
  m_IlyPTMName        (""),
  m_IlyPTMHere        (0),
  m_IlyPTMMat         (""),
  m_IlyPTMWidth       (0),
  m_IlyPTMLength      (0),
  m_IlyPTMHeight      (0),
  m_vecIlyPTMZ        (),
  m_vecIlyPTMPhi      (),
  m_IlyFanOutName  (""),
  m_IlyFanOutHere  (0),
  m_IlyFanOutMat   (""),
  m_IlyFanOutWidth (0),
  m_IlyFanOutLength(0),
  m_IlyFanOutHeight(0),
  m_vecIlyFanOutZ  (),
  m_vecIlyFanOutPhi(),
  m_IlyDiffName    (""),
  m_IlyDiffMat     (""),
  m_IlyDiffOff     (0),
  m_IlyDiffLength  (0),
  m_IlyBndlName    (""),
  m_IlyBndlMat     (""),
  m_IlyBndlOff     (0),
  m_IlyBndlLength  (0),
  m_IlyFEMName     (""),
  m_IlyFEMMat      (""),
  m_IlyFEMWidth    (0),
  m_IlyFEMLength   (0),
  m_IlyFEMHeight   (0),
  m_vecIlyFEMZ     (),
  m_vecIlyFEMPhi   (),
  m_HawRName     (""),
  m_FawName      (""),
  m_FawHere      (0),
  m_HawRHBIG     (0),
  m_HawRhsml     (0),
  m_HawRCutY     (0),
  m_HawRCutZ     (0),
  m_HawRCutDelY  (0),
  m_HawYOffCry   (0),
  m_NFawPerSupm  (0),
  m_FawPhiOff    (0),
  m_FawDelPhi    (0),
  m_FawPhiRot    (0),
  m_FawRadOff    (0),
  m_GridHere      (0),
  m_GridName     (""),
  m_GridMat      (""),
  m_GridThick    (0),
  m_BackXOff     (0),
  m_BackYOff     (0),
  m_BackHere      (0),
  m_BackSideName          (""),
  m_BackSideHere          (0),
  m_BackSideLength        (0),
  m_BackSideHeight        (0),
  m_BackSideWidth         (0),
  m_BackSideYOff1        (0),
  m_BackSideYOff2        (0),
  m_BackSideAngle        (0),
  m_BackSideMat           (""),
  m_BackPlateName    (""),
  m_BackPlateHere          (0),
  m_BackPlateLength  (0),
  m_BackPlateThick   (0),
  m_BackPlateWidth   (0),
  m_BackPlateMat     (""),
  m_BackPlate2Name    (""),
  m_BackPlate2Thick   (0),
  m_BackPlate2Mat     (""),
  m_GrilleName      (""),
  m_GrilleHere      (0),
  m_GrilleThick     (0),
  m_GrilleWidth     (0),
  m_GrilleZSpace    (0),
  m_GrilleMat       (""),
  m_vecGrilleHeight (),
  m_vecGrilleZOff   (),
  m_GrEdgeSlotName     (""),
  m_GrEdgeSlotMat      (""),
  m_GrEdgeSlotHere     (0),
  m_GrEdgeSlotHeight   (0),
  m_GrEdgeSlotWidth    (0),
  m_GrMidSlotName      (""),
  m_GrMidSlotMat       (""),
  m_GrMidSlotHere      (0),
  m_GrMidSlotWidth     (0),
  m_GrMidSlotXOff      (0),
  m_vecGrMidSlotHeight (),
  m_BackPipeHere      (0),
  m_BackPipeName    (""),
  m_vecBackPipeDiam (),
  m_vecBackPipeThick (),
  m_BackPipeMat     (""),
  m_BackPipeWaterMat (""),

  m_vecBackCoolName       (),
  m_BackCoolHere      (0),
  m_BackCoolBarHere      (0),
  m_BackCoolBarWidth       (0),
  m_BackCoolBarHeight      (0),
  m_BackCoolMat           (""),
  m_BackCoolBarName       (""),
  m_BackCoolBarThick      (0),
  m_BackCoolBarMat        (""),
  m_BackCoolBarSSName     (""),
  m_BackCoolBarSSThick    (0),
  m_BackCoolBarSSMat      (""),
  m_BackCoolBarWaName     (""),
  m_BackCoolBarWaThick    (0),
  m_BackCoolBarWaMat      (""),
  m_BackCoolVFEHere      (0),
  m_BackCoolVFEName       (""),
  m_BackCoolVFEMat        (""),
  m_BackVFEName           (""),
  m_BackVFEMat            (""),
  m_vecBackVFELyrThick    (),
  m_vecBackVFELyrName     (),
  m_vecBackVFELyrMat      (),
  m_vecBackCoolNSec       (),
  m_vecBackCoolSecSep     (),
  m_vecBackCoolNPerSec    (),     

  m_BackMiscHere      (0),
  m_vecBackMiscThick (),
  m_vecBackMiscName  (),
  m_vecBackMiscMat   (),
  m_BackCBStdSep         (0),
  m_PatchPanelHere      (0),
  m_PatchPanelName   (""),
  m_vecPatchPanelThick (),
  m_vecPatchPanelNames  (),
  m_vecPatchPanelMat   (),
  m_BackCoolTankHere      (0),
  m_BackCoolTankName    (""),
  m_BackCoolTankWidth   (0),
  m_BackCoolTankThick   (0),
  m_BackCoolTankMat     (""),
  m_BackCoolTankWaName  (""),
  m_BackCoolTankWaWidth (0),
  m_BackCoolTankWaMat   (""),
  m_BackBracketName     (""),
  m_BackBracketHeight   (0),
  m_BackBracketMat      (""),

  m_DryAirTubeHere      (0),
  m_DryAirTubeName      (""),
  m_MBCoolTubeNum       (0),
  m_DryAirTubeInnDiam   (0),
  m_DryAirTubeOutDiam   (0),
  m_DryAirTubeMat       (""),
  m_MBCoolTubeHere      (0),
  m_MBCoolTubeName      (""),
  m_MBCoolTubeInnDiam   (0),
  m_MBCoolTubeOutDiam   (0),
  m_MBCoolTubeMat       (""),
  m_MBManifHere      (0),
  m_MBManifName         (""),
  m_MBManifInnDiam      (0),
  m_MBManifOutDiam      (0),
  m_MBManifMat          (""),
  m_MBLyrHere      (0),
  m_vecMBLyrThick       (0),
  m_vecMBLyrName        (),
  m_vecMBLyrMat         (),


  m_PincerRodHere      (0),
  m_PincerRodName      (""),
  m_PincerRodMat       (""),
  m_vecPincerRodAzimuth(),
  m_PincerEnvName      (""),
  m_PincerEnvMat       (""),
  m_PincerEnvWidth     (0),
  m_PincerEnvHeight    (0),
  m_PincerEnvLength    (0),
  m_vecPincerEnvZOff   (),
  m_PincerBlkName      (""),
  m_PincerBlkMat       (""),
  m_PincerBlkLength    (0),
  m_PincerShim1Name    (""),
  m_PincerShimHeight   (0),
  m_PincerShim2Name    (""),
  m_PincerShimMat      (""),
  m_PincerShim1Width   (0),
  m_PincerShim2Width   (0),
  m_PincerCutName      (""),
  m_PincerCutMat       (""),
  m_PincerCutWidth    (0),
  m_PincerCutHeight    (0)

{
   LogDebug("EcalGeom") << "DDEcalBarrelAlgo info: Creating an instance" ;
}

DDEcalBarrelAlgo::~DDEcalBarrelAlgo() {}




void DDEcalBarrelAlgo::initialize(const DDNumericArguments      & nArgs,
				  const DDVectorArguments       & vArgs,
				  const DDMapArguments          & /*mArgs*/,
				  const DDStringArguments       & sArgs,
				  const DDStringVectorArguments & vsArgs) {

   LogDebug("EcalGeom") << "DDEcalBarrelAlgo info: Initialize" ;
   m_idNameSpace = DDCurrentNamespace::ns();
   // TRICK!
   m_idNameSpace = parent().name().ns();
   // barrel parent volume
   m_BarName     = sArgs["BarName" ] ;
   m_BarMat      = sArgs["BarMat"  ] ;
   m_vecBarZPts  = vArgs["BarZPts" ] ;
   m_vecBarRMin  = vArgs["BarRMin" ] ;
   m_vecBarRMax  = vArgs["BarRMax" ] ;
   m_vecBarTran  = vArgs["BarTran" ] ;
   m_vecBarRota  = vArgs["BarRota" ] ;
   m_vecBarRota2 = vArgs["BarRota2" ] ;
   m_vecBarRota3 = vArgs["BarRota3" ] ;
   m_BarPhiLo    = nArgs["BarPhiLo" ] ;
   m_BarPhiHi    = nArgs["BarPhiHi" ] ;
   m_BarHere     = nArgs["BarHere" ] ;

   m_SpmName     = sArgs["SpmName"] ;
   m_SpmMat      = sArgs["SpmMat" ] ;
   m_vecSpmZPts  = vArgs["SpmZPts"] ;
   m_vecSpmRMin  = vArgs["SpmRMin"] ;
   m_vecSpmRMax  = vArgs["SpmRMax"] ;
   m_vecSpmTran  = vArgs["SpmTran"] ;
   m_vecSpmRota  = vArgs["SpmRota"] ;
   m_vecSpmBTran = vArgs["SpmBTran"] ;
   m_vecSpmBRota = vArgs["SpmBRota"] ;
   m_SpmNPerHalf = static_cast<unsigned int> (nArgs["SpmNPerHalf"]) ;
   m_SpmLowPhi   = nArgs["SpmLowPhi"] ;
   m_SpmDelPhi   = nArgs["SpmDelPhi"] ;
   m_SpmPhiOff   = nArgs["SpmPhiOff"] ;
   m_vecSpmHere  = vArgs["SpmHere"] ;
   m_SpmCutName  = sArgs["SpmCutName"] ;
   m_SpmCutThick = nArgs["SpmCutThick"] ;
   m_SpmCutShow  = int(nArgs["SpmCutShow"]) ;
   m_vecSpmCutTM = vArgs["SpmCutTM"] ;
   m_vecSpmCutTP = vArgs["SpmCutTP"] ;
   m_SpmCutRM    = nArgs["SpmCutRM"] ;
   m_SpmCutRP    = nArgs["SpmCutRP"] ;
   m_SpmExpThick = nArgs["SpmExpThick"] ;
   m_SpmExpWide  = nArgs["SpmExpWide"] ;
   m_SpmExpYOff  = nArgs["SpmExpYOff"] ;
   m_SpmSideName = sArgs["SpmSideName"] ;
   m_SpmSideMat  = sArgs["SpmSideMat" ] ;
   m_SpmSideHigh = nArgs["SpmSideHigh"] ;
   m_SpmSideThick= nArgs["SpmSideThick"] ;
   m_SpmSideYOffM= nArgs["SpmSideYOffM"] ;
   m_SpmSideYOffP= nArgs["SpmSideYOffP"] ;

   m_NomCryDimAF    = nArgs["NomCryDimAF"] ;
   m_NomCryDimLZ    = nArgs["NomCryDimLZ"] ;
   m_vecNomCryDimBF = vArgs["NomCryDimBF"] ;
   m_vecNomCryDimCF = vArgs["NomCryDimCF"] ;
   m_vecNomCryDimAR = vArgs["NomCryDimAR"] ;
   m_vecNomCryDimBR = vArgs["NomCryDimBR"] ;
   m_vecNomCryDimCR = vArgs["NomCryDimCR"] ;

   m_UnderAF = nArgs["UnderAF"] ; 
   m_UnderLZ = nArgs["UnderLZ"] ; 
   m_UnderBF = nArgs["UnderBF"] ; 
   m_UnderCF = nArgs["UnderCF"] ; 
   m_UnderAR = nArgs["UnderAR"] ; 
   m_UnderBR = nArgs["UnderBR"] ; 
   m_UnderCR = nArgs["UnderCR"] ; 

   m_WallThAlv = nArgs["WallThAlv"] ;
   m_WrapThAlv = nArgs["WrapThAlv"] ;
   m_ClrThAlv  = nArgs["ClrThAlv"] ;
   m_vecGapAlvEta = vArgs["GapAlvEta"] ;

   m_WallFrAlv = nArgs["WallFrAlv"] ;
   m_WrapFrAlv = nArgs["WrapFrAlv"] ;
   m_ClrFrAlv  = nArgs["ClrFrAlv"] ;

   m_WallReAlv = nArgs["WallReAlv"] ;
   m_WrapReAlv = nArgs["WrapReAlv"] ;
   m_ClrReAlv  = nArgs["ClrReAlv"] ;

   m_NCryTypes     = static_cast<unsigned int> ( nArgs["NCryTypes"] ) ;
   m_NCryPerAlvEta = static_cast<unsigned int> ( nArgs["NCryPerAlvEta"] ) ;

   m_CryName  = sArgs["CryName"] ;
   m_ClrName  = sArgs["ClrName"] ;
   m_WrapName = sArgs["WrapName"] ; 
   m_WallName = sArgs["WallName"] ; 

   m_CryMat  = sArgs["CryMat"] ; 
   m_ClrMat  = sArgs["ClrMat"] ; 
   m_WrapMat = sArgs["WrapMat"] ; 
   m_WallMat = sArgs["WallMat"] ;

   m_APDName  = sArgs["APDName"] ;
   m_APDHere  = nArgs["APDHere"] ;
   m_APDMat   = sArgs["APDMat"] ;
   m_APDSide  = nArgs["APDSide"] ;
   m_APDThick = nArgs["APDThick"] ;
   m_APDZ     = nArgs["APDZ"] ;
   m_APDX1    = nArgs["APDX1"] ;
   m_APDX2    = nArgs["APDX2"] ;

   m_WebHere     = nArgs["WebHere"] ;
   m_WebPlName   = sArgs["WebPlName"] ;
   m_WebClrName  = sArgs["WebClrName"] ;
   m_WebPlMat    = sArgs["WebPlMat"] ;
   m_WebClrMat   = sArgs["WebClrMat"] ;
   m_vecWebPlTh  = vArgs["WebPlTh"] ;
   m_vecWebClrTh = vArgs["WebClrTh"] ;
   m_vecWebLength= vArgs["WebLength"] ;

   m_IlyHere     = nArgs["IlyHere"] ;
   m_IlyName     = sArgs["IlyName"] ;
   m_IlyPhiLow   = nArgs["IlyPhiLow"] ;
   m_IlyDelPhi   = nArgs["IlyDelPhi"] ;
   m_vecIlyMat   = vsArgs["IlyMat"] ;
   m_vecIlyThick = vArgs["IlyThick"] ;

   m_IlyPipeName      = sArgs["IlyPipeName"] ;
   m_IlyPipeHere      = nArgs["IlyPipeHere"] ;
   m_IlyPipeMat       = sArgs["IlyPipeMat"] ;
   m_IlyPipeOD        = nArgs["IlyPipeOD"] ;
   m_IlyPipeID        = nArgs["IlyPipeID"] ;
   m_vecIlyPipeLength = vArgs["IlyPipeLength"] ;
   m_vecIlyPipeType   = vArgs["IlyPipeType"] ;
   m_vecIlyPipePhi    = vArgs["IlyPipePhi"] ;
   m_vecIlyPipeZ      = vArgs["IlyPipeZ"] ;

   m_IlyPTMName   = sArgs["IlyPTMName"] ;
   m_IlyPTMHere   = nArgs["IlyPTMHere"] ;
   m_IlyPTMMat    = sArgs["IlyPTMMat"] ;
   m_IlyPTMWidth  = nArgs["IlyPTMWidth"] ;
   m_IlyPTMLength = nArgs["IlyPTMLength"] ;
   m_IlyPTMHeight = nArgs["IlyPTMHeight"] ;
   m_vecIlyPTMZ   = vArgs["IlyPTMZ"] ;
   m_vecIlyPTMPhi = vArgs["IlyPTMPhi"] ;

   m_IlyFanOutName   = sArgs["IlyFanOutName"] ;
   m_IlyFanOutHere   = nArgs["IlyFanOutHere"] ;
   m_IlyFanOutMat    = sArgs["IlyFanOutMat"] ;
   m_IlyFanOutWidth  = nArgs["IlyFanOutWidth"] ;
   m_IlyFanOutLength = nArgs["IlyFanOutLength"] ;
   m_IlyFanOutHeight = nArgs["IlyFanOutHeight"] ;
   m_vecIlyFanOutZ   = vArgs["IlyFanOutZ"] ;
   m_vecIlyFanOutPhi = vArgs["IlyFanOutPhi"] ;
   m_IlyDiffName     = sArgs["IlyDiffName"] ;
   m_IlyDiffMat      = sArgs["IlyDiffMat"] ;
   m_IlyDiffOff      = nArgs["IlyDiffOff"] ;
   m_IlyDiffLength   = nArgs["IlyDiffLength"] ;
   m_IlyBndlName     = sArgs["IlyBndlName"] ;
   m_IlyBndlMat      = sArgs["IlyBndlMat"] ;
   m_IlyBndlOff      = nArgs["IlyBndlOff"] ;
   m_IlyBndlLength   = nArgs["IlyBndlLength"] ;
   m_IlyFEMName      = sArgs["IlyFEMName"] ;
   m_IlyFEMMat       = sArgs["IlyFEMMat"] ;
   m_IlyFEMWidth     = nArgs["IlyFEMWidth"] ;
   m_IlyFEMLength    = nArgs["IlyFEMLength"] ;
   m_IlyFEMHeight    = nArgs["IlyFEMHeight"] ;
   m_vecIlyFEMZ      = vArgs["IlyFEMZ"] ;
   m_vecIlyFEMPhi    = vArgs["IlyFEMPhi"];

   m_HawRName   = sArgs["HawRName"] ;
   m_FawName    = sArgs["FawName"] ;
   m_FawHere    = nArgs["FawHere"] ;
   m_HawRHBIG   = nArgs["HawRHBIG"] ;
   m_HawRhsml   = nArgs["HawRhsml"] ;
   m_HawRCutY   = nArgs["HawRCutY"] ;
   m_HawRCutZ   = nArgs["HawRCutZ"] ;
   m_HawRCutDelY= nArgs["HawRCutDelY"] ;
   m_HawYOffCry = nArgs["HawYOffCry"] ;

   m_NFawPerSupm=  static_cast<unsigned int> ( nArgs["NFawPerSupm"] ) ;
   m_FawPhiOff  = nArgs["FawPhiOff"] ;
   m_FawDelPhi  = nArgs["FawDelPhi"] ;
   m_FawPhiRot  = nArgs["FawPhiRot"] ;
   m_FawRadOff  = nArgs["FawRadOff"] ;

   m_GridHere     = nArgs["GridHere"] ;
   m_GridName   = sArgs["GridName"]  ;
   m_GridMat    = sArgs["GridMat"]   ;
   m_GridThick  = nArgs["GridThick"] ;

   m_BackHere         = nArgs["BackHere"] ;
   m_BackXOff         = nArgs["BackXOff"] ;
   m_BackYOff         = nArgs["BackYOff"] ;
   m_BackSideName     = sArgs["BackSideName"] ;
   m_BackSideHere     = nArgs["BackSideHere"] ;
   m_BackSideLength   = nArgs["BackSideLength"] ;
   m_BackSideHeight   = nArgs["BackSideHeight"] ;
   m_BackSideWidth    = nArgs["BackSideWidth"] ;
   m_BackSideYOff1    = nArgs["BackSideYOff1"] ;
   m_BackSideYOff2    = nArgs["BackSideYOff2"] ;
   m_BackSideAngle    = nArgs["BackSideAngle"] ;
   m_BackSideMat      = sArgs["BackSideMat"] ;
   m_BackPlateName    = sArgs["BackPlateName"] ;
   m_BackPlateHere    = nArgs["BackPlateHere"] ;
   m_BackPlateLength  = nArgs["BackPlateLength"] ;
   m_BackPlateThick   = nArgs["BackPlateThick"] ;
   m_BackPlateWidth   = nArgs["BackPlateWidth"] ;
   m_BackPlateMat     = sArgs["BackPlateMat"] ;
   m_BackPlate2Name    = sArgs["BackPlate2Name"] ;
   m_BackPlate2Thick   = nArgs["BackPlate2Thick"] ;
   m_BackPlate2Mat     = sArgs["BackPlate2Mat"] ;
   m_GrilleName       = sArgs["GrilleName"] ;
   m_GrilleHere       = nArgs["GrilleHere"] ;
   m_GrilleThick      = nArgs["GrilleThick"] ;
   m_GrilleWidth      = nArgs["GrilleWidth"] ;
   m_GrilleZSpace     = nArgs["GrilleZSpace"] ;
   m_GrilleMat        = sArgs["GrilleMat"] ;
   m_vecGrilleHeight  = vArgs["GrilleHeight"] ;
   m_vecGrilleZOff    = vArgs["GrilleZOff"] ;

   m_GrEdgeSlotName     = sArgs["GrEdgeSlotName"] ;
   m_GrEdgeSlotMat      = sArgs["GrEdgeSlotMat"] ;
   m_GrEdgeSlotHere     = nArgs["GrEdgeSlotHere"] ;
   m_GrEdgeSlotHeight   = nArgs["GrEdgeSlotHeight"] ;
   m_GrEdgeSlotWidth    = nArgs["GrEdgeSlotWidth"] ;
   m_GrMidSlotName      = sArgs["GrMidSlotName"] ;
   m_GrMidSlotMat       = sArgs["GrMidSlotMat"] ;
   m_GrMidSlotHere      = nArgs["GrMidSlotHere"] ;
   m_GrMidSlotWidth     = nArgs["GrMidSlotWidth"] ;
   m_GrMidSlotXOff      = nArgs["GrMidSlotXOff"] ;
   m_vecGrMidSlotHeight = vArgs["GrMidSlotHeight"] ;

   m_BackPipeHere     = nArgs["BackPipeHere"] ;
   m_BackPipeName    = sArgs["BackPipeName"] ;
   m_vecBackPipeDiam = vArgs["BackPipeDiam"] ;
   m_vecBackPipeThick = vArgs["BackPipeThick"] ;
   m_BackPipeMat     = sArgs["BackPipeMat"] ;
   m_BackPipeWaterMat = sArgs["BackPipeWaterMat"] ;


   m_BackCoolHere          = nArgs["BackCoolHere"] ;
   m_vecBackCoolName       = vsArgs["BackCoolName"] ;
   m_BackCoolBarHere       = nArgs["BackCoolBarHere"] ;
   m_BackCoolBarWidth      = nArgs["BackCoolBarWidth"] ; 
   m_BackCoolBarHeight     = nArgs["BackCoolBarHeight"] ; 
   m_BackCoolMat           = sArgs["BackCoolMat"] ;     
   m_BackCoolBarName       = sArgs["BackCoolBarName"] ;      
   m_BackCoolBarThick      = nArgs["BackCoolBarThick"] ;  
   m_BackCoolBarMat        = sArgs["BackCoolBarMat"] ; 
   m_BackCoolBarSSName     = sArgs["BackCoolBarSSName"] ;   
   m_BackCoolBarSSThick    = nArgs["BackCoolBarSSThick"] ;
   m_BackCoolBarSSMat      = sArgs["BackCoolBarSSMat"] ;
   m_BackCoolBarWaName     = sArgs["BackCoolBarWaName"] ; 
   m_BackCoolBarWaThick    = nArgs["BackCoolBarWaThick"] ;
   m_BackCoolBarWaMat      = sArgs["BackCoolBarWaMat"] ;
   m_BackCoolVFEHere       = nArgs["BackCoolVFEHere"] ;
   m_BackCoolVFEName       = sArgs["BackCoolVFEName"] ; 
   m_BackCoolVFEMat        = sArgs["BackCoolVFEMat"] ;  
   m_BackVFEName           = sArgs["BackVFEName"] ;   
   m_BackVFEMat            = sArgs["BackVFEMat"] ;      
   m_vecBackVFELyrThick    = vArgs["BackVFELyrThick"] ;   
   m_vecBackVFELyrName     = vsArgs["BackVFELyrName"] ;
   m_vecBackVFELyrMat      = vsArgs["BackVFELyrMat"] ;
   m_vecBackCoolNSec       = vArgs["BackCoolNSec"] ; 
   m_vecBackCoolSecSep     = vArgs["BackCoolSecSep"] ;  
   m_vecBackCoolNPerSec    = vArgs["BackCoolNPerSec"] ;  
   m_BackCBStdSep          = nArgs["BackCBStdSep"] ;

   m_BackMiscHere       = nArgs["BackMiscHere"] ;
   m_vecBackMiscThick   = vArgs["BackMiscThick"] ;
   m_vecBackMiscName    = vsArgs["BackMiscName"] ;
   m_vecBackMiscMat     = vsArgs["BackMiscMat"] ;
   m_PatchPanelHere     = nArgs["PatchPanelHere"] ;
   m_vecPatchPanelThick = vArgs["PatchPanelThick"] ;
   m_vecPatchPanelNames = vsArgs["PatchPanelNames"] ;
   m_vecPatchPanelMat   = vsArgs["PatchPanelMat"] ;
   m_PatchPanelName     = sArgs["PatchPanelName"] ;

   m_BackCoolTankHere    = nArgs["BackCoolTankHere"] ;
   m_BackCoolTankName    = sArgs["BackCoolTankName"] ;
   m_BackCoolTankWidth   = nArgs["BackCoolTankWidth"] ;
   m_BackCoolTankThick   = nArgs["BackCoolTankThick"] ;
   m_BackCoolTankMat     = sArgs["BackCoolTankMat"] ;
   m_BackCoolTankWaName  = sArgs["BackCoolTankWaName"] ;
   m_BackCoolTankWaWidth = nArgs["BackCoolTankWaWidth"] ;
   m_BackCoolTankWaMat   = sArgs["BackCoolTankWaMat"] ;
   m_BackBracketName     = sArgs["BackBracketName"] ;
   m_BackBracketHeight   = nArgs["BackBracketHeight"] ;
   m_BackBracketMat      = sArgs["BackBracketMat"] ;

   m_DryAirTubeHere     = nArgs["DryAirTubeHere"] ;
   m_DryAirTubeName     = sArgs["DryAirTubeName"];
   m_MBCoolTubeNum      = static_cast<unsigned int> ( nArgs["MBCoolTubeNum"] ) ;
   m_DryAirTubeInnDiam  = nArgs["DryAirTubeInnDiam"];
   m_DryAirTubeOutDiam  = nArgs["DryAirTubeOutDiam"];
   m_DryAirTubeMat      = sArgs["DryAirTubeMat"];
   m_MBCoolTubeHere     = nArgs["MBCoolTubeHere"] ;
   m_MBCoolTubeName     = sArgs["MBCoolTubeName"];
   m_MBCoolTubeInnDiam  = nArgs["MBCoolTubeInnDiam"];
   m_MBCoolTubeOutDiam  = nArgs["MBCoolTubeOutDiam"];
   m_MBCoolTubeMat      = sArgs["MBCoolTubeMat"];
   m_MBManifHere        = nArgs["MBManifHere"] ;
   m_MBManifName        = sArgs["MBManifName"];
   m_MBManifInnDiam     = nArgs["MBManifInnDiam"];
   m_MBManifOutDiam     = nArgs["MBManifOutDiam"];
   m_MBManifMat         = sArgs["MBManifMat"];
   m_MBLyrHere          = nArgs["MBLyrHere"] ;
   m_vecMBLyrThick      = vArgs["MBLyrThick"];
   m_vecMBLyrName       = vsArgs["MBLyrName"];
   m_vecMBLyrMat        = vsArgs["MBLyrMat"];

   m_PincerRodHere      = nArgs["PincerRodHere"];
   m_PincerRodName      = sArgs["PincerRodName"];
   m_PincerRodMat       = sArgs["PincerRodMat"];
   m_vecPincerRodAzimuth= vArgs["PincerRodAzimuth"];
   m_PincerEnvName      = sArgs["PincerEnvName"];
   m_PincerEnvMat       = sArgs["PincerEnvMat"];
   m_PincerEnvWidth     = nArgs["PincerEnvWidth"];
   m_PincerEnvHeight    = nArgs["PincerEnvHeight"];
   m_PincerEnvLength    = nArgs["PincerEnvLength"];
   m_vecPincerEnvZOff   = vArgs["PincerEnvZOff"];
   m_PincerBlkName      = sArgs["PincerBlkName"];
   m_PincerBlkMat       = sArgs["PincerBlkMat"];
   m_PincerBlkLength    = nArgs["PincerBlkLength"];
   m_PincerShim1Name    = sArgs["PincerShim1Name"];
   m_PincerShimHeight   = nArgs["PincerShimHeight"];
   m_PincerShim2Name    = sArgs["PincerShim2Name"];
   m_PincerShimMat      = sArgs["PincerShimMat"];
   m_PincerShim1Width   = nArgs["PincerShim1Width"];
   m_PincerShim2Width   = nArgs["PincerShim2Width"];
   m_PincerCutName      = sArgs["PincerCutName"];
   m_PincerCutMat       = sArgs["PincerCutMat"];
   m_PincerCutWidth     = nArgs["PincerCutWidth"];
   m_PincerCutHeight    = nArgs["PincerCutHeight"];

   
   LogDebug("EcalGeom") << "DDEcalBarrelAlgo info: end initialize" ;
}

////////////////////////////////////////////////////////////////////
// DDEcalBarrelAlgo methods...
////////////////////////////////////////////////////////////////////

void DDEcalBarrelAlgo::execute(DDCompactView& cpv) 
{
   LogDebug("EcalGeom") << "******** DDEcalBarrelAlgo execute!" << std::endl ;

   if( barHere() != 0 )
   {
      const unsigned int copyOne (1) ;
      const unsigned int copyTwo (2) ;
      // Barrel parent volume----------------------------------------------------------
      cpv.position( DDLogicalPart( barName(), barMat(), 
			    DDSolidFactory::polycone(
			       barName(), barPhiLo(), ( barPhiHi() - barPhiLo() ), 
			       vecBarZPts(), vecBarRMin(), vecBarRMax())),
	     parent().name() , 
	     copyOne, 
	     DDTranslation(vecBarTran()[0],
			   vecBarTran()[1],
			   vecBarTran()[2]), 
	     myrot(barName().name()+"Rot",
		   Rota(Vec3(vecBarRota3()[0],
			     vecBarRota3()[1],
			     vecBarRota3()[2]),
			vecBarRota3()[3])*
		   Rota(Vec3(vecBarRota2()[0],
			     vecBarRota2()[1],
			     vecBarRota2()[2]),
			vecBarRota2()[3])*
		   Rota(Vec3(vecBarRota()[0],
			     vecBarRota()[1],
			     vecBarRota()[2]),
			vecBarRota()[3]) ) ) ;
      // End Barrel parent volume----------------------------------------------------------


      // Supermodule parent------------------------------------------------------------

      const DDName spmcut1ddname ( ( 0 != spmCutShow() ) ?
				   spmName() : ddname( m_SpmName + "CUT1" ) ) ;
      const DDSolid ddspm ( DDSolidFactory::polycone(
			       spmcut1ddname,
			       spmLowPhi(),  spmDelPhi(),
			       vecSpmZPts(), vecSpmRMin(), vecSpmRMax())) ;

      const unsigned int indx ( vecSpmRMax().size()/2 ) ;


      // Deal with the cut boxes first
      const DDSolid spmCutBox ( DDSolidFactory::box(
				   spmCutName(), 
				   1.05*(vecSpmRMax()[indx] - vecSpmRMin()[indx])/2.,  
				   spmCutThick()/2.,
				   fabs( vecSpmZPts().back() - vecSpmZPts().front() )/2.+1*mm) ) ;
      const std::vector<double>& cutBoxParms ( spmCutBox.parameters() ) ;
      const DDLogicalPart spmCutLog ( spmCutName(), spmMat(), spmCutBox ) ;

      // Now the expansion box
      const double xExp ( spmExpThick()/2. ) ;
      const double yExp ( spmExpWide()/2. ) ;
      const double zExp ( fabs( vecSpmZPts().back() -
				vecSpmZPts().front() )/2.) ;
      const DDName expName ( m_SpmName + "EXP" ) ;
      const DDSolid spmExpBox ( DDSolidFactory::box(
				   expName , 
				   xExp    ,  
				   yExp    ,
				   zExp     )) ;
      const DDTranslation expTra ( vecSpmRMax().back() - xExp, spmExpYOff(),
				   vecSpmZPts().front() + zExp ) ;
      const DDLogicalPart expLog ( expName, spmMat(), spmExpBox ) ;

      // Supermodule side platess
      const DDSolid sideSolid ( DDSolidFactory::box(
				   spmSideName(), 
				   spmSideHigh()/2.,  
				   spmSideThick()/2.,
				   fabs( vecSpmZPts()[1] - vecSpmZPts()[0] )/2.) ) ;
      const std::vector<double>& sideParms ( sideSolid.parameters() ) ;
      const DDLogicalPart sideLog ( spmSideName(), spmSideMat(), sideSolid ) ;
      
      DDSolid temp1 ;
      DDSolid temp2 ;
      for( unsigned int icopy(1); icopy <= 2; ++icopy )
      {
	 const std::vector<double>& tvec ( 1==icopy ? vecSpmCutTM() : vecSpmCutTP() ) ;
	 const double rang               ( 1==icopy ? spmCutRM() : spmCutRP() ) ;
	 
	 const Tl3D tr ( tvec[0], tvec[1], tvec[2] );
	 const RoZ3D   ro ( rang ) ;
	 const Tf3D alltrot (
	    RoZ3D( 1==icopy ? spmLowPhi() : spmLowPhi()+spmDelPhi() )*
	    Tl3D( ( vecSpmRMax()[indx]+vecSpmRMin()[indx])/2.,
		  0,
		  (vecSpmZPts().front()+ vecSpmZPts().back())/2.)*
	    tr*ro) ;

	 const DDRotation    ddrot ( myrot(spmCutName().name() + 
					   std::to_string(icopy),
					   alltrot.getRotation() ) ) ;
	 const DDTranslation ddtra ( alltrot.getTranslation() ) ;

	 
	 const Tl3D trSide ( tvec[0], 
			     tvec[1] + ( 1==icopy ? 1. : -1. )*( cutBoxParms[1]+sideParms[1] )
			     + ( 1==icopy ? spmSideYOffM() : spmSideYOffP() ), 
			     tvec[2] );
	 const RoZ3D   roSide ( rang ) ;
	 const Tf3D sideRot (
	    RoZ3D( 1==icopy ? spmLowPhi() : spmLowPhi()+spmDelPhi() )*
	    Tl3D( vecSpmRMin().front() + sideParms[0] ,
		  0,
		  vecSpmZPts().front()+ sideParms[2] )*
	    trSide*roSide) ;

	 const DDRotation    sideddrot ( myrot(spmSideName().name() + 
					       std::to_string(icopy),
					       sideRot.getRotation() ) ) ;
	 const DDTranslation sideddtra ( sideRot.getTranslation() ) ;

	 cpv.position( sideLog,
		spmName(), 
		icopy, 
		sideddtra,
		sideddrot ) ;


	 if( 0 != spmCutShow() ) // do this if we are "showing" the boxes
	 {
	    cpv.position( spmCutLog,
		   spmName(), 
		   icopy, 
		   ddtra,
		   ddrot ) ;
	 }
	 else // do this if we are subtracting the boxes
	 {
	    if( 1 == icopy )
	    {
	       temp1 = DDSolidFactory::subtraction( DDName( m_SpmName+"_T1" ),
						    spmcut1ddname, spmCutBox,
						    ddtra, ddrot ) ;
	    }
	    else
	    {
	       temp2 = DDSolidFactory::subtraction( spmName(),
						    temp1, spmCutBox,
						    ddtra, ddrot ) ;
	    }
	 }
      }

      const DDLogicalPart spmLog (spmName(), spmMat(), 
				  ((0 != spmCutShow()) ? ddspm : temp2)  ) ;

      const double dphi ( 360.*deg/(1.*spmNPerHalf() ) ) ;
      for( unsigned int iphi (0); iphi<2*spmNPerHalf() ; ++iphi ) 
      {
	 const double phi ( iphi*dphi + spmPhiOff() ) ; //- 0.000130/deg ) ;

	 // this base rotation includes the base translation & rotation
	 // plus flipping for the negative z hemisphere, plus
	 // the phi rotation for this module
	 const Tf3D rotaBase ( RoZ3D( phi )*
			       ( iphi < spmNPerHalf() ? Ro3D() :
				 RoX3D( 180.*deg ) )*
			       Ro3D( vecSpmBRota()[3],
				     Vec3( vecSpmBRota()[0],
					   vecSpmBRota()[1],
					   vecSpmBRota()[2]))*
			       Tl3D( Vec3( vecSpmBTran()[0],
					   vecSpmBTran()[1],
					   vecSpmBTran()[2] ))) ;

	 // here the individual rotations & translations of the supermodule
	 // are implemented on top of the overall "base" rotation & translation

	 const unsigned int offr ( 4*iphi ) ;
	 const unsigned int offt ( 3*iphi ) ;

	 const Ro3D r1 ( vecSpmRota()[     offr+3], 
			 Vec3(vecSpmRota()[offr+0],
			      vecSpmRota()[offr+1],
			      vecSpmRota()[offr+2]  ) ) ;

	 const Tf3D rotaExtra ( r1*Tl3D( Vec3(vecSpmTran()[offt+0],
					      vecSpmTran()[offt+1],
					      vecSpmTran()[offt+2]  ) ) ) ;

	 const Tf3D both ( rotaExtra*rotaBase ) ;

	 const DDRotation rota ( myrot( spmName().name()+std::to_string(phi/deg), 
					both.getRotation() ) );

	 if( vecSpmHere()[iphi] != 0 )
	 {
	   // convert from CLHEP to DDTranslation & etc. -- Michael Case
	   DDTranslation myTran(both.getTranslation().x(), both.getTranslation().y(), both.getTranslation().z());
	    cpv.position( spmLog,
		   barName(), 
		   iphi+1, 
		   myTran,
		   rota                     ) ;
	 }
      } 
      // End Supermodule parent------------------------------------------------------------

      // Begin Inner Layer volumes---------------------------------------------------------
      const double  ilyLength  ( vecSpmZPts()[1] - vecSpmZPts()[0] ) ;
      double        ilyRMin    ( vecSpmRMin()[0] ) ;
      double        ilyThick   ( 0 ) ;
      for( unsigned int ilyx ( 0 ) ; ilyx != vecIlyThick().size() ; ++ilyx )
      {
	 ilyThick += vecIlyThick()[ilyx] ;
      }
      const DDName        ilyDDName  ( ddname( ilyName() ) ) ;
      const DDSolid       ilySolid   ( DDSolidFactory::tubs( ilyDDName,
							     ilyLength/2,
							     ilyRMin,
							     ilyRMin + ilyThick,
							     ilyPhiLow(),
							     ilyDelPhi() ) ) ;
      const DDLogicalPart ilyLog     ( ilyDDName, spmMat(), ilySolid ) ;
      cpv.position( ilyLog,
	     spmLog, 
	     copyOne, 
	     DDTranslation(0,0, ilyLength/2 ),
	     DDRotation() ) ;

      DDLogicalPart ilyPipeLog[200] ;

      if( 0 != ilyPipeHere() )
      {
	 for( unsigned int iPipeType ( 0 ) ; iPipeType != vecIlyPipeLength().size(); ++iPipeType )
	 {
	    const DDName pName ( ddname( ilyPipeName() + "_" + 
					 std::to_string(iPipeType+1) ) ) ;
	 
	    DDSolid ilyPipeSolid ( DDSolidFactory::tubs( pName ,
							 vecIlyPipeLength()[iPipeType]/2.,
							 0,
							 ilyPipeOD()/2,
							 0*deg, 360*deg ) ) ;
	    ilyPipeLog[iPipeType] = DDLogicalPart( pName, ilyPipeMat(), ilyPipeSolid ) ;
		  
	    const DDName pWaName ( ddname( ilyPipeName() + "Wa_" + 
					   std::to_string(iPipeType+1) ) ) ;
	    DDSolid ilyPipeWaSolid ( DDSolidFactory::tubs( pWaName ,
							   vecIlyPipeLength()[iPipeType]/2.,
							   0, 
							   ilyPipeID()/2,
							   0*deg, 360*deg ) ) ;
	    const DDLogicalPart ilyPipeWaLog ( pWaName, backPipeWaterMat(), ilyPipeWaSolid ) ;

	    cpv.position( ilyPipeWaLog,
		   pName, 
		   copyOne, 
		   DDTranslation(0,0,0),
		   DDRotation() ) ;
	 }
      } 

      DDSolid ilyPTMSolid ( DDSolidFactory::box( ilyPTMName(), 
						 ilyPTMHeight()/2.,
						 ilyPTMWidth()/2.,  
						 ilyPTMLength()/2.   ) ) ;
      const DDLogicalPart ilyPTMLog ( ilyPTMName(),
				      ilyPTMMat(),
				      ilyPTMSolid ) ;

      DDSolid ilyFanOutSolid ( DDSolidFactory::box( ilyFanOutName(), 
						    ilyFanOutHeight()/2.,
						    ilyFanOutWidth()/2.,  
						    ilyFanOutLength()/2.   ) ) ;
      const DDLogicalPart ilyFanOutLog ( ilyFanOutName(),
					 ilyFanOutMat(),
					 ilyFanOutSolid ) ;

      DDSolid ilyFEMSolid ( DDSolidFactory::box( ilyFEMName(), 
						 ilyFEMHeight()/2.,
						 ilyFEMWidth()/2.,  
						 ilyFEMLength()/2.   ) ) ;
      const DDLogicalPart ilyFEMLog ( ilyFEMName(),
				      ilyFEMMat(),
				      ilyFEMSolid ) ;

      DDSolid ilyDiffSolid ( DDSolidFactory::box( ilyDiffName(), 
						  ilyFanOutHeight()/2.,
						  ilyFanOutWidth()/2.,  
						  ilyDiffLength()/2.   ) ) ;
      const DDLogicalPart ilyDiffLog ( ilyDiffName(),
				       ilyDiffMat(),
				       ilyDiffSolid ) ;

      DDSolid ilyBndlSolid ( DDSolidFactory::box( ilyBndlName(), 
						  ilyFanOutHeight()/2.,
						  ilyFanOutWidth()/2.,  
						  ilyBndlLength()/2.   ) ) ;
      const DDLogicalPart ilyBndlLog ( ilyBndlName(),
				       ilyBndlMat(),
				       ilyBndlSolid ) ;
      cpv.position( ilyDiffLog,
	     ilyFanOutName(), 
	     copyOne, 
	     DDTranslation(0,0, -ilyFanOutLength()/2 + ilyDiffLength()/2 + ilyDiffOff() ),
	     DDRotation() ) ;
      cpv.position( ilyBndlLog,
	     ilyFanOutName(), 
	     copyOne, 
	     DDTranslation(0,0, -ilyFanOutLength()/2 + ilyBndlLength()/2 + ilyBndlOff() ),
	     DDRotation() ) ;

      for( unsigned int ily ( 0 ) ; ily != vecIlyThick().size() ; ++ily )
      {
	 const double        ilyRMax     ( ilyRMin + vecIlyThick()[ily] ) ;
	 const DDName        xilyName    ( ddname( ilyName() + std::to_string(ily) ) ) ;
	 const DDSolid       xilySolid   ( DDSolidFactory::tubs( xilyName,
								 ilyLength/2,
								 ilyRMin,
								 ilyRMax,
								 ilyPhiLow(),
								 ilyDelPhi() ) ) ;

	 const DDLogicalPart xilyLog     ( xilyName, ddmat(vecIlyMat()[ily]), xilySolid ) ;

	 if( 0 != ilyHere() )
	 {
	    cpv.position( xilyLog,
		   ilyLog, 
		   copyOne, 
		   DDTranslation(0,0,0),
		   DDRotation() ) ;

	    unsigned int copyNum[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} ;

	    if( 10*mm <  vecIlyThick()[ily] &&
		vecIlyThick().size() != (ily+1) &&
		0     != ilyPipeHere()         )
	    {
	       if( 0 != ilyPTMHere() )
	       {
		  unsigned int ptmCopy ( 0 ) ;
		  for( unsigned int ilyPTM ( 0 ) ; ilyPTM != vecIlyPTMZ().size() ; ++ilyPTM )
		  {
		     const double radius ( ilyRMax - 1*mm - ilyPTMHeight()/2. ) ;
		     const double phi    ( vecIlyPTMPhi()[ilyPTM] ) ;
		     const double yy     ( radius*sin(phi) ) ;
		     const double xx     ( radius*cos(phi) ) ;
		     ++ptmCopy ; 
		     cpv.position( ilyPTMLog,
			    xilyLog, 
			    ptmCopy, 
			    DDTranslation(xx,yy, vecIlyPTMZ()[ilyPTM] -ilyLength/2 ),
			    myrot( ilyPTMLog.name().name() + "_rot" +
				   std::to_string( ptmCopy ) , CLHEP::HepRotationZ( phi ) )) ;
		  }
	       }
	       if( 0 != ilyFanOutHere() )
	       {
		  unsigned int fanOutCopy ( 0 ) ;
		  for( unsigned int ilyFO ( 0 ) ; ilyFO != vecIlyFanOutZ().size() ; ++ilyFO )
		  {
		     const double radius ( ilyRMax - 1*mm - ilyFanOutHeight()/2. ) ;
		     const double phi    ( vecIlyFanOutPhi()[ilyFO] ) ;
		     const double yy     ( radius*sin(phi) ) ;
		     const double xx     ( radius*cos(phi) ) ;
		     ++fanOutCopy ; 
		     cpv.position( ilyFanOutLog,
			    xilyLog, 
			    fanOutCopy, 
			    DDTranslation(xx,yy, vecIlyFanOutZ()[ilyFO] -ilyLength/2 ),
			    myrot( ilyFanOutLog.name().name() + "_rot" +
				   std::to_string( fanOutCopy ) , 
				   CLHEP::HepRotationZ( phi )*CLHEP::HepRotationY( 180*deg ) )) ;
		  }
		  unsigned int femCopy ( 0 ) ;
		  for( unsigned int ilyFEM ( 0 ) ; ilyFEM != vecIlyFEMZ().size() ; ++ilyFEM )
		  {
		     const double radius ( ilyRMax - 1*mm - ilyFEMHeight()/2. ) ;
		     const double phi    ( vecIlyFEMPhi()[ilyFEM] ) ;
		     const double yy     ( radius*sin(phi) ) ;
		     const double xx     ( radius*cos(phi) ) ;
		     ++femCopy ; 
		     cpv.position( ilyFEMLog,
			    xilyLog, 
			    femCopy, 
			    DDTranslation(xx,yy, vecIlyFEMZ()[ilyFEM] -ilyLength/2 ),
			    myrot( ilyFEMLog.name().name() + "_rot" +
				   std::to_string( femCopy ) , CLHEP::HepRotationZ( phi ) )) ;
		  }
	       }
	       for( unsigned int iPipe ( 0 ) ; iPipe != vecIlyPipePhi().size(); ++iPipe )
	       {
		  const unsigned int type ( static_cast<unsigned int> ( round( vecIlyPipeType()[iPipe] ) ) ) ;
//		  std::cout<<" iPipe, type= " << iPipe << ", " << type << std::endl ;
		  const double       zz   ( -ilyLength/2 + vecIlyPipeZ()[iPipe] +
					    ( 9>type ? vecIlyPipeLength()[type]/2. : 0 ) ) ;

		  for( unsigned int ly ( 0 ) ; ly != 2 ; ++ly )
		  {
		     const double radius ( 0 == ly ? ilyRMin + ilyPipeOD()/2. + 1*mm :
					   ilyRMax - ilyPipeOD()/2. - 1*mm  ) ;
		     const double phi    ( vecIlyPipePhi()[iPipe] ) ;
		     const double yy     ( radius*sin(phi) ) ;
		     const double xx     ( radius*cos(phi) ) ;
		     ++copyNum[type],
		     cpv.position( ilyPipeLog[type],
			    xilyLog, 
			    copyNum[type],
			    DDTranslation(xx,yy,zz),
			    ( 9 > type ? DDRotation() :
			      myrot( ilyPipeLog[type].name().name() + "_rot" +
				     std::to_string( copyNum[type] ) , Rota( Vec3(xx,yy,0), 90*deg) ) ) ) ;
		  }
	       }
	    }
	 }
	 ilyRMin = ilyRMax ;
      }      
      // End Inner Layer volumes---------------------------------------------------------

      const DDName clyrName ( DDName( "ECLYR" ) ) ;
      std::vector<double> cri;
      std::vector<double> cro;
      std::vector<double> czz;
      czz.emplace_back( vecSpmZPts()[1] ) ;
      cri.emplace_back( vecSpmRMin()[0] ) ;
      cro.emplace_back( vecSpmRMin()[0] + 25*mm ) ;
      czz.emplace_back( vecSpmZPts()[2] ) ;
      cri.emplace_back( vecSpmRMin()[2] ) ;
      cro.emplace_back( vecSpmRMin()[2] + 10*mm ) ;
      const DDSolid clyrSolid ( DDSolidFactory::polycone( clyrName, -9.5*deg,  19*deg, czz,cri,cro) ) ;
      const DDLogicalPart clyrLog ( clyrName, ddmat(vecIlyMat()[4]), clyrSolid ) ;
      cpv.position( clyrLog, spmLog, copyOne, DDTranslation(0,0,0), DDRotation() ) ;

      // Begin Alveolar Wedge parent ------------------------------------------------------
//----------------

      // the next few lines accumulate dimensions appropriate to crystal type 1
      // which we use to set some of the features of the half-alveolar wedge (hawR).

//      const double ANom1 ( vecNomCryDimAR()[0] ) ;
      const double BNom1 ( vecNomCryDimCR()[0] ) ;
      const double bNom1 ( vecNomCryDimCF()[0] ) ;
//      const double HNom1 ( vecNomCryDimBR()[0] ) ;
//      const double hNom1 ( vecNomCryDimBF()[0] ) ;
      const double sWall1( wallThAlv() ) ;
      const double fWall1( wallFrAlv() ) ;
//      const double rWall1( wallReAlv() ) ;
      const double sWrap1( wrapThAlv() ) ;
      const double fWrap1( wrapFrAlv() ) ;
//      const double rWrap1( wrapReAlv() ) ;
      const double sClr1 ( clrThAlv() ) ;
      const double fClr1 ( clrFrAlv() ) ;
//      const double rClr1 ( clrReAlv() ) ;
      const double LNom1 ( nomCryDimLZ() ) ;
      const double beta1      ( atan( ( BNom1 - bNom1 )/LNom1 ) ) ;
//      const double cosbeta1   ( cos( beta1 ) ) ;
      const double sinbeta1   ( sin( beta1 ) ) ;

      const double tana_hawR ( ( BNom1 - bNom1 )/LNom1 ) ;

      const double H_hawR ( hawRHBIG() ) ;
      const double h_hawR ( hawRhsml() ) ;
      const double a_hawR ( bNom1 + sClr1 + 2*sWrap1 + 2*sWall1
			    - sinbeta1*( fClr1 + fWrap1 + fWall1 ) ) ;
      const double B_hawR ( a_hawR + H_hawR*tana_hawR ) ;
      const double b_hawR ( a_hawR + h_hawR*tana_hawR ) ;
      const double L_hawR ( vecSpmZPts()[2] ) ;

      const Trap trapHAWR (
	 a_hawR/2.,           //double aHalfLengthXNegZLoY , // bl1, A/2
	 a_hawR/2.,           //double aHalfLengthXPosZLoY , // bl2, a/2
	 b_hawR/2.,           //double aHalfLengthXPosZHiY , // tl2, b/2
	 H_hawR/2.,           //double aHalfLengthYNegZ    , // h1, H/2
	 h_hawR/2.,           //double aHalfLengthYPosZ    , // h2, h/2
	 L_hawR/2.,           //double aHalfLengthZ        , // dz,  L/2
	 90*deg,              //double aAngleAD            , // alfa1
	 0,                   //double aCoord15X           , // x15
	 0                    //double aCoord15Y             // y15
	 ) ;

      const DDName        hawRName1  ( ddname( hawRName().name() + "1") ) ;
      const DDSolid       hawRSolid1 ( mytrap(hawRName1.name(), trapHAWR ) ) ;
      const DDLogicalPart hawRLog1   ( hawRName1, spmMat(), hawRSolid1 ) ;

      const double al1_fawR ( atan( ( B_hawR - a_hawR )/H_hawR ) + M_PI_2 ) ;

      // here is trap for Full Alveolar Wedge
      const Trap trapFAW (
	 a_hawR,           //double aHalfLengthXNegZLoY , // bl1, A/2
	 a_hawR,           //double aHalfLengthXPosZLoY , // bl2, a/2
	 b_hawR,           //double aHalfLengthXPosZHiY , // tl2, b/2
	 H_hawR/2.,        //double aHalfLengthYNegZ    , // h1, H/2
	 h_hawR/2.,        //double aHalfLengthYPosZ    , // h2, h/2
	 L_hawR/2.,        //double aHalfLengthZ        , // dz,  L/2
	 al1_fawR,         //double aAngleAD            , // alfa1
	 0,                //double aCoord15X           , // x15
	 0                 //double aCoord15Y             // y15
	 ) ;

      const DDName        fawName1  ( ddname( fawName().name() + "1") ) ;
      const DDSolid       fawSolid1 ( mytrap( fawName1.name(), trapFAW ) ) ;
      const DDLogicalPart fawLog1   ( fawName1, spmMat(), fawSolid1 ) ;

      const Trap::VertexList vHAW ( trapHAWR.vertexList() ) ;
      const Trap::VertexList vFAW ( trapFAW.vertexList() ) ;

      const double hawBoxClr ( 1*mm ) ;

      // HAW cut box to cut off back end of wedge
      const DDName  hawCutName ( ddname( hawRName().name() + "CUTBOX" ) ) ;
      const DDSolid hawCutBox  ( DDSolidFactory::box(
				    hawCutName, 
				    b_hawR/2 + hawBoxClr,  
				    hawRCutY()/2,
				    hawRCutZ()/2 ) ) ;
      const std::vector<double>& hawBoxParms ( hawCutBox.parameters() ) ;
      const DDLogicalPart hawCutLog ( hawCutName, spmMat(), hawCutBox ) ;
      
      const Pt3D b1 (  hawBoxParms[0],  hawBoxParms[1],  hawBoxParms[2] ) ;
      const Pt3D b2 ( -hawBoxParms[0],  hawBoxParms[1],  hawBoxParms[2] ) ;
      const Pt3D b3 ( -hawBoxParms[0],  hawBoxParms[1], -hawBoxParms[2] ) ;

      const double zDel ( sqrt( 4*hawBoxParms[2]*hawBoxParms[2] 
				-(h_hawR-hawRCutDelY())*(h_hawR-hawRCutDelY())  ) ) ;

      const Tf3D hawCutForm ( 
	 b1, b2, b3,
	 vHAW[2]    + Pt3D( hawBoxClr, -hawRCutDelY(), 0), 
	 vHAW[1]    + Pt3D(-hawBoxClr, -hawRCutDelY(), 0),
	 Pt3D( vHAW[0].x() -hawBoxClr, vHAW[0].y(), vHAW[0].z() - zDel  ) ) ;

      const DDSolid       hawRSolid ( DDSolidFactory::subtraction(
					 hawRName(),
					 hawRSolid1, hawCutBox,
					 DDTranslation(hawCutForm.getTranslation().x(), hawCutForm.getTranslation().y(), hawCutForm.getTranslation().z()),
					 myrot( hawCutName.name()+"R",
						hawCutForm.getRotation() ) ) ) ;
      const DDLogicalPart hawRLog   ( hawRName(), spmMat(), hawRSolid ) ;

      // FAW cut box to cut off back end of wedge
      const DDName  fawCutName ( ddname( fawName().name() + "CUTBOX") ) ;
      const DDSolid fawCutBox  ( DDSolidFactory::box(
				    fawCutName, 
				    2*hawBoxParms[0],  
				    hawBoxParms[1],
				    hawBoxParms[2] ) ) ;

      const std::vector<double>& fawBoxParms ( fawCutBox.parameters() ) ;
      const DDLogicalPart fawCutLog ( fawCutName, spmMat(), fawCutBox ) ;
      
      const Pt3D bb1 (  fawBoxParms[0],  fawBoxParms[1],  fawBoxParms[2] ) ;
      const Pt3D bb2 ( -fawBoxParms[0],  fawBoxParms[1],  fawBoxParms[2] ) ;
      const Pt3D bb3 ( -fawBoxParms[0],  fawBoxParms[1], -fawBoxParms[2] ) ;

      const Tf3D fawCutForm ( 
	 bb1, bb2, bb3,
	 vFAW[2]   + Pt3D( 2*hawBoxClr,-5*mm,0), 
	 vFAW[1]   + Pt3D(-2*hawBoxClr,-5*mm,0),
	 Pt3D( vFAW[1].x()-2*hawBoxClr, vFAW[1].y()-trapFAW.h(), vFAW[1].z() - zDel ) ) ;

      const DDSolid       fawSolid ( DDSolidFactory::subtraction(
					 fawName(),
					 fawSolid1, fawCutBox,
					 DDTranslation(fawCutForm.getTranslation().x(), fawCutForm.getTranslation().y(), fawCutForm.getTranslation().z()) ,
					 myrot( fawCutName.name()+"R",
						fawCutForm.getRotation() ) ) ) ;
      const DDLogicalPart fawLog   ( fawName(), spmMat(), fawSolid ) ;


      const Tf3D hawRform ( vHAW[3], vHAW[0], vHAW[1], // HAW inside FAW
			    vFAW[3], 0.5*(vFAW[0]+vFAW[3]), 0.5*(vFAW[1]+vFAW[2] ) ) ;
      cpv.position( hawRLog,
	     fawLog, 
	     copyOne, 
	     DDTranslation(hawRform.getTranslation().x(), hawRform.getTranslation().y(), hawRform.getTranslation().z()),
	     myrot( hawRName().name()+"R", 
		    hawRform.getRotation() ) ) ;

      cpv.position( hawRLog,
	     fawLog, 
	     copyTwo, 
	     DDTranslation( -hawRform.getTranslation().x(),
			    -hawRform.getTranslation().y(),
			    -hawRform.getTranslation().z() ),
	     myrot( hawRName().name()+"RotRefl",
		    CLHEP::HepRotationY(180*deg)*                 // rotate about Y after refl thru Z
		    CLHEP::HepRep3x3(1,0,0, 0,1,0, 0,0,-1) ) ) ;

/* this for display of haw cut box instead of subtraction
      cpv.position( hawCutLog,
	     hawRName, 
	     copyOne, 
	     hawCutForm.getTranslation(),
	     myrot( hawCutName.name()+"R", 
		    hawCutForm.getRotation() )   ) ;
*/

      for( unsigned int iPhi ( 1 ); iPhi <= nFawPerSupm() ; ++iPhi )
      {
	 const double rPhi ( fawPhiOff() + ( iPhi - 0.5 )*fawDelPhi() ) ;

	 const Tf3D fawform ( RoZ3D( rPhi )*
			      Tl3D( fawRadOff() + ( trapFAW.H() + trapFAW.h() )/4 ,
				    0, 
				    trapFAW.L()/2 )*
			      RoZ3D( -90*deg + fawPhiRot() ) ) ;
	 if( fawHere() )
	    cpv.position( fawLog,
		   spmLog, 
		   iPhi, 
		   DDTranslation(fawform.getTranslation().x(), fawform.getTranslation().y(), fawform.getTranslation().z() ),
			  myrot( fawName().name()+"_Rot" + std::to_string(iPhi), 
			  fawform.getRotation() ) ) ;
      }

      // End Alveolar Wedge parent ------------------------------------------------------

      // Begin Grid + Tablet insertion

      const double h_Grid ( gridThick() ) ;

      const Trap trapGrid (
	 ( B_hawR - h_Grid*( B_hawR - a_hawR )/H_hawR )/2, // bl1, A/2
	 ( b_hawR - h_Grid*( B_hawR - a_hawR )/H_hawR )/2, // bl2, a/2
	 b_hawR/2., // tl2, b/2
	 h_Grid/2., // h1, H/2
	 h_Grid/2., // h2, h/2
	 (L_hawR-8*cm)/2., // dz,  L/2
	 90*deg,    // alfa1
	 0,         // x15
	 H_hawR - h_hawR // y15
	 ) ;

      const DDSolid       gridSolid ( mytrap( gridName().name(), trapGrid ) ) ;
      const DDLogicalPart gridLog   ( gridName(), gridMat(), gridSolid ) ;

      const Trap::VertexList vGrid ( trapGrid.vertexList() ) ;

      const Tf3D gridForm ( vGrid[4],                   vGrid[5], vGrid[6], // Grid inside HAW
			    vHAW[5] - Pt3D(0,h_Grid,0),  vHAW[5],  vHAW[6]   ) ;

      if( 0 != gridHere() )
	 cpv.position( gridLog,
		hawRLog, 
		copyOne, 
		DDTranslation(gridForm.getTranslation().x(), gridForm.getTranslation().y(), gridForm.getTranslation().z() ),
		myrot( gridName().name()+"R", 
		       gridForm.getRotation() ) ) ;

      // End Grid + Tablet insertion

      // begin filling Wedge with crystal plus supports --------------------------

      const double aNom ( nomCryDimAF() ) ;
      const double LNom ( nomCryDimLZ() ) ;

      const double AUnd ( underAR() ) ;
      const double aUnd ( underAF() ) ;
//      const double BUnd ( underCR() ) ;
      const double bUnd ( underCF() ) ;
      const double HUnd ( underBR() ) ;
      const double hUnd ( underBF() ) ;
      const double LUnd ( underLZ() ) ;

      const double sWall ( wallThAlv() ) ;
      const double sWrap ( wrapThAlv() ) ;
      const double sClr  ( clrThAlv() ) ;

      const double fWall ( wallFrAlv() ) ;
      const double fWrap ( wrapFrAlv() ) ;
      const double fClr  ( clrFrAlv() ) ;

      const double rWall ( wallReAlv() ) ;
      const double rWrap ( wrapReAlv() ) ;
      const double rClr  ( clrReAlv() ) ;
      
      // theta is angle in yz plane between z axis & leading edge of crystal
      double theta ( 90*deg ) ;
      double zee   ( 0*mm ) ;
      double side  ( 0*mm ) ;
      double zeta  ( 0*deg ) ; // increment in theta for last crystal

      for( unsigned int cryType ( 1 ) ; cryType <= nCryTypes() ; ++cryType )
      {
	 const std::string sType ( "_" + 
				   std::string( 10>cryType ? "0" : "") + 
				   std::to_string( cryType ) ) ;

	 LogDebug("EcalGeom") << "Crytype=" << cryType ;
	 const double ANom ( vecNomCryDimAR()[ cryType-1 ] ) ;
	 const double BNom ( vecNomCryDimCR()[ cryType-1 ] ) ;
	 const double bNom ( vecNomCryDimCF()[ cryType-1 ] ) ;
	 const double HNom ( vecNomCryDimBR()[ cryType-1 ] ) ;
	 const double hNom ( vecNomCryDimBF()[ cryType-1 ] ) ;

	 const double alfCry ( 90*deg + atan( ( bNom - bUnd - aNom + aUnd )/
					      ( hNom - hUnd ) ) ) ;

	 const Trap trapCry (
	    ( ANom - AUnd )/2.,           //double aHalfLengthXNegZLoY , // bl1, A/2
	    ( aNom - aUnd )/2.,           //double aHalfLengthXPosZLoY , // bl2, a/2
	    ( bNom - bUnd )/2.,           //double aHalfLengthXPosZHiY , // tl2, b/2
	    ( HNom - HUnd )/2.,           //double aHalfLengthYNegZ    , // h1, H/2
	    ( hNom - hUnd )/2.,           //double aHalfLengthYPosZ    , // h2, h/2
	    ( LNom - LUnd )/2.,           //double aHalfLengthZ        , // dz,  L/2
	    alfCry,                       //double aAngleAD            , // alfa1
	    aNom - aUnd - ANom + AUnd,    //double aCoord15X           , // x15
	    hNom - hUnd - HNom + HUnd     //double aCoord15Y             // y15      
	 ) ;

	 const DDName        cryDDName ( cryName() + sType ) ;
	 const DDSolid       crySolid  ( mytrap( cryDDName.name(), trapCry ) ) ;
	 const DDLogicalPart cryLog    ( cryDDName, cryMat(), crySolid ) ;


//++++++++++++++++++++++++++++++++++  APD ++++++++++++++++++++++++++++++++++

	 const DDName        apdDDName ( apdName().name() + sType ) ;

	 const Trap trapAPD (
	    apdSide()/2.,    //double aHalfLengthXNegZLoY , // bl1, A/2
	    apdSide()/2.,    //double aHalfLengthXPosZLoY , // bl2, a/2
	    apdSide()/2.,    //double aHalfLengthXPosZHiY , // tl2, b/2
	    apdSide()/2.,    //double aHalfLengthYNegZ    , // h1,  H/2
	    apdSide()/2.,    //double aHalfLengthYPosZ    , // h2,  h/2
	    apdThick()/2.,   // dz,  L/2
	    90*deg,                //double aAngleAD            , // alfa1
	    0,          //double aCoord15X           , // x15
	    0           //double aCoord15Y             // y15  
	    ) ;

//	 DDSolid apdSolid ( DDSolidFactory::box( apdDDName, 
//						 apdSide()/2.,
//						 apdSide()/2.,  
//						 apdThick()/2.   ) ) ;
	 const DDSolid       apdSolid  ( mytrap( apdDDName.name(), trapAPD ) ) ;
	 const DDLogicalPart apdLog ( apdDDName, apdMat(), apdSolid ) ;
	 
	 unsigned int copyAPD ( 0 ) ;
//++++++++++++++++++++++++++++++++++ END APD ++++++++++++++++++++++++++++++++++


	 const double delta     ( atan( ( HNom - hNom )/LNom ) ) ;
//unused	 const double cosdelta  ( cos( delta ) ) ;
	 const double sindelta  ( sin( delta ) ) ;

	 const double gamma     ( atan( ( ANom - aNom )/LNom ) ) ;
//unused	 const double cosgamma  ( cos( gamma ) ) ;
	 const double singamma  ( sin( gamma ) ) ;

	 const double beta      ( atan( ( BNom - bNom )/LNom ) ) ;
//unused	 const double cosbeta   ( cos( beta ) ) ;
	 const double sinbeta   ( sin( beta ) ) ;

	 // Now clearance trap
	 const double alfClr ( 90*deg + atan( ( bNom - aNom )/( hNom + sClr ) ) ) ;

	 const Trap trapClr (
	    ( ANom + sClr + rClr*singamma )/2.,    //double aHalfLengthXNegZLoY , // bl1, A/2
	    ( aNom + sClr - fClr*singamma )/2.,    //double aHalfLengthXPosZLoY , // bl2, a/2
	    ( bNom + sClr - fClr*sinbeta  )/2.,    //double aHalfLengthXPosZHiY , // tl2, b/2
	    ( HNom + sClr + rClr*sindelta )/2.,    //double aHalfLengthYNegZ    , // h1,  H/2
	    ( hNom + sClr - fClr*sindelta )/2.,    //double aHalfLengthYPosZ    , // h2,  h/2
	    ( LNom + fClr + rClr )/2., // dz,  L/2
	    alfClr,                //double aAngleAD            , // alfa1
	    aNom - ANom ,          //double aCoord15X           , // x15
	    hNom - HNom            //double aCoord15Y             // y15  
	 ) ;

	 const DDName        clrDDName ( clrName() + sType ) ;
	 const DDSolid       clrSolid  ( mytrap( clrDDName.name(), trapClr ) ) ;
	 const DDLogicalPart clrLog    ( clrDDName, clrMat(), clrSolid ) ;

	 // Now wrap trap

	 const double alfWrap ( 90*deg + atan( ( bNom - aNom )/
					       ( hNom + sClr + 2*sWrap ) ) ) ;

	 const Trap trapWrap (
	    ( trapClr.A() + 2*sWrap + rWrap*singamma )/2, // bl1, A/2
	    ( trapClr.a() + 2*sWrap - fWrap*singamma )/2, // bl2, a/2
	    ( trapClr.b() + 2*sWrap - fWrap*sinbeta  )/2, // tl2, b/2
	    ( trapClr.H() + 2*sWrap + rWrap*sindelta )/2, // h1,  H/2
	    ( trapClr.h() + 2*sWrap - fWrap*sindelta )/2, // h2,  h/2
	    ( trapClr.L() + fWrap + rWrap )/2., // dz,  L/2
	    alfWrap,                       //double aAngleAD            , // alfa1
	    aNom - ANom - (cryType>9 ? 0 : 0.020*mm) ,
	    hNom - HNom                    //double aCoord15Y             // y15  
	 ) ;

	 const DDName        wrapDDName ( wrapName() + sType ) ;
	 const DDSolid       wrapSolid  ( mytrap( wrapDDName.name(), trapWrap ) ) ;
	 const DDLogicalPart wrapLog    ( wrapDDName, wrapMat(), wrapSolid ) ;

	 // Now wall trap

	 const double alfWall ( 90*deg + atan( ( bNom - aNom )/
					       ( hNom + sClr + 2*sWrap + 2*sWall ) ) ) ;

	 const Trap trapWall (
	    ( trapWrap.A() + 2*sWall + rWall*singamma )/2,  // A/2 
	    ( trapWrap.a() + 2*sWall - fWall*singamma )/2,  // a/2
	    ( trapWrap.b() + 2*sWall - fWall*sinbeta  )/2,  // b/2
	    ( trapWrap.H() + 2*sWall + rWall*sindelta )/2,  // H/2
	    ( trapWrap.h() + 2*sWall - fWall*sindelta )/2,  // h/2
	    ( trapWrap.L() + fWall + rWall )/2.,  // L/2
	    alfWall,                             // alfa1
	    aNom - ANom - (cryType<10? 0.150*mm : 0.100*mm ) ,       
	    hNom - HNom                          // y15
	 ) ;

	 const DDName        wallDDName ( wallName() + sType ) ;
	 const DDSolid       wallSolid  ( mytrap( wallDDName.name(), trapWall ) ) ;
	 const DDLogicalPart wallLog    ( wallDDName, wallMat(), wallSolid ) ;
	 
/*	 std::cout << "Traps:\n a: " 
		<< trapCry.a() << ", " 
		<< trapClr.a() << ", " 
		<< trapWrap.a() << ", " 
		<< trapWall.a() << "\n b: " 
		<< trapCry.b() << ", " 
		<< trapClr.b() << ", " 
		<< trapWrap.b() << ", " 
		<< trapWall.b() << "\n A: " 
		<< trapCry.A() << ", " 
		<< trapClr.A() << ", " 
		<< trapWrap.A() << ", " 
		<< trapWall.A() << "\n B: " 
		<< trapCry.B() << ", " 
		<< trapClr.B() << ", " 
		<< trapWrap.B() << ", " 
		<< trapWall.B() << "\n h: " 
		<< trapCry.h() << ", " 
		<< trapClr.h() << ", " 
		<< trapWrap.h() << ", " 
		<< trapWall.h() << "\n H: " 
		<< trapCry.H() << ", " 
		<< trapClr.H() << ", " 
		<< trapWrap.H() << ", " 
		<< trapWall.H() << "\n a1: " 
		<< trapCry.a1()/deg << ", " 
		<< trapClr.a1()/deg << ", " 
		<< trapWrap.a1()/deg << ", " 
		<< trapWall.a1()/deg << "\n a4: " 
		<< trapCry.a4()/deg << ", " 
		<< trapClr.a4()/deg << ", " 
		<< trapWrap.a4()/deg << ", " 
		<< trapWall.a4()/deg << "\n x15: " 
		<< trapCry.x15() << ", " 
		<< trapClr.x15() << ", " 
		<< trapWrap.x15() << ", " 
		<< trapWall.x15() << "\n y15: " 
		<< trapCry.y15() << ", " 
		<< trapClr.y15() << ", " 
		<< trapWrap.y15() << ", " 
		<< trapWall.y15()
		<< std::endl ;
*/
	 // Now for placement of cry within clr
	 const Vec3 cryToClr ( 0, 0, ( rClr - fClr )/2 ) ;

	 cpv.position( cryLog,
		clrLog, 
		copyOne, 
		DDTranslation ( 0, 0, ( rClr - fClr )/2 ), //SAME as cryToClr above.
		DDRotation() ) ;

	 if( 0 != apdHere() )
	 { 
	    cpv.position( apdLog,
		   clrLog, 
		   ++copyAPD, 
		   DDTranslation( +trapCry.bl1() - apdX1(),
				  +trapCry.h1()  - apdZ(),
				  -trapCry.dz()  - apdThick()/2. + (rClr - fClr)/2. ),
		   DDRotation() ) ;
	    cpv.position( apdLog,
		   clrLog, 
		   ++copyAPD, 
		   DDTranslation( +trapCry.bl1() - apdX2(),
				  +trapCry.h1()  - apdZ(),
				  -trapCry.dz()  - apdThick()/2. + (rClr - fClr)/2.),
		   DDRotation() ) ;
	 }

	 const Vec3 clrToWrap ( 0, 0, ( rWrap - fWrap )/2 ) ;

	 cpv.position( clrLog,
		wrapLog, 
		copyOne, 
		DDTranslation ( 0, 0, ( rWrap - fWrap )/2 ), //SAME as cryToWrap
		DDRotation() ) ;


	 // Now for placement of clr within wall
	 const Vec3 wrapToWall1 ( 0, 0, ( rWall - fWall )/2 ) ;
	 const Vec3 wrapToWall ( Vec3( (cryType>9?0:0.005*mm),0,0 )+wrapToWall1 ) ;

	 cpv.position( wrapLog,
		wallLog, 
		copyOne, 
		DDTranslation ( Vec3( (cryType>9?0:0.005*mm),0,0 )+wrapToWall1 ), //SAME as wrapToWall
		DDRotation() ) ;

         const Trap::VertexList vWall ( trapWall.vertexList() ) ;
         const Trap::VertexList vCry  ( trapCry.vertexList() ) ;

	 const double sidePrime  ( ( trapWall.a() - trapCry.a() )/2 ) ;
	 const double frontPrime ( fWall + fWrap + fClr + LUnd/2 ) ;

	 // define web plates with clearance ===========================================

	 if( 1 == cryType ) // first web plate: inside clearance volume
	 {
	    web( 0,
		 trapWall.b(),
		 trapWall.B(),
		 trapWall.L(),
		 theta,
		 vHAW[4] + Pt3D( 0, hawYOffCry(), 0 ),
		 hawRLog,
		 zee,
		 sidePrime,
		 frontPrime,
		 delta, cpv ) ;
	    zee += vecGapAlvEta()[0] ;
	 }

	 for( unsigned int etaAlv ( 1 ) ; etaAlv <= nCryPerAlvEta() ; ++etaAlv )
	 {
	    LogDebug("EcalGeom") << "theta=" << theta/deg
		      << ", sidePrime=" << sidePrime << ", frontPrime=" << frontPrime
		      << ",  zeta="<<zeta<<", delta="<<delta<<",  zee=" << zee;

	    zee += 0.075*mm +
	       ( side*cos(zeta) + trapWall.h() - sidePrime )/sin(theta) ;

	    LogDebug("EcalGeom") << "New zee="<< zee ;

	    // make transform for placing enclosed crystal

	    const Pt3D trap2 ( vCry[2] + cryToClr + clrToWrap + wrapToWall ) ;
	    
	    const Pt3D trap3 ( trap2 + Pt3D( 0,           
					     -trapCry.h(),
					     0 ) ) ;
	    const Pt3D trap1 ( trap3 + Pt3D( -trapCry.a(),
					     0,           
					     0 ) ) ;

	    const Pt3D wedge3 ( vHAW[4] + Pt3D( sidePrime,
						hawYOffCry(), 
						zee ) ) ;
	    const Pt3D wedge2 ( wedge3  + Pt3D( 0,
						trapCry.h()*cos(theta),
						-trapCry.h()*sin(theta)  ) ) ;
	    const Pt3D wedge1 ( wedge3  + Pt3D( trapCry.a(),
						0,
						0            ) ) ;

	    const Tf3D tForm1 ( trap1,  trap2,  trap3,
				wedge1, wedge2, wedge3    ) ;

	    const double xx ( 0.050*mm ) ;

	    const Tf3D tForm ( HepGeom::Translate3D(xx,0,0)*tForm1 ) ;

	    cpv.position( wallLog,
		   hawRLog, 
		   etaAlv, 
		   DDTranslation(tForm.getTranslation().x(), tForm.getTranslation().y(), tForm.getTranslation().z() ),
			  myrot( wallLog.name().name() + "_" + std::to_string( etaAlv ),
			  tForm.getRotation() ) ) ;

	    theta     -= delta ;
	    side       = sidePrime ;
	    zeta       = delta ;
	 }
	 if( 5 == cryType ||
	     9 == cryType ||
	    13 == cryType ||
	    17 == cryType    ) // web plates
	 {
	    const unsigned int webIndex ( cryType/4 ) ;
	    zee += 0.5*vecGapAlvEta()[cryType]/sin(theta) ;
	    web( webIndex,
		 trapWall.a(),
		 trapWall.A(),
		 trapWall.L(),
		 theta,
		 vHAW[4] + Pt3D( 0, hawYOffCry(), 0 ),
		 hawRLog,
		 zee ,
		 sidePrime,
		 frontPrime,
		 delta, cpv ) ;
	    zee += 0.5*vecGapAlvEta()[cryType]/sin(theta) ;
	 }
	 else
	 {
	    if( 17 != cryType ) zee += vecGapAlvEta()[cryType]/sin(theta) ;
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

      if( 0 != backHere() )
      {

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!     Begin Back Cover Plate     !!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      const DDTranslation outtra ( backXOff() + backSideHeight()/2,
				   backYOff(),
				   backSideLength()/2 ) ;

      const double realBPthick ( backPlateThick() + backPlate2Thick() ) ;

      DDSolid backPlateSolid ( DDSolidFactory::box( backPlateName(), 
						    backPlateWidth()/2.,  
						    realBPthick/2.,
						    backPlateLength()/2.   ) ) ;
      const std::vector<double>& backPlateParms ( backPlateSolid.parameters() ) ;
      const DDLogicalPart backPlateLog ( backPlateName(),
					 backPlateMat(),
					 backPlateSolid ) ;
      
      const DDTranslation backPlateTra ( backSideHeight()/2 + 
					 backPlateParms[1],
					 0*mm,
					 backPlateParms[2] -
					 backSideLength()/2 ) ;

      DDSolid backPlate2Solid ( DDSolidFactory::box( backPlate2Name(), 
						    backPlateWidth()/2.,  
						    backPlate2Thick()/2.,
						    backPlateLength()/2.   ) ) ;

      const DDLogicalPart backPlate2Log ( backPlate2Name(),
					  backPlate2Mat(),
					  backPlate2Solid ) ;
      
      const DDTranslation backPlate2Tra ( 0,
					  -backPlateParms[1] + backPlate2Thick()/2., 0 ) ;
      if( 0 != backPlateHere() )
      {
	 cpv.position( backPlate2Log,
		backPlateName(), 
		copyOne, 
		backPlate2Tra,
		DDRotation() ) ;

	 cpv.position( backPlateLog,
		spmName(), 
		copyOne, 
		outtra + backPlateTra,
		myrot( backPlateName().name()+"Rot5",
		       CLHEP::HepRotationZ(270*deg)    ) ) ;

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

      const Trap trapBS (
	 backSideWidth()/2.,  //double aHalfLengthXNegZLoY , // bl1, A/2
	 backSideWidth()/2.,  //double aHalfLengthXPosZLoY , // bl2, a/2
	 backSideWidth()/4.,  //double aHalfLengthXPosZHiY , // tl2, b/2
	 backSideHeight()/2., //double aHalfLengthYNegZ    , // h1, H/2
	 backSideHeight()/2., //double aHalfLengthYPosZ    , // h2, h/2
	 backSideLength()/2., //double aHalfLengthZ        , // dz,  L/2
	 backSideAngle(),              //double aAngleAD            , // alfa1
	 0,                   //double aCoord15X           , // x15
	 0                    //double aCoord15Y             // y15
	 ) ;

      const DDSolid       backSideSolid ( mytrap( backSideName().name(), trapBS ) ) ;
      const DDLogicalPart backSideLog ( backSideName(), 
					backSideMat(), 
					backSideSolid ) ;
      
      const DDTranslation backSideTra1 ( 0*mm,
					 backPlateWidth()/2 + backSideYOff1(),
					 1*mm ) ;
      if( 0 != backSideHere() )
      {
	 cpv.position( backSideLog,
		spmName(), 
		copyOne, 
		outtra + backSideTra1,
		myrot( backSideName().name()+"Rot8",
		       CLHEP::HepRotationX(180*deg)*CLHEP::HepRotationZ(90*deg)    ) ) ;
	     
	 const DDTranslation backSideTra2( 0*mm,
					   -backPlateWidth()/2 + backSideYOff2(),
					   1*mm ) ;
	 cpv.position( backSideLog,
		spmName(), 
		copyTwo, 
		outtra + backSideTra2,
		myrot( backSideName().name()+"Rot9",
		       CLHEP::HepRotationZ(90*deg)    ) ) ;
      }
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!     End Back Side Plates       !!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	     
//=====================
      const double backCoolWidth ( backCoolBarWidth() + 2.*backCoolTankWidth() ) ;

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!     Begin Mother Board Cooling Manifold Setup !!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      const double manifCut ( 2*mm ) ;

      DDSolid mBManifSolid ( DDSolidFactory::tubs( mBManifName() ,
						   backCoolWidth/2. - manifCut,
						   0, 
						   mBManifOutDiam()/2,
						   0*deg, 360*deg ) ) ;
      const DDLogicalPart mBManifLog ( mBManifName(), mBManifMat(), mBManifSolid ) ;

      const DDName mBManifWaName ( ddname( mBManifName().name() + "Wa" ) ) ;
      DDSolid mBManifWaSolid ( DDSolidFactory::tubs( mBManifWaName ,
						     backCoolWidth/2.-manifCut,
						     0, 
						     mBManifInnDiam()/2,
						     0*deg, 360*deg ) ) ;
      const DDLogicalPart mBManifWaLog ( mBManifWaName, backPipeWaterMat(), 
					 mBManifWaSolid ) ;
      cpv.position( mBManifWaLog,
	     mBManifName(), 
	     copyOne, 
	     DDTranslation(0,0,0),
	     DDRotation() ) ;

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
      const double deltaY ( -5*mm ) ;

      DDSolid grEdgeSlotSolid ( DDSolidFactory::box( grEdgeSlotName(), 
						     grEdgeSlotHeight()/2.,  
						     grEdgeSlotWidth()/2.,
						     grilleThick()/2.   ) ) ;
      const DDLogicalPart grEdgeSlotLog ( grEdgeSlotName(), grEdgeSlotMat(), grEdgeSlotSolid );

      unsigned int edgeSlotCopy ( 0 ) ;
      unsigned int midSlotCopy ( 0 ) ;

      DDLogicalPart grMidSlotLog[4] ;

      for( unsigned int iGr ( 0 ) ; iGr != vecGrilleHeight().size() ; ++iGr )
      {
	DDName gName ( ddname( grilleName() + std::to_string( iGr ) ) ) ;
	 DDSolid grilleSolid ( DDSolidFactory::box( gName, 
						    vecGrilleHeight()[iGr]/2.,  
						    backCoolWidth/2.,
						    grilleThick()/2.   ) ) ;
	 const DDLogicalPart grilleLog ( gName,
					 grilleMat(),
					 grilleSolid ) ;
	 
	 const DDTranslation grilleTra ( -realBPthick/2 -
					 vecGrilleHeight()[iGr]/2,
					 deltaY,
					 vecGrilleZOff()[iGr] +
					 grilleThick()/2 - backSideLength()/2 ) ;
	 const DDTranslation gTra ( outtra + backPlateTra + grilleTra ) ;

	 if( 0 != grMidSlotHere() &&
	     0 != iGr   )
	 {
	    if( 0 == (iGr-1)%2      )
	    {
	      DDName mName ( ddname( grMidSlotName() + std::to_string( iGr/2 ) ) ) ;
	       DDSolid grMidSlotSolid ( DDSolidFactory::box(
					   mName, 
					   vecGrMidSlotHeight()[(iGr-1)/2]/2.,
					   grMidSlotWidth()/2.,
					   grilleThick()/2.   ) ) ;
	       grMidSlotLog[(iGr-1)/2] = DDLogicalPart( mName, grMidSlotMat(), grMidSlotSolid ) ;
	    }	       
	    cpv.position( grMidSlotLog[(iGr-1)/2],
		   gName, 
		   ++midSlotCopy, 
		   DDTranslation( vecGrilleHeight()[iGr]/2. - vecGrMidSlotHeight()[(iGr-1)/2]/2.,
				  +grMidSlotXOff(),    0 ),
		   DDRotation() ) ;
	    cpv.position( grMidSlotLog[(iGr-1)/2],
		   gName, 
		   ++midSlotCopy, 
		   DDTranslation( vecGrilleHeight()[iGr]/2. - vecGrMidSlotHeight()[(iGr-1)/2]/2.,
			 -grMidSlotXOff(),    0 ),
		   DDRotation() ) ;
	 }

	 if( 0 != grEdgeSlotHere() &&
	     0 != iGr                   )
	 {
	    cpv.position( grEdgeSlotLog,
		   gName, 
		   ++edgeSlotCopy, 
		   DDTranslation( vecGrilleHeight()[iGr]/2. - grEdgeSlotHeight()/2.,
				  backCoolWidth/2           - grEdgeSlotWidth()/2.,    0 ),
		   DDRotation() ) ;
	    cpv.position( grEdgeSlotLog,
		   gName, 
		   ++edgeSlotCopy, 
		   DDTranslation( vecGrilleHeight()[iGr]/2. - grEdgeSlotHeight()/2.,
			 -backCoolWidth/2          + grEdgeSlotWidth()/2.,    0 ),
		   DDRotation() ) ;
	 }
	 if( 0 != grilleHere() )
	    cpv.position( grilleLog,
		   spmName(), 
		   iGr, 
		   gTra,
		   DDRotation() ) ;

	 if( ( 0 != iGr%2 )         &&
	     ( 0 != mBManifHere() )     )
	 {
	    cpv.position( mBManifLog,
		   spmName(),
		   iGr,
		   gTra - DDTranslation( -mBManifOutDiam()/2. +
					 vecGrilleHeight()[iGr]/2.,manifCut, 
					 grilleThick()/2.+3*mBManifOutDiam()/2.) ,
		   myrot( mBManifName().name()+"R1",
			  CLHEP::HepRotationX(90*deg)             ) ) ;
	    cpv.position( mBManifLog,
		   spmName(),
		   iGr-1,
		   gTra - DDTranslation( -3*mBManifOutDiam()/2. +
					 vecGrilleHeight()[iGr]/2.,manifCut, 
					 grilleThick()/2+3*mBManifOutDiam()/2.) ,
		   myrot( mBManifName().name()+"R2",
			  CLHEP::HepRotationX(90*deg)             ) ) ;
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

      DDSolid backCoolBarSolid ( DDSolidFactory::box( backCoolBarName(), 
						      backCoolBarHeight()/2.,
						      backCoolBarWidth()/2.,  
						      backCoolBarThick()/2.   ) ) ;
      const DDLogicalPart backCoolBarLog ( backCoolBarName(),
					   backCoolBarMat(),
					   backCoolBarSolid ) ;

      DDSolid backCoolBarSSSolid ( DDSolidFactory::box( backCoolBarSSName(), 
							backCoolBarHeight()/2.,
							backCoolBarWidth()/2.,  
							backCoolBarSSThick()/2.   ) ) ;
      const DDLogicalPart backCoolBarSSLog ( backCoolBarSSName(),
					     backCoolBarSSMat(),
					     backCoolBarSSSolid ) ;
      const DDTranslation backCoolBarSSTra (0,0,0) ;
      cpv.position( backCoolBarSSLog,
	     backCoolBarName(), 
	     copyOne, 
	     backCoolBarSSTra,
	     DDRotation() ) ;

      DDSolid backCoolBarWaSolid ( DDSolidFactory::box( backCoolBarWaName(), 
							backCoolBarHeight()/2.,
							backCoolBarWidth()/2.,  
							backCoolBarWaThick()/2.   ) ) ;
      const DDLogicalPart backCoolBarWaLog ( backCoolBarWaName(),
					     backCoolBarWaMat(),
					     backCoolBarWaSolid ) ;
      const DDTranslation backCoolBarWaTra (0,0,0) ;
      cpv.position( backCoolBarWaLog,
	     backCoolBarSSName(), 
	     copyOne, 
	     backCoolBarWaTra,
	     DDRotation() ) ;

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

      double thickVFE ( 0 ) ;
      for( unsigned int iLyr ( 0 ) ; iLyr != vecBackVFELyrThick().size() ; ++iLyr )
      {
	 thickVFE += vecBackVFELyrThick()[iLyr] ;
      }
      DDSolid backVFESolid ( DDSolidFactory::box( backVFEName(), 
						  backCoolBarHeight()/2.,
						  backCoolBarWidth()/2.,  
						  thickVFE/2.   ) ) ;
      const DDLogicalPart backVFELog ( backVFEName(),
				       backVFEMat(),
				       backVFESolid ) ;
      DDTranslation offTra ( 0,0,-thickVFE/2 ) ;
      for( unsigned int iLyr ( 0 ) ; iLyr != vecBackVFELyrThick().size() ; ++iLyr )
      {
	 DDSolid backVFELyrSolid ( DDSolidFactory::box( ddname(vecBackVFELyrName()[iLyr]), 
							backCoolBarHeight()/2.,
							backCoolBarWidth()/2.,  
							vecBackVFELyrThick()[iLyr]/2.   ) ) ;
	 const DDLogicalPart backVFELyrLog ( ddname(vecBackVFELyrName()[iLyr]),
					     ddmat(vecBackVFELyrMat()[iLyr]),
					     backVFELyrSolid ) ;
	 const DDTranslation backVFELyrTra (0,0, vecBackVFELyrThick()[iLyr]/2) ;
	 cpv.position( backVFELyrLog,
		backVFEName(), 
		copyOne, 
		backVFELyrTra + offTra,
		DDRotation() ) ;
	 offTra += 2*backVFELyrTra ;
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

      const double halfZCoolVFE ( thickVFE + backCoolBarThick()/2. ) ;
      DDSolid backCoolVFESolid ( DDSolidFactory::box( backCoolVFEName(), 
						      backCoolBarHeight()/2.,
						      backCoolBarWidth()/2.,  
						      halfZCoolVFE   ) ) ;
      const DDLogicalPart backCoolVFELog ( backCoolVFEName(),
					   backCoolVFEMat(),
					   backCoolVFESolid ) ;
      if( 0 != backCoolBarHere() )
	 cpv.position( backCoolBarLog    ,
		backCoolVFEName() , 
		copyOne           , 
		DDTranslation()   ,
		DDRotation()       ) ;
      if( 0 != backCoolVFEHere() )
	 cpv.position( backVFELog        ,
		backCoolVFEName() , 
		copyOne           , 
		DDTranslation( 0,0, backCoolBarThick()/2. + thickVFE/2. )   ,
		DDRotation()       ) ;
      cpv.position( backVFELog        ,
	     backCoolVFEName() , 
	     copyTwo           , 
	     DDTranslation( 0,0, -backCoolBarThick()/2. - thickVFE/2. )   ,
	     myrot( backVFEName().name() + "Flip",
		    CLHEP::HepRotationX( 180*deg ) )        ) ;

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
      unsigned int iCVFECopy ( 1 ) ;
      unsigned int iSep ( 0 ) ;
      unsigned int iNSec ( 0 ) ;
      const unsigned int nMisc ( vecBackMiscThick().size()/4 ) ;
      for( unsigned int iMod ( 0 ) ; iMod != 4 ; ++iMod )
      {
	 const double pipeLength ( vecGrilleZOff()[2*iMod+1] -
				   vecGrilleZOff()[2*iMod  ] -
				   grilleThick()   - 3*mm            ) ;

	 const double pipeZPos ( vecGrilleZOff()[2*iMod+1] - pipeLength/2 - 1.5*mm  ) ;



	 // accumulate total height of parent volume

	 double backCoolHeight ( backCoolBarHeight() + mBCoolTubeOutDiam() ) ;
	 for( unsigned int iMisc ( 0 ) ; iMisc != nMisc ; ++iMisc )
	 {
	    backCoolHeight += vecBackMiscThick()[ iMod*nMisc + iMisc ] ;
	 }
	 double bottomThick ( mBCoolTubeOutDiam() ) ;
	 for( unsigned int iMB ( 0 ) ; iMB != vecMBLyrThick().size() ; ++iMB )
	 {
	    backCoolHeight += vecMBLyrThick()[iMB] ;
	    bottomThick    += vecMBLyrThick()[iMB] ;
	 }

	 DDName backCName ( ddname( vecBackCoolName()[iMod] ) ) ;
	 const double halfZBCool ( ( pipeLength - 2*mBManifOutDiam() - grilleZSpace() )/2 ) ;
	 DDSolid backCoolSolid ( DDSolidFactory::box( backCName ,
						      backCoolHeight/2.,  
						      backCoolWidth/2.,
						      halfZBCool   ) ) ;
	 const DDLogicalPart backCoolLog ( backCName,
					   spmMat(),
					   backCoolSolid ) ;
	 
	 const DDTranslation bCoolTra ( -realBPthick/2 +
					backCoolHeight/2    -
					vecGrilleHeight()[2*iMod],
					deltaY,
					vecGrilleZOff()[2*iMod] +
					grilleThick() + grilleZSpace() +
					halfZBCool - 
					backSideLength()/2 ) ;
	 if( 0 != backCoolHere() )
	    cpv.position( backCoolLog,
		   spmName(), 
		   iMod+1, 
		   outtra + backPlateTra + bCoolTra,
		   DDRotation() ) ;

//===
	 const double backCoolTankHeight ( backCoolBarHeight() ) ;// - backBracketHeight() ) ;

	 const double halfZTank ( halfZBCool - 5*cm ) ;

	 DDName bTankName ( ddname( backCoolTankName()+std::to_string(iMod+1) ) ) ;
	 DDSolid backCoolTankSolid ( DDSolidFactory::box( bTankName ,
							  backCoolTankHeight/2.,  
							  backCoolTankWidth()/2.,
							  halfZTank  ) ) ;
	 const DDLogicalPart backCoolTankLog ( bTankName,
					       backCoolTankMat(),
					       backCoolTankSolid ) ;
	 if( 0 != backCoolTankHere() )
	    cpv.position( backCoolTankLog,
		   backCName, 
		   copyOne, 
		   DDTranslation( -backCoolHeight/2 + 
				  backCoolTankHeight/2. + 
				  bottomThick,
				  backCoolBarWidth()/2. + backCoolTankWidth()/2., 0),
		   DDRotation() ) ;

	 DDName bTankWaName ( ddname( backCoolTankWaName()+std::to_string(iMod+1) ) ) ;
	 DDSolid backCoolTankWaSolid ( DDSolidFactory::box( bTankWaName ,
							    backCoolTankHeight/2. -
							    backCoolTankThick()/2.,  
							    backCoolTankWaWidth()/2.,
							    halfZTank -
							    backCoolTankThick()/2. ) ) ;
	 const DDLogicalPart backCoolTankWaLog ( bTankWaName,
						 backCoolTankWaMat(),
						 backCoolTankWaSolid ) ;
	 cpv.position( backCoolTankWaLog,
		bTankName, 
		copyOne, 
		DDTranslation(0,0,0),
		DDRotation() ) ;

	 DDName bBracketName ( ddname( backBracketName()+std::to_string(iMod+1) ) ) ;
	 DDSolid backBracketSolid ( DDSolidFactory::box( bBracketName ,
							 backBracketHeight()/2.,  
							 backCoolTankWidth()/2.,
							 halfZTank   ) ) ;
	 const DDLogicalPart backBracketLog ( bBracketName,
					      backBracketMat(),
					      backBracketSolid ) ;
	 if( 0 != backCoolTankHere() )
	    cpv.position( backBracketLog,
		   backCName, 
		   copyOne, 
		   DDTranslation( backCoolBarHeight() - 
				  backCoolHeight/2. - 
				  backBracketHeight()/2. +
				  bottomThick,
				  -backCoolBarWidth()/2. - backCoolTankWidth()/2., 0),
		   DDRotation() ) ;

/*	 cpv.position( backBracketLog,
		backCName, 
		copyTwo, 
		DDTranslation( backCoolBarHeight() - backCoolHeight/2. - backBracketHeight()/2.,
			       backCoolBarWidth()/2. + backCoolTankWidth()/2., 0),
			       DDRotation() ) ;*/

//===

	 DDTranslation bSumTra ( backCoolBarHeight() - 
				 backCoolHeight/2. +
				 bottomThick, 0, 0 ) ;
	 for( unsigned int j ( 0 ) ; j != nMisc ; ++j ) // loop over miscellaneous layers
	 {
	    const DDName bName ( ddname( vecBackMiscName()[ iMod*nMisc + j ] ) ) ;

	    DDSolid bSolid ( DDSolidFactory::box( bName ,
						  vecBackMiscThick()[ iMod*nMisc + j ]/2,  
						  backCoolBarWidth()/2. + backCoolTankWidth(),
						  halfZBCool ) ) ;

	    const DDLogicalPart bLog ( bName, ddmat(vecBackMiscMat()[ iMod*nMisc + j ]), bSolid ) ;
 
	    const DDTranslation bTra ( vecBackMiscThick()[ iMod*nMisc + j ]/2, 0*mm, 0*mm ) ;


	    if( 0 != backMiscHere() )
	       cpv.position( bLog,
		      backCName, 
		      copyOne, 
		      bSumTra + bTra,
		      DDRotation() ) ;

	    bSumTra += 2*bTra ;
	 }

	 const double bHalfWidth ( backCoolBarWidth()/2. + backCoolTankWidth() ) ;

	 if( 0 != mBLyrHere() )
	 {
	    DDTranslation mTra ( -backCoolHeight/2. + mBCoolTubeOutDiam(), 0, 0 ) ;
	    for( unsigned int j ( 0 ) ; j != vecMBLyrThick().size() ; ++j ) // loop over MB layers
	    {
	       const DDName mName ( ddname( vecMBLyrName()[j] + "_" + 
					    std::to_string(iMod+1) ) ) ;

	       DDSolid mSolid ( DDSolidFactory::box( mName ,
						     vecMBLyrThick()[j]/2,  
						     bHalfWidth,
						     halfZBCool ) ) ;

	       const DDLogicalPart mLog ( mName, ddmat(vecMBLyrMat()[j]), mSolid ) ;
 
	       mTra += DDTranslation( vecMBLyrThick()[j]/2.0, 0*mm, 0*mm ) ;
	       cpv.position( mLog,
		      backCName, 
		      copyOne, 
		      mTra,
		      DDRotation() ) ;
	       mTra += DDTranslation( vecMBLyrThick()[j]/2.0, 0*mm, 0*mm ) ;
	    }
	 }

	 if( 0 != mBCoolTubeHere() )
	 {
	    const DDName mBName ( ddname( mBCoolTubeName() + "_" + 
					  std::to_string(iMod+1) ) ) ;
	    
	    DDSolid mBCoolTubeSolid ( DDSolidFactory::tubs( mBName ,
							    halfZBCool,
							    0, 
							    mBCoolTubeOutDiam()/2,
							    0*deg, 360*deg ) ) ;
	    const DDLogicalPart mBLog ( mBName, mBCoolTubeMat(), mBCoolTubeSolid ) ;

	    const DDName mBWaName ( ddname( mBCoolTubeName() + "Wa_" + 
					    std::to_string(iMod+1) ) ) ;
	    DDSolid mBCoolTubeWaSolid ( DDSolidFactory::tubs( mBWaName ,
							      halfZBCool,
							      0, 
							      mBCoolTubeInnDiam()/2,
							      0*deg, 360*deg ) ) ;
	    const DDLogicalPart mBWaLog ( mBWaName, backPipeWaterMat(), 
					  mBCoolTubeWaSolid ) ;
	    cpv.position( mBWaLog,
		   mBName, 
		   copyOne, 
		   DDTranslation(0,0,0),
		   DDRotation() ) ;

	    for( unsigned int j ( 0 ) ; j != mBCoolTubeNum() ; ++j ) // loop over all MB cooling circuits
	    {
	       cpv.position( mBLog,
		      backCName, 
		      2*j + 1, 
		      DDTranslation(-backCoolHeight/2.0+mBCoolTubeOutDiam()/2.,
				    -bHalfWidth + (j+1)*bHalfWidth/5, 0),
		      DDRotation() ) ;
	    }
	 }


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!! Begin Back Water Pipes   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	 if( 0 != backPipeHere() &&
	     0 != iMod               )
	 {
	    DDName bPipeName ( ddname( backPipeName() + "_" + 
				       std::to_string( iMod+1 ) ) ) ; 
	    DDName bInnerName ( ddname( backPipeName() + 
					"_H2O_" + std::to_string( iMod+1 ) ) ) ; 
	 
	    DDSolid backPipeSolid ( DDSolidFactory::tubs( bPipeName ,
							  pipeLength/2,
							  0*mm, 
							  vecBackPipeDiam()[iMod]/2,
							  0*deg, 360*deg ) ) ;

	    DDSolid backInnerSolid ( DDSolidFactory::tubs( bInnerName ,
							   pipeLength/2,
							   0*mm, 
							   vecBackPipeDiam()[iMod]/2 -
							   vecBackPipeThick()[iMod],
							   0*deg, 360*deg ) ) ;

	    const DDLogicalPart backPipeLog ( bPipeName, 
					      backPipeMat(), 
					      backPipeSolid ) ;

	    const DDLogicalPart backInnerLog ( bInnerName, 
					       backPipeWaterMat(), 
					       backInnerSolid ) ;

	    const DDTranslation bPipeTra1 ( backXOff() + 
					    backSideHeight() -
					    0.7*vecBackPipeDiam()[iMod],
					    backYOff() +
					    backPlateWidth()/2 -
					    backSideWidth() -
					    0.7*vecBackPipeDiam()[iMod],
					    pipeZPos ) ;

	    cpv.position( backPipeLog,
		   spmName(), 
		   copyOne, 
		   bPipeTra1,
		   DDRotation() ) ;

	    const DDTranslation bPipeTra2 ( bPipeTra1.x(),
					    backYOff() -
					    backPlateWidth()/2 +
					    backSideWidth() +
					    vecBackPipeDiam()[iMod],
					    bPipeTra1.z()  ) ;

	    cpv.position( backPipeLog,
		   spmName(), 
		   copyTwo, 
		   bPipeTra2,
		   DDRotation() ) ;

	    cpv.position( backInnerLog,
		   bPipeName, 
		   copyOne, 
		   DDTranslation(),
		   DDRotation() ) ;
	 }
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!! End Back Water Pipes   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//=================================================

	 if( 0 != dryAirTubeHere() )
	 {
	   DDName dryAirTubName ( ddname( dryAirTubeName() + std::to_string( iMod+1 ) ) ) ; 

	    DDSolid dryAirTubeSolid ( DDSolidFactory::tubs( dryAirTubName ,
							    pipeLength/2,
							    dryAirTubeInnDiam()/2, 
							    dryAirTubeOutDiam()/2,
							    0*deg, 360*deg ) ) ;

	    const DDLogicalPart dryAirTubeLog ( dryAirTubName , 
						dryAirTubeMat(), 
						dryAirTubeSolid ) ;

	    const DDTranslation dryAirTubeTra1 ( backXOff() + 
						 backSideHeight() -
						 0.7*dryAirTubeOutDiam() -
						 vecBackPipeDiam()[iMod],
						 backYOff() +
						 backPlateWidth()/2 -
						 backSideWidth() -
						 1.2*dryAirTubeOutDiam(),
						 pipeZPos ) ;

	    cpv.position( dryAirTubeLog,
		   spmName(), 
		   copyOne, 
		   dryAirTubeTra1,
		   DDRotation() ) ;

	    const DDTranslation dryAirTubeTra2 ( dryAirTubeTra1.x(),
						 backYOff() -
						 backPlateWidth()/2 +
						 backSideWidth() +
						 0.7*dryAirTubeOutDiam() ,
						 dryAirTubeTra1.z()  ) ;
	 
	    cpv.position( dryAirTubeLog,
		   spmName(), 
		   copyTwo, 
		   dryAirTubeTra2,
		   DDRotation() ) ;
	 }
	 //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

         //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         //!!!!!!!!!!!!!! Begin Placement of Cooling + VFE Cards          !!!!!!
	 //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	 //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	 DDTranslation cTra ( backCoolBarHeight()/2. - 
			      backCoolHeight/2. +
			      bottomThick, 0 ,
			      -halfZTank + halfZCoolVFE ) ;
	 const unsigned int numSec ( static_cast<unsigned int> (vecBackCoolNSec()[iMod]) ) ; 
	 for( unsigned int jSec ( 0 ) ; jSec != numSec ; ++jSec )
	 {
	    const unsigned int nMax ( static_cast<unsigned int> (vecBackCoolNPerSec()[iNSec++]) ) ; 
	    for( unsigned int iBar ( 0 ) ; iBar !=  nMax ; ++iBar )
	    {
	       cpv.position( backCoolVFELog,
		      backCName, 
		      iCVFECopy++, 
		      cTra,
		      DDRotation() ) ;	       
	       cTra += DDTranslation( 0, 0, backCBStdSep() ) ;
	    }
	    cTra -= DDTranslation( 0, 0, backCBStdSep() ) ; // backspace to previous
	    if( jSec != numSec-1 ) cTra += DDTranslation( 
	       0, 0, vecBackCoolSecSep()[iSep++] ) ; // now take atypical step
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

      double patchHeight ( 0  ) ;
      for( unsigned int iPatch ( 0 ) ; iPatch != vecPatchPanelThick().size() ; ++iPatch )
      {
	 patchHeight += vecPatchPanelThick()[iPatch] ;
      }

      DDSolid patchSolid ( DDSolidFactory::box( patchPanelName() ,
						patchHeight/2.,  
						backCoolBarWidth()/2.,
						( vecSpmZPts().back() -
						  vecGrilleZOff().back() -
						  grilleThick() )/2   ) ) ;

      const std::vector<double>& patchParms ( patchSolid.parameters() ) ;

      const DDLogicalPart patchLog ( patchPanelName(), spmMat(), patchSolid ) ;
	 
      const DDTranslation patchTra ( backXOff() + 4*mm ,
				     0*mm,
				     vecGrilleZOff().back() +
				     grilleThick() +
				     patchParms[2]  ) ;
      if( 0 != patchPanelHere() )
	 cpv.position( patchLog,
		spmName(), 
		copyOne, 
		patchTra,
		DDRotation() ) ;

      DDTranslation pTra (-patchParms[0],0,0) ;

      for( unsigned int j ( 0 ) ; j != vecPatchPanelNames().size() ; ++j )
      {
	 const DDName pName ( ddname( vecPatchPanelNames()[j] ) ) ;

	 DDSolid pSolid ( DDSolidFactory::box( pName ,
					       vecPatchPanelThick()[j]/2.,  
					       patchParms[1],
					       patchParms[2] ) ) ;
	 
	 const DDLogicalPart pLog ( pName, ddmat(vecPatchPanelMat()[j]), pSolid ) ;
	 
	 pTra += DDTranslation( vecPatchPanelThick()[j]/2, 0*mm, 0*mm ) ;
	 
	 cpv.position( pLog,
		patchPanelName(), 
		copyOne, 
		pTra,
		DDRotation() ) ;
	 
	 pTra += DDTranslation( vecPatchPanelThick()[j]/2, 0*mm, 0*mm ) ;
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

      if( 0 != pincerRodHere() )
      {
	 // Make hierarchy of rods, envelopes, blocks, shims, and cutouts

	 DDSolid rodSolid ( DDSolidFactory::box( pincerRodName() ,
						 pincerEnvWidth()/2.,  
						 pincerEnvHeight()/2.,
						 ilyLength/2   ) ) ;
	 const DDLogicalPart rodLog ( pincerRodName(), pincerRodMat(), rodSolid ) ;

	 DDSolid envSolid ( DDSolidFactory::box( pincerEnvName() ,
						 pincerEnvWidth()/2.,  
						 pincerEnvHeight()/2.,
						 pincerEnvLength()/2   ) ) ;
	 const DDLogicalPart envLog ( pincerEnvName(), pincerEnvMat(), envSolid ) ;
	 const std::vector<double>& envParms ( envSolid.parameters() ) ;

	 DDSolid blkSolid ( DDSolidFactory::box( pincerBlkName() ,
						 pincerEnvWidth()/2.,  
						 pincerEnvHeight()/2.,
						 pincerBlkLength()/2   ) ) ;
	 const DDLogicalPart blkLog ( pincerBlkName(), pincerBlkMat(), blkSolid ) ;
	 const std::vector<double>& blkParms ( blkSolid.parameters() ) ;
	 cpv.position( blkLog,
		pincerEnvName(), 
		copyOne,
		DDTranslation(0,0, pincerEnvLength()/2 - pincerBlkLength()/2 ),
		DDRotation() ) ;

	 DDSolid cutSolid ( DDSolidFactory::box( pincerCutName() ,
						 pincerCutWidth()/2.,  
						 pincerCutHeight()/2.,
						 pincerBlkLength()/2   ) ) ;
	 const DDLogicalPart cutLog ( pincerCutName(), pincerCutMat(), cutSolid ) ;
	 const std::vector<double>& cutParms ( cutSolid.parameters() ) ;
	 cpv.position( cutLog,
		pincerBlkName(), 
		copyOne,
		DDTranslation( +blkParms[0] - cutParms[0] - pincerShim1Width() + pincerShim2Width(),
			       -blkParms[1] + cutParms[1],     0  ),
		DDRotation() ) ;

	 DDSolid shim2Solid ( DDSolidFactory::box( pincerShim2Name() ,
						   pincerShim2Width()/2.,  
						   pincerShimHeight()/2.,
						   pincerBlkLength()/2   ) ) ;
	 const DDLogicalPart shim2Log ( pincerShim2Name(), pincerShimMat(), shim2Solid ) ;
	 const std::vector<double>& shim2Parms ( shim2Solid.parameters() ) ;
	 cpv.position( shim2Log,
		pincerCutName(), 
		copyOne,
		DDTranslation( +cutParms[0] - shim2Parms[0],
			       -cutParms[1] + shim2Parms[1],     0  ),
		DDRotation() ) ;

	 DDSolid shim1Solid ( DDSolidFactory::box( pincerShim1Name() ,
						   pincerShim1Width()/2.,  
						   pincerShimHeight()/2.,
						   (pincerEnvLength()-
						    pincerBlkLength())/2   ) ) ;

	 const DDLogicalPart shim1Log ( pincerShim1Name(), pincerShimMat(), shim1Solid ) ;
	 const std::vector<double>& shim1Parms ( shim1Solid.parameters() ) ;
	 cpv.position( shim1Log,
		pincerEnvName(), 
		copyOne,
		DDTranslation( +envParms[0] - shim1Parms[0],
			       -envParms[1] + shim1Parms[1],
			       -envParms[2] + shim1Parms[2] ),
		DDRotation() ) ;

	 for( unsigned int iEnv ( 0 ) ; iEnv != vecPincerEnvZOff().size() ; ++iEnv )
	 {
	    cpv.position( envLog,
		   pincerRodName(), 
		   1+iEnv, 
		   DDTranslation(0,0, -ilyLength/2. + vecPincerEnvZOff()[iEnv] - pincerEnvLength()/2. ),
		   DDRotation() ) ;
	 }

	 // Place the rods
//	 const double radius ( fawRadOff() - pincerEnvHeight()/2 -1*mm ) ;
	 const double radius ( ilyRMin - pincerEnvHeight()/2 - 1*mm ) ;

	 const DDName        xilyName    ( ddname( ilyName() +
						   std::to_string(vecIlyMat().size()-1) ) ) ;

	 for( unsigned int iRod ( 0 ) ; iRod != vecPincerRodAzimuth().size() ; ++iRod )
	 {
	   const DDTranslation rodTra ( radius*cos(vecPincerRodAzimuth()[iRod]) ,
					radius*sin(vecPincerRodAzimuth()[iRod]) ,
					0 ) ;
	   
	   cpv.position( rodLog,
		  xilyName, 
		  1+iRod, 
		  rodTra,
			 myrot( pincerRodName().name() + std::to_string(iRod),
			 CLHEP::HepRotationZ( 90*deg + vecPincerRodAzimuth()[iRod] ) ) ) ;
	 }
      }
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!! End   Pincers       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      }
   } 

   LogDebug("EcalGeom") << "******** DDEcalBarrelAlgo test: end it..." ;
}

///Create a DDRotation from a string converted to DDName and CLHEP::HepRotation converted to DDRotationMatrix. -- Michael Case
DDRotation
DDEcalBarrelAlgo::myrot( const std::string&      s,
			 const CLHEP::HepRotation& r ) const 
{
  return DDrot( ddname( m_idNameSpace + ":" + s ), std::make_unique<DDRotationMatrix>( r.xx(), r.xy(), r.xz(), r.yx(), r.yy(), r.yz(), r.zx(), r.zy(), r.zz() ) ) ; 
}

 
DDMaterial
DDEcalBarrelAlgo::ddmat( const std::string& s ) const
{
   return DDMaterial( ddname( s ) ) ; 
}

DDName
DDEcalBarrelAlgo::ddname( const std::string& s ) const
{ 
   const pair<std::string,std::string> temp ( DDSplit(s) ) ;
   if ( temp.second.empty() ) {
     return DDName( temp.first,
		    m_idNameSpace ) ;
   } else {
     return DDName( temp.first, temp.second );
   } 
}  

DDSolid    
DDEcalBarrelAlgo::mytrap( const std::string& s,
			  const EcalTrapezoidParameters& t ) const
{
   return DDSolidFactory::trap( ddname( s ),
				t.dz(), 
				t.theta(), 
				t.phi(), 
				t.h1(), 
				t.bl1(), 
				t.tl1(),
				t.alp1(), 
				t.h2(), 
				t.bl2(), 
				t.tl2(), 
				t.alp2()         ) ;
}

void 
DDEcalBarrelAlgo::web( unsigned int        iWeb,
		       double              bWeb,
		       double              BWeb,
		       double              LWeb,
		       double              theta,
		       const HepGeom::Point3D<double> &   corner,
		       const DDLogicalPart& logPar,
		       double&             zee,
		       double              side,
		       double              front,
		       double              delta,
  		       DDCompactView&      cpv  )
{
   const unsigned int copyOne (1) ;

   const double LWebx ( vecWebLength()[iWeb] ) ;

   const double BWebx ( bWeb + ( BWeb - bWeb )*LWebx/LWeb ) ;

   const double thick ( vecWebPlTh()[iWeb] + vecWebClrTh()[iWeb] ) ;
   const Trap trapWebClr (
      BWebx/2,        // A/2 
      bWeb/2,        // a/2
      bWeb/2,        // b/2
      thick/2,       // H/2
      thick/2,       // h/2
      LWebx/2,        // L/2
      90*deg,        // alfa1
      bWeb - BWebx ,  // x15
      0              // y15
      ) ;
   const DDName        webClrDDName ( webClrName() + std::to_string( iWeb ) ) ;
   const DDSolid       webClrSolid  ( mytrap( webClrDDName.name(), trapWebClr ) ) ;
   const DDLogicalPart webClrLog    ( webClrDDName, webClrMat(), webClrSolid ) ;
	    
   const Trap trapWebPl (
      trapWebClr.A()/2,                     // A/2 
      trapWebClr.a()/2,                     // a/2
      trapWebClr.b()/2,                     // b/2
      vecWebPlTh()[iWeb]/2,                 // H/2
      vecWebPlTh()[iWeb]/2,                 // h/2
      trapWebClr.L()/2.,                    // L/2
      90*deg,                               // alfa1
      trapWebClr.b() - trapWebClr.B() ,     // x15
      0                                     // y15
      ) ;
   const DDName        webPlDDName  ( webPlName() + std::to_string( iWeb ) ) ;
   const DDSolid       webPlSolid   ( mytrap( webPlDDName.fullname(), trapWebPl ) ) ;
   const DDLogicalPart webPlLog     ( webPlDDName, webPlMat(), webPlSolid ) ;

   cpv.position( webPlLog,     // place plate inside clearance volume
	  webClrDDName, 
	  copyOne, 
	  DDTranslation(0,0,0),
	  DDRotation() ) ;

   const Trap::VertexList vWeb ( trapWebClr.vertexList() ) ;

   zee += trapWebClr.h()/sin(theta) ;

   const double beta ( theta + delta ) ;

   const double zWeb ( zee - front*cos(beta) + side*sin(beta) ) ;
   const double yWeb ( front*sin(beta) + side*cos(beta) ) ;

   const Pt3D wedge3 ( corner + Pt3D( 0, -yWeb, zWeb ) ) ;
   const Pt3D wedge2 ( wedge3 + Pt3D( 0,
				      trapWebClr.h()*cos(theta),
				      -trapWebClr.h()*sin(theta)  ) ) ;
   const Pt3D wedge1 ( wedge3 + Pt3D( trapWebClr.a(), 0, 0 ) ) ;

   LogDebug("EcalGeom")<<"trap1="<<vWeb[0]<<", trap2="<<vWeb[2]<<", trap3="<<vWeb[3] ;

   LogDebug("EcalGeom")<<"wedge1="<<wedge1<<", wedge2="<<wedge2<<", wedge3="<<wedge3 ;

   const Tf3D tForm ( vWeb[0], vWeb[2], vWeb[3],
		      wedge1,   wedge2, wedge3    ) ;
   
   if( 0 != webHere() )
      cpv.position( webClrLog,
	     logPar, 
	     copyOne, 
	     DDTranslation(tForm.getTranslation().x(), tForm.getTranslation().y(), tForm.getTranslation().z()),
		    myrot( webClrLog.name().name() + std::to_string( iWeb ),
		    tForm.getRotation() ) ) ;
}
