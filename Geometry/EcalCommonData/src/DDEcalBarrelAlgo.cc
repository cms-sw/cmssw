//////////////////////////////////////////////////////////////////////////////
// File: DDEcalBarrelAlgo.cc
// Description: Geometry factory class for Ecal Barrel
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/EcalCommonData/interface/DDEcalBarrelAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>

DDEcalBarrelAlgo::DDEcalBarrelAlgo() :
  m_idNameSpace  ( "" ),
  m_BarName      ( "" ),
  m_BarMat       ( "" ),
  m_vecBarZPts   (    ),
  m_vecBarRMin   (    ),
  m_vecBarRMax   (    ),
  m_vecBarTran   (    ),
  m_vecBarRota   (    ),
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
  m_WebPlName    (""),    
  m_WebClrName   (""),    
  m_WebPlMat     (""),
  m_WebClrMat    (""),
  m_vecWebPlTh   (),
  m_vecWebClrTh  (),
  m_vecWebLength (),
  m_IlyName      (),
  m_IlyPhiLow    (0),
  m_IlyDelPhi    (0),
  m_vecIlyMat    (),
  m_vecIlyThick  (),
  m_HawRName     (""),
  m_FawName      (""),
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
  m_GridName     (""),
  m_GridMat      (""),
  m_GridThick    (0),
  m_BackXOff     (0),
  m_BackYOff     (0),
  m_BackSideName          (""),
  m_BackSideLength        (0),
  m_BackSideHeight        (0),
  m_BackSideWidth         (0),
  m_BackSideYOff1        (0),
  m_BackSideYOff2        (0),
  m_BackSideAngle        (0),
  m_BackSideMat           (""),
  m_BackPlateName    (""),
  m_BackPlateLength  (0),
  m_BackPlateThick   (0),
  m_BackPlateWidth   (0),
  m_BackPlateMat     (""),
  m_GrilleName      (""),
  m_GrilleThick     (0),
  m_GrilleWidth     (0),
  m_GrilleZSpace    (0),
  m_GrilleMat       (""),
  m_vecGrilleHeight (),
  m_vecGrilleZOff   (),
  m_BackPipeName    (""),
  m_vecBackPipeDiam (),
  m_BackPipeThick   (0),
  m_BackPipeMat     (""),
  m_BackPipeWaterMat (""),

  m_vecBackCoolName       (),
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

  m_vecBackCoolLength(),
  m_vecBackMiscThick (),
  m_vecBackMiscName  (),
  m_vecBackMiscMat   (),
  m_BackCBStdSep         (0),
  m_PatchPanelName   (""),
  m_vecPatchPanelThick (),
  m_vecPatchPanelNames  (),
  m_vecPatchPanelMat   (),
 m_BackCoolTankName    (""),
 m_BackCoolTankWidth   (0),
 m_BackCoolTankThick   (0),
 m_BackCoolTankMat     (""),
 m_BackCoolTankWaName  (""),
 m_BackCoolTankWaWidth (0),
 m_BackCoolTankWaMat   (""),
 m_BackBracketName     (""),
 m_BackBracketHeight   (0),
 m_BackBracketMat      ("")
{
   edm::LogInfo("EcalGeom") << "DDEcalBarrelAlgo info: Creating an instance" ;
}

DDEcalBarrelAlgo::~DDEcalBarrelAlgo() {}




void DDEcalBarrelAlgo::initialize(const DDNumericArguments      & nArgs,
				  const DDVectorArguments       & vArgs,
				  const DDMapArguments          & mArgs,
				  const DDStringArguments       & sArgs,
				  const DDStringVectorArguments & vsArgs) {

   edm::LogInfo("EcalGeom") << "DDEcalBarrelAlgo info: Initialize" ;
   m_idNameSpace = DDCurrentNamespace::ns();

   // barrel parent volume
   m_BarName     = sArgs["BarName" ] ;
   m_BarMat      = sArgs["BarMat"  ] ;
   m_vecBarZPts  = vArgs["BarZPts" ] ;
   m_vecBarRMin  = vArgs["BarRMin" ] ;
   m_vecBarRMax  = vArgs["BarRMax" ] ;
   m_vecBarTran  = vArgs["BarTran" ] ;
   m_vecBarRota  = vArgs["BarRota" ] ;
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
   m_SpmCutShow  = nArgs["SpmCutShow"] ;
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

   m_WebPlName   = sArgs["WebPlName"] ;
   m_WebClrName  = sArgs["WebClrName"] ;
   m_WebPlMat    = sArgs["WebPlMat"] ;
   m_WebClrMat   = sArgs["WebClrMat"] ;
   m_vecWebPlTh  = vArgs["WebPlTh"] ;
   m_vecWebClrTh = vArgs["WebClrTh"] ;
   m_vecWebLength= vArgs["WebLength"] ;

   m_IlyName     = sArgs["IlyName"] ;
   m_IlyPhiLow   = nArgs["IlyPhiLow"] ;
   m_IlyDelPhi   = nArgs["IlyDelPhi"] ;
   m_vecIlyMat   = vsArgs["IlyMat"] ;
   m_vecIlyThick = vArgs["IlyThick"] ;

   m_HawRName   = sArgs["HawRName"] ;
   m_FawName    = sArgs["FawName"] ;
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

   m_GridName   = sArgs["GridName"]  ;
   m_GridMat    = sArgs["GridMat"]   ;
   m_GridThick  = nArgs["GridThick"] ;

   m_BackXOff         = nArgs["BackXOff"] ;
   m_BackYOff         = nArgs["BackYOff"] ;
   m_BackSideName     = sArgs["BackSideName"] ;
   m_BackSideLength   = nArgs["BackSideLength"] ;
   m_BackSideHeight   = nArgs["BackSideHeight"] ;
   m_BackSideWidth    = nArgs["BackSideWidth"] ;
   m_BackSideYOff1    = nArgs["BackSideYOff1"] ;
   m_BackSideYOff2    = nArgs["BackSideYOff2"] ;
   m_BackSideAngle    = nArgs["BackSideAngle"] ;
   m_BackSideMat      = sArgs["BackSideMat"] ;
   m_BackPlateName    = sArgs["BackPlateName"] ;
   m_BackPlateLength  = nArgs["BackPlateLength"] ;
   m_BackPlateThick   = nArgs["BackPlateThick"] ;
   m_BackPlateWidth   = nArgs["BackPlateWidth"] ;
   m_BackPlateMat     = sArgs["BackPlateMat"] ;
   m_GrilleName       = sArgs["GrilleName"] ;
   m_GrilleThick      = nArgs["GrilleThick"] ;
   m_GrilleWidth      = nArgs["GrilleWidth"] ;
   m_GrilleZSpace     = nArgs["GrilleZSpace"] ;
   m_GrilleMat        = sArgs["GrilleMat"] ;
   m_vecGrilleHeight  = vArgs["GrilleHeight"] ;
   m_vecGrilleZOff    = vArgs["GrilleZOff"] ;
   m_BackPipeName    = sArgs["BackPipeName"] ;
   m_vecBackPipeDiam = vArgs["BackPipeDiam"] ;
   m_BackPipeThick   = nArgs["BackPipeThick"] ;
   m_BackPipeMat     = sArgs["BackPipeMat"] ;
   m_BackPipeWaterMat = sArgs["BackPipeWaterMat"] ;


   m_vecBackCoolName       = vsArgs["BackCoolName"] ;
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

   m_vecBackCoolLength  = vArgs["BackCoolLength"] ;
   m_vecBackMiscThick   = vArgs["BackMiscThick"] ;
   m_vecBackMiscName    = vsArgs["BackMiscName"] ;
   m_vecBackMiscMat     = vsArgs["BackMiscMat"] ;
   m_vecPatchPanelThick = vArgs["PatchPanelThick"] ;
   m_vecPatchPanelNames = vsArgs["PatchPanelNames"] ;
   m_vecPatchPanelMat   = vsArgs["PatchPanelMat"] ;
   m_PatchPanelName     = sArgs["PatchPanelName"] ;

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

   edm::LogInfo("EcalGeom") << "DDEcalBarrelAlgo info: end initialize" ;
}

////////////////////////////////////////////////////////////////////
// DDEcalBarrelAlgo methods...
////////////////////////////////////////////////////////////////////

void DDEcalBarrelAlgo::execute() 
{
   edm::LogInfo("EcalGeom") << "******** DDEcalBarrelAlgo execute!" << std::endl ;

   if( barHere() != 0 )
   {
      const unsigned int copyOne (1) ;
      const unsigned int copyTwo (2) ;
      // Barrel parent volume----------------------------------------------------------
      DDpos( DDLogicalPart( barName(), barMat(), 
			    DDSolidFactory::polycone(
			       barName(), 0, 360.*deg, 
			       vecBarZPts(), vecBarRMin(), vecBarRMax())),
	     parent().name() , 
	     copyOne, 
	     DDTranslation(vecBarTran()[0],
			   vecBarTran()[1],
			   vecBarTran()[2]), 
	     myrot(barName().name()+"Rot",
		   Rota(Vec3(vecBarRota()[0],
			     vecBarRota()[1],
			     vecBarRota()[2]),
			vecBarRota()[3]))) ;
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
				   fabs( vecSpmZPts().back() - vecSpmZPts().front() )/2.) ) ;
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

/*      const DDName unionName ( ddname( m_SpmName + "UNI" ) ) ;
      if( 0 != spmCutShow() )
      {
	 DDpos( expLog, spmName(), copyOne, expTra, DDRotation() ) ;
      }
      else
      {
	 const DDSolid unionSolid ( DDSolidFactory::unionSolid(
				       unionName,
				       spmcut1ddname, expName,
				       expTra, DDRotation() ) ) ;
				       }*/


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
		  vecSpmZPts().front()+cutBoxParms[2])*
	    tr*ro) ;

	 const DDRotation    ddrot ( myrot(spmCutName().name() + 
					   dbl_to_string(icopy),
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
					       dbl_to_string(icopy),
					       sideRot.getRotation() ) ) ;
	 const DDTranslation sideddtra ( sideRot.getTranslation() ) ;

	 DDpos( sideLog,
		spmName(), 
		icopy, 
		sideddtra,
		sideddrot ) ;


	 if( 0 != spmCutShow() ) // do this if we are "showing" the boxes
	 {
	    DDpos( spmCutLog,
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

	 const DDRotation rota ( myrot( spmName().name()+dbl_to_string(phi/deg), 
					both.getRotation() ) );

	 if( vecSpmHere()[iphi] != 0 )
	 {
	    DDpos( spmLog,
		   barName(), 
		   iphi+1, 
		   both.getTranslation(),
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
      DDpos( ilyLog,
	     spmLog, 
	     copyOne, 
	     Vec3(0,0, ilyLength/2 ),
	     DDRotation() ) ;

      for( unsigned int ily ( 0 ) ; ily != vecIlyThick().size() ; ++ily )
      {
	 const double        ilyRMax     ( ilyRMin + vecIlyThick()[ily] ) ;
	 const DDName        xilyName    ( ddname( ilyName() + int_to_string(ily) ) ) ;
	 const DDSolid       xilySolid   ( DDSolidFactory::tubs( xilyName,
								 ilyLength/2,
								 ilyRMin,
								 ilyRMax,
								 ilyPhiLow(),
								 ilyDelPhi() ) ) ;

	 const DDLogicalPart xilyLog     ( xilyName, ddmat(vecIlyMat()[ily]), xilySolid ) ;

	 DDpos( xilyLog,
		ilyLog, 
		copyOne, 
		Vec3(0,0,0),
		DDRotation() ) ;

	 ilyRMin = ilyRMax ;
      }      
      // End Inner Layer volumes---------------------------------------------------------



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
					 hawCutForm.getTranslation(),
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
					 fawCutForm.getTranslation(),
					 myrot( fawCutName.name()+"R",
						fawCutForm.getRotation() ) ) ) ;
      const DDLogicalPart fawLog   ( fawName(), spmMat(), fawSolid ) ;


      const Tf3D hawRform ( vHAW[3], vHAW[0], vHAW[1], // HAW inside FAW
			    vFAW[3], 0.5*(vFAW[0]+vFAW[3]), 0.5*(vFAW[1]+vFAW[2] ) ) ;

      DDpos( hawRLog,
	     fawLog, 
	     copyOne, 
	     hawRform.getTranslation(),
	     myrot( hawRName().name()+"R", 
		    hawRform.getRotation() ) ) ;

      DDpos( hawRLog,
	     fawLog, 
	     copyTwo, 
	     Vec3( -hawRform.getTranslation().x(),
		   -hawRform.getTranslation().y(),
		   -hawRform.getTranslation().z() ),
	     myrot( hawRName().name()+"RotRefl",
		    HepRotationY(180*deg)*                 // rotate about Y after refl thru Z
		    HepRep3x3(1,0,0, 0,1,0, 0,0,-1) ) ) ;

/* this for display of haw cut box instead of subtraction
      DDpos( hawCutLog,
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
	 DDpos( fawLog,
		spmLog, 
		iPhi, 
		fawform.getTranslation(),
		myrot( fawName().name()+"_Rot" + int_to_string(iPhi), 
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

      DDpos( gridLog,
	     hawRLog, 
	     copyOne, 
	     gridForm.getTranslation(),
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
				   int_to_string( cryType ) ) ;

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
	    aNom - ANom ,                  //double aCoord15X           , // x15
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
	    aNom - ANom ,                        // x15
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
	 const Vec3 cryToClr ( 0,0, ( rClr - fClr )/2 ) ;

	 DDpos( cryLog,
		clrLog, 
		copyOne, 
		cryToClr,
		DDRotation() ) ;

	 const Vec3 clrToWrap ( 0, 0, ( rWrap - fWrap )/2 ) ;

	 DDpos( clrLog,
		wrapLog, 
		copyOne, 
		clrToWrap,
		DDRotation() ) ;


	 // Now for placement of clr within wall
	 const Vec3 wrapToWall ( 0, 0, ( rWall - fWall )/2 ) ;

	 DDpos( wrapLog,
		wallLog, 
		copyOne, 
		wrapToWall,
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
		 delta ) ;
	    zee += vecGapAlvEta()[0] ;
	 }

	 for( unsigned int etaAlv ( 1 ) ; etaAlv <= nCryPerAlvEta() ; ++etaAlv )
	 {
	    LogDebug("EcalGeom") << "theta=" << theta/deg
				 << ", sidePrime=" << sidePrime << ", frontPrime=" << frontPrime
				 << ",  zeta="<<zeta<<", delta="<<delta<<",  zee=" << zee ;

	    zee += side*cos(zeta)/sin(theta) + 
	       ( trapWall.h() - sidePrime )/sin(theta) ;

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

	    const Tf3D tForm ( trap1,  trap2,  trap3,
			       wedge1, wedge2, wedge3    ) ;


	    DDpos( wallLog,
		   hawRLog, 
		   etaAlv, 
		   tForm.getTranslation(),
		   myrot( wallLog.name().name() + "_" + int_to_string( etaAlv ),
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
		 delta ) ;
	 }
	 if( 17 != cryType ) zee += vecGapAlvEta()[cryType]/sin(theta) ;
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
      
      const DDTranslation outtra ( backXOff() + backSideHeight()/2,
				   backYOff(),
				   backSideLength()/2 ) ;

      DDSolid backPlateSolid ( DDSolidFactory::box( backPlateName(), 
						    backPlateWidth()/2.,  
						    backPlateThick()/2.,
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
      DDpos( backPlateLog,
	     spmName(), 
	     copyOne, 
	     outtra + backPlateTra,
	     myrot( backPlateName().name()+"Rot5",
		    HepRotationZ(270*deg)    ) ) ;


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
					 0*mm ) ;
      DDpos( backSideLog,
	     spmName(), 
	     copyOne, 
	     outtra + backSideTra1,
	     myrot( backSideName().name()+"Rot8",
		    HepRotationX(180*deg)*HepRotationZ(90*deg)    ) ) ;
	     
      const DDTranslation backSideTra2( 0*mm,
					-backPlateWidth()/2 + backSideYOff2(),
					0*mm ) ;
      DDpos( backSideLog,
	     spmName(), 
	     copyTwo, 
	     outtra + backSideTra2,
	     myrot( backSideName().name()+"Rot9",
		    HepRotationZ(90*deg)    ) ) ;
	     
      for( unsigned int iGr ( 0 ) ; iGr != vecGrilleHeight().size() ; ++iGr )
      {
	 DDName gName ( ddname( grilleName() + int_to_string( iGr ) ) ) ;
	 DDSolid grilleSolid ( DDSolidFactory::box( gName, 
						    vecGrilleHeight()[iGr]/2.,  
						    grilleWidth()/2.,
						    grilleThick()/2.   ) ) ;
	 const DDLogicalPart grilleLog ( gName,
					 grilleMat(),
					 grilleSolid ) ;
	 
	 const DDTranslation grilleTra ( -backPlateThick()/2 -
					 vecGrilleHeight()[iGr]/2,
					 0*mm,
					 vecGrilleZOff()[iGr] +
					 grilleThick()/2 - backSideLength()/2 ) ;
	 DDpos( grilleLog,
		spmName(), 
		iGr, 
		outtra + backPlateTra + grilleTra,
		DDRotation() ) ;
      }

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
      DDpos( backCoolBarSSLog,
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
      DDpos( backCoolBarWaLog,
	     backCoolBarSSName(), 
	     copyOne, 
	     backCoolBarWaTra,
	     DDRotation() ) ;

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
	 DDpos( backVFELyrLog,
		backVFEName(), 
		copyOne, 
		backVFELyrTra + offTra,
		DDRotation() ) ;
	 offTra += 2*backVFELyrTra ;
      }
      const double halfZCoolVFE ( thickVFE + backCoolBarThick()/2. ) ;
      DDSolid backCoolVFESolid ( DDSolidFactory::box( backCoolVFEName(), 
						      backCoolBarHeight()/2.,
						      backCoolBarWidth()/2.,  
						      halfZCoolVFE   ) ) ;
      const DDLogicalPart backCoolVFELog ( backCoolVFEName(),
					   backCoolVFEMat(),
					   backCoolVFESolid ) ;
      DDpos( backCoolBarLog    ,
	     backCoolVFEName() , 
	     copyOne           , 
	     DDTranslation()   ,
	     DDRotation()       ) ;
      DDpos( backVFELog        ,
	     backCoolVFEName() , 
	     copyOne           , 
	     DDTranslation( 0,0, backCoolBarThick()/2. + thickVFE/2. )   ,
	     DDRotation()       ) ;
      DDpos( backVFELog        ,
	     backCoolVFEName() , 
	     copyTwo           , 
	     DDTranslation( 0,0, -backCoolBarThick()/2. - thickVFE/2. )   ,
	     myrot( backVFEName().name() + "Flip",
		    HepRotationX( 180*deg ) )        ) ;

      unsigned int iCVFECopy ( 1 ) ;
      unsigned int iSep ( 0 ) ;
      unsigned int iNSec ( 0 ) ;
      const unsigned int nMisc ( vecBackMiscThick().size()/vecBackCoolLength().size() ) ;
      for( unsigned int iMod ( 0 ) ; iMod != vecBackCoolLength().size() ; ++iMod )
      {
	 double backCoolHeight ( backCoolBarHeight() ) ;
	 for( unsigned int iMisc ( 0 ) ; iMisc != nMisc ; ++iMisc )
	 {
	    backCoolHeight += vecBackMiscThick()[ iMod*nMisc + iMisc ] ;
	 }

	 DDName backCName ( ddname( vecBackCoolName()[iMod] ) ) ;
	 const double halfZBCool ( vecBackCoolLength()[iMod]/2. ) ;
	 DDSolid backCoolSolid ( DDSolidFactory::box( backCName ,
						      backCoolHeight/2.,  
						      backCoolBarWidth()/2. + backCoolTankWidth(),
						      halfZBCool   ) ) ;
	 const DDLogicalPart backCoolLog ( backCName,
					   spmMat(),
					   backCoolSolid ) ;
	 
	 const DDTranslation bCoolTra ( -backPlateThick()/2 +
					backCoolHeight/2 -
					vecGrilleHeight()[2*iMod],
					0*mm,
					vecGrilleZOff()[2*iMod] +
					grilleThick() + grilleZSpace() +
					halfZBCool - 
					backSideLength()/2 ) ;
	 DDpos( backCoolLog,
		spmName(), 
		iMod+1, 
		outtra + backPlateTra + bCoolTra,
		DDRotation() ) ;

//===
	 const double backCoolTankHeight ( backCoolBarHeight() - backBracketHeight() ) ;

	 DDName bTankName ( ddname( backCoolTankName()+int_to_string(iMod+1) ) ) ;
	 DDSolid backCoolTankSolid ( DDSolidFactory::box( bTankName ,
							  backCoolTankHeight/2.,  
							  backCoolTankWidth()/2.,
							  halfZBCool   ) ) ;
	 const DDLogicalPart backCoolTankLog ( bTankName,
					       backCoolTankMat(),
					       backCoolTankSolid ) ;
	 DDpos( backCoolTankLog,
		backCName, 
		copyOne, 
		DDTranslation( -backCoolHeight/2 + backCoolTankHeight/2.,
			       backCoolBarWidth()/2. + backCoolTankWidth()/2., 0),
		DDRotation() ) ;

	 DDName bTankWaName ( ddname( backCoolTankWaName()+int_to_string(iMod+1) ) ) ;
	 DDSolid backCoolTankWaSolid ( DDSolidFactory::box( bTankWaName ,
							    backCoolTankHeight/2. -
							    backCoolTankThick()/2.,  
							    backCoolTankWaWidth()/2.,
							    halfZBCool -
							    backCoolTankThick()/2. ) ) ;
	 const DDLogicalPart backCoolTankWaLog ( bTankWaName,
						 backCoolTankWaMat(),
						 backCoolTankWaSolid ) ;
	 DDpos( backCoolTankWaLog,
		bTankName, 
		copyOne, 
		DDTranslation(0,0,0),
		DDRotation() ) ;

	 DDName bBracketName ( ddname( backBracketName()+int_to_string(iMod+1) ) ) ;
	 DDSolid backBracketSolid ( DDSolidFactory::box( bBracketName ,
							 backBracketHeight()/2.,  
							 backCoolTankWidth()/2.,
							 halfZBCool   ) ) ;
	 const DDLogicalPart backBracketLog ( bBracketName,
					      backBracketMat(),
					      backBracketSolid ) ;
	 DDpos( backBracketLog,
		backCName, 
		copyOne, 
		DDTranslation( backCoolBarHeight() - backCoolHeight/2. - backBracketHeight()/2.,
			       -backCoolBarWidth()/2. - backCoolTankWidth()/2., 0),
		DDRotation() ) ;

	 DDpos( backBracketLog,
		backCName, 
		copyTwo, 
		DDTranslation( backCoolBarHeight() - backCoolHeight/2. - backBracketHeight()/2.,
			       backCoolBarWidth()/2. + backCoolTankWidth()/2., 0),
		DDRotation() ) ;

//===

	 DDTranslation bSumTra ( backCoolBarHeight() - backCoolHeight/2., 0, 0 ) ;
	 for( unsigned int j ( 0 ) ; j != nMisc ; ++j )
	 {
	    const DDName bName ( ddname( vecBackMiscName()[ iMod*nMisc + j ] ) ) ;

	    DDSolid bSolid ( DDSolidFactory::box( bName ,
						  vecBackMiscThick()[ iMod*nMisc + j ]/2,  
						  backCoolBarWidth()/2. + backCoolTankWidth(),
						  halfZBCool ) ) ;

	    const DDLogicalPart bLog ( bName, ddmat(vecBackMiscMat()[ iMod*nMisc + j ]), bSolid ) ;
 
	    const DDTranslation bTra ( vecBackMiscThick()[ iMod*nMisc + j ]/2, 0*mm, 0*mm ) ;

	    DDpos( bLog,
		   backCName, 
		   copyOne, 
		   bSumTra + bTra,
		   DDRotation() ) ;

	    bSumTra += 2*bTra ;
	 }

	 DDName bPipeName ( ddname( backPipeName() + "_" + 
				    int_to_string( iMod+1 ) ) ) ; 
	 DDName bInnerName ( ddname( backPipeName() + 
				     "_H2O_" + int_to_string( iMod+1 ) ) ) ; 

	 const double pipeLength ( vecGrilleZOff()[2*iMod+1] -
				   vecGrilleZOff()[2*iMod  ] -
				   grilleThick()               ) ;

	 const double pipeZPos ( vecGrilleZOff()[2*iMod+1] - pipeLength/2  ) ;

	 DDSolid backPipeSolid ( DDSolidFactory::tubs( bPipeName ,
						       pipeLength/2,
						       0*mm, 
						       vecBackPipeDiam()[iMod]/2,
						       0*deg, 360*deg ) ) ;

	 DDSolid backInnerSolid ( DDSolidFactory::tubs( bInnerName ,
							pipeLength/2,
							0*mm, 
							vecBackPipeDiam()[iMod]/2 -
							backPipeThick(),
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

	 DDpos( backPipeLog,
		spmName(), 
		copyOne, 
		bPipeTra1,
		DDRotation() ) ;

	 if( 0 != iMod )
	 {
	    const DDTranslation bPipeTra2 ( bPipeTra1.x(),
					    backYOff() -
					    backPlateWidth()/2 +
					    backSideWidth() +
					    vecBackPipeDiam()[iMod],
					    bPipeTra1.z()  ) ;

	    DDpos( backPipeLog,
		   spmName(), 
		   copyTwo, 
		   bPipeTra2,
		   DDRotation() ) ;
	 }

	 DDpos( backInnerLog,
		bPipeName, 
		copyOne, 
		DDTranslation(),
		DDRotation() ) ;
	 //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	 DDTranslation cTra ( backCoolBarHeight()/2. - backCoolHeight/2., 0 ,
			      -halfZBCool + halfZCoolVFE ) ;
	 const unsigned int numSec ( static_cast<unsigned int> (vecBackCoolNSec()[iMod]) ) ; 
	 for( unsigned int jSec ( 0 ) ; jSec != numSec ; ++jSec )
	 {
	    const unsigned int nMax ( static_cast<unsigned int> (vecBackCoolNPerSec()[iNSec++]) ) ; 
	    for( unsigned int iBar ( 0 ) ; iBar !=  nMax ; ++iBar )
	    {
	       DDpos( backCoolVFELog,
		      backCName, 
		      iCVFECopy++, 
		      cTra,
		      DDRotation() ) ;	       
	       cTra += DDTranslation( 0, 0, backCBStdSep() ) ;
	    }
	    cTra -= DDTranslation( 0, 0, backCBStdSep() ) ; // backspace to previous
	    if( jSec != numSec-1 ) cTra += 
				      DDTranslation( 0, 0,
						     vecBackCoolSecSep()[iSep++] ) ; // now take atypical step
	 }
      }

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
	 
      const DDTranslation patchTra ( backXOff() ,
				     0*mm,
				     vecGrilleZOff().back() +
				     grilleThick() +
				     patchParms[2] ) ;

      DDpos( patchLog,
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
	 
	 pTra += Hep3Vector( vecPatchPanelThick()[j]/2, 0*mm, 0*mm ) ;
	 
	 DDpos( pLog,
		patchPanelName(), 
		copyOne, 
		pTra,
		DDRotation() ) ;
	 
	 pTra += Hep3Vector( vecPatchPanelThick()[j]/2, 0*mm, 0*mm ) ;
      }
   } 

   LogDebug("EcalGeom") << "******** DDEcalBarrelAlgo test: end it..." ;
}

DDRotation
DDEcalBarrelAlgo::myrot( const std::string&      s,
		     const DDRotationMatrix& r ) const 
{
   return DDrot( ddname( m_idNameSpace + ":" + s ), new DDRotationMatrix( r ) ) ; 
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
   return DDName( temp.first,
		  temp.second ) ; 
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
		       const HepPoint3D&   corner,
		       const DDLogicalPart logPar,
		       double&             zee,
		       double              side,
		       double              front,
		       double              delta  )
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
   const DDName        webClrDDName ( webClrName() + int_to_string( iWeb ) ) ;
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
   const DDName        webPlDDName  ( webPlName() + int_to_string( iWeb ) ) ;
   const DDSolid       webPlSolid   ( mytrap( webPlDDName, trapWebPl ) ) ;
   const DDLogicalPart webPlLog     ( webPlDDName, webPlMat(), webPlSolid ) ;

   DDpos( webPlLog,     // place plate inside clearance volume
	  webClrDDName, 
	  copyOne, 
	  Vec3(0,0,0),
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
   
   DDpos( webClrLog,
	  logPar, 
	  copyOne, 
	  tForm.getTranslation(),
	  myrot( webClrLog.name().name() + int_to_string( iWeb ),
		 tForm.getRotation() ) ) ;
}
