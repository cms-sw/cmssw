#ifndef DD_EcalBarrelNewAlgo_h
#define DD_EcalBarrelNewAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "Geometry/CaloGeometry/interface/EcalTrapezoidParameters.h"
#include "CLHEP/Geometry/Transform3D.h" 

class DDEcalBarrelNewAlgo : public DDAlgorithm {
 public:

      typedef EcalTrapezoidParameters Trap ;
      typedef HepGeom::Point3D<double>               Pt3D ;
      typedef HepGeom::Transform3D          Tf3D ;
      typedef HepGeom::ReflectZ3D           RfZ3D ;
      typedef HepGeom::Translate3D          Tl3D ;
      typedef HepGeom::Rotate3D             Ro3D ;
      typedef HepGeom::RotateZ3D            RoZ3D ;
      typedef HepGeom::RotateY3D            RoY3D ;
      typedef HepGeom::RotateX3D            RoX3D ;

      typedef CLHEP::Hep3Vector              Vec3 ;
      typedef CLHEP::HepRotation             Rota ;

      //Constructor and Destructor
      DDEcalBarrelNewAlgo();
      ~DDEcalBarrelNewAlgo() override;

      void initialize(const DDNumericArguments      & nArgs,
		      const DDVectorArguments       & vArgs,
		      const DDMapArguments          & mArgs,
		      const DDStringArguments       & sArgs,
		      const DDStringVectorArguments & vsArgs) override;
      void execute(DDCompactView& cpv) override;

      DDMaterial ddmat(  const std::string& s ) const ;
      DDName     ddname( const std::string& s ) const ;
      DDRotation myrot(  const std::string& s,
			 const CLHEP::HepRotation& r ) const ;
      DDSolid    mytrap( const std::string& s,
			 const Trap&        t ) const ;

      const std::string&         idNameSpace() const { return m_idNameSpace   ; }

      // barrel parent volume
      DDName                     barName()     const { return ddname( m_BarName ) ; }
      DDMaterial                 barMat()      const { return ddmat(  m_BarMat  ) ; }
      const std::vector<double>& vecBarZPts()  const { return m_vecBarZPts        ; }
      const std::vector<double>& vecBarRMin()  const { return m_vecBarRMin        ; }
      const std::vector<double>& vecBarRMax()  const { return m_vecBarRMax        ; }
      const std::vector<double>& vecBarTran()  const { return m_vecBarTran        ; }
      const std::vector<double>& vecBarRota()  const { return m_vecBarRota        ; }
      const std::vector<double>& vecBarRota2() const { return m_vecBarRota2       ; }
      const std::vector<double>& vecBarRota3() const { return m_vecBarRota3       ; }
      double                     barPhiLo()    const { return m_BarPhiLo          ; }
      double                     barPhiHi()    const { return m_BarPhiHi          ; }
      double                     barHere()     const { return m_BarHere           ; }

      DDName                     spmName()     const { return ddname( m_SpmName ) ; } 
      DDMaterial                 spmMat()      const { return ddmat(  m_SpmMat  ) ; }
      const std::vector<double>& vecSpmZPts()  const { return m_vecSpmZPts ; }
      const std::vector<double>& vecSpmRMin()  const { return m_vecSpmRMin ; }
      const std::vector<double>& vecSpmRMax()  const { return m_vecSpmRMax ; }
      const std::vector<double>& vecSpmTran()  const { return m_vecSpmTran ; }
      const std::vector<double>& vecSpmRota()  const { return m_vecSpmRota ; }
      const std::vector<double>& vecSpmBTran() const { return m_vecSpmBTran ; }
      const std::vector<double>& vecSpmBRota() const { return m_vecSpmBRota ; }
      unsigned int               spmNPerHalf() const { return m_SpmNPerHalf ; }
      double                     spmLowPhi()   const { return m_SpmLowPhi ; }
      double                     spmDelPhi()   const { return m_SpmDelPhi ; }
      double                     spmPhiOff()   const { return m_SpmPhiOff ; }
      const std::vector<double>& vecSpmHere()  const { return m_vecSpmHere ; }
      DDName                     spmCutName()  const { return ddname( m_SpmCutName ) ; }
      double                     spmCutThick() const { return m_SpmCutThick ; }
      int                        spmCutShow()  const { return m_SpmCutShow ; } 
      double                     spmCutRM()    const { return m_SpmCutRM ; }
      double                     spmCutRP()    const { return m_SpmCutRP ; }
      const std::vector<double>& vecSpmCutTM() const { return m_vecSpmCutTM ; }
      const std::vector<double>& vecSpmCutTP() const { return m_vecSpmCutTP ; }
      double                     spmExpThick() const { return m_SpmExpThick ; }
      double                     spmExpWide()  const { return m_SpmExpWide ; }
      double                     spmExpYOff()  const { return m_SpmExpYOff ; }
      DDName                     spmSideName() const { return ddname( m_SpmSideName ) ; } 
      DDMaterial                 spmSideMat()  const { return ddmat(  m_SpmSideMat  ) ; }
      double                     spmSideHigh() const { return m_SpmSideHigh ; }
      double                     spmSideThick() const { return m_SpmSideThick ; }
      double                     spmSideYOffM() const { return m_SpmSideYOffM ; }
      double                     spmSideYOffP() const { return m_SpmSideYOffP ; }

      double                     nomCryDimAF()    const { return m_NomCryDimAF    ; } 
      double                     nomCryDimLZ()    const { return m_NomCryDimLZ    ; }
      const std::vector<double>& vecNomCryDimBF() const { return m_vecNomCryDimBF ; }
      const std::vector<double>& vecNomCryDimCF() const { return m_vecNomCryDimCF ; }
      const std::vector<double>& vecNomCryDimAR() const { return m_vecNomCryDimAR ; }
      const std::vector<double>& vecNomCryDimBR() const { return m_vecNomCryDimBR ; }
      const std::vector<double>& vecNomCryDimCR() const { return m_vecNomCryDimCR ; }

      double                     underAF()     const { return m_UnderAF ; }
      double                     underLZ()     const { return m_UnderLZ ; }
      double                     underBF()     const { return m_UnderBF ; }
      double                     underCF()     const { return m_UnderCF ; }
      double                     underAR()     const { return m_UnderAR ; }
      double                     underBR()     const { return m_UnderBR ; }
      double                     underCR()     const { return m_UnderCR ; }

      double                     wallThAlv()   const { return m_WallThAlv ; }
      double                     wrapThAlv()   const { return m_WrapThAlv ; }
      double                     clrThAlv()    const { return m_ClrThAlv  ; }
      const std::vector<double>& vecGapAlvEta() const { return m_vecGapAlvEta ; }

      double                     wallFrAlv()   const { return m_WallFrAlv ; }
      double                     wrapFrAlv()   const { return m_WrapFrAlv ; }
      double                     clrFrAlv()    const { return m_ClrFrAlv ; }

      double                     wallReAlv()   const { return m_WallReAlv ; }
      double                     wrapReAlv()   const { return m_WrapReAlv ; }
      double                     clrReAlv()    const { return m_ClrReAlv ; }

      unsigned int               nCryTypes()     const { return m_NCryTypes ; }
      unsigned int               nCryPerAlvEta() const { return m_NCryPerAlvEta ; }

      const std::string&         cryName()  const { return m_CryName ; } 
      const std::string&         clrName()  const { return m_ClrName ; } 
      const std::string&         wrapName() const { return m_WrapName ; } 
      const std::string&         wallName() const { return m_WallName ; } 

      DDMaterial                 cryMat()   const { return ddmat( m_CryMat ) ; } 
      DDMaterial                 clrMat()   const { return ddmat( m_ClrMat ) ; } 
      DDMaterial                 wrapMat()  const { return ddmat( m_WrapMat ) ; } 
      DDMaterial                 wallMat()  const { return ddmat( m_WallMat ) ; } 
      DDName                   capName () const { return ddname(m_capName) ; }
      double                   capHere () const { return m_capHere ; }
      DDMaterial               capMat  () const { return ddmat(m_capMat)  ; }
      double                   capXSize() const { return m_capXSize ; }
      double                   capYSize() const { return m_capYSize ; }
      double                   capThick() const { return m_capThick; }

      DDName                   cerName () const { return ddname(m_CERName) ; }
      DDMaterial               cerMat  () const { return ddmat(m_CERMat)  ; }
      double                   cerXSize() const { return m_CERXSize ; }
      double                   cerYSize() const { return m_CERYSize ; }
      double                   cerThick() const { return m_CERThick; }

      DDName                   bsiName () const { return ddname(m_BSiName) ; }
      DDMaterial               bsiMat  () const { return ddmat(m_BSiMat)  ; }
      double                   bsiXSize() const { return m_BSiXSize ; }
      double                   bsiYSize() const { return m_BSiYSize ; }
      double                   bsiThick() const { return m_BSiThick; }

      DDName                   atjName () const { return ddname(m_ATJName) ; }
      DDMaterial               atjMat  () const { return ddmat(m_ATJMat)  ; }
      double                   atjThick() const { return m_ATJThick; }

      DDName                   sglName () const { return ddname(m_SGLName) ; }
      DDMaterial               sglMat  () const { return ddmat(m_SGLMat)  ; }
      double                   sglThick() const { return m_SGLThick; }

      DDName                   aglName () const { return ddname(m_AGLName) ; }
      DDMaterial               aglMat  () const { return ddmat(m_AGLMat)  ; }
      double                   aglThick() const { return m_AGLThick; }

      DDName                   andName () const { return ddname(m_ANDName) ; }
      DDMaterial               andMat  () const { return ddmat(m_ANDMat)  ; }
      double                   andThick() const { return m_ANDThick; }

      DDName                   apdName () const { return ddname(m_APDName) ; }
      DDMaterial               apdMat  () const { return ddmat(m_APDMat)  ; }
      double                   apdSide () const { return m_APDSide ; }
      double                   apdThick() const { return m_APDThick; }
      double                   apdZ    () const { return m_APDZ    ; }
      double                   apdX1   () const { return m_APDX1   ; }
      double                   apdX2   () const { return m_APDX2   ; }

      double                     webHere()      const { return m_WebHere     ; }
      const std::string&         webPlName()    const { return m_WebPlName          ; }
      const std::string&         webClrName()   const { return m_WebClrName         ; }
      DDMaterial                 webPlMat()     const { return ddmat( m_WebPlMat )  ; } 
      DDMaterial                 webClrMat()    const { return ddmat( m_WebClrMat ) ; } 
      const std::vector<double>& vecWebPlTh()   const { return m_vecWebPlTh         ; }
      const std::vector<double>& vecWebClrTh()  const { return m_vecWebClrTh        ; } 
      const std::vector<double>& vecWebLength() const { return m_vecWebLength       ; } 

      double                          ilyHere()     const { return m_IlyHere     ; }
      const std::string&              ilyName()     const { return m_IlyName     ; }
      double                          ilyPhiLow()   const { return m_IlyPhiLow   ; }
      double                          ilyDelPhi()   const { return m_IlyDelPhi   ; }
      const std::vector<std::string>& vecIlyMat()   const { return m_vecIlyMat   ; }
      const std::vector<double>&      vecIlyThick() const { return m_vecIlyThick ; }

      const std::string&         ilyPipeName     () const { return m_IlyPipeName         ;}
      double                     ilyPipeHere     () const { return m_IlyPipeHere          ;}
      DDMaterial                 ilyPipeMat      () const { return ddmat(m_IlyPipeMat)           ;}
      double                     ilyPipeOD       () const { return m_IlyPipeOD            ;}
      double                     ilyPipeID       () const { return m_IlyPipeID            ;}
      const std::vector<double>& vecIlyPipeLength() const { return m_vecIlyPipeLength     ;}
      const std::vector<double>& vecIlyPipeType  () const { return m_vecIlyPipeType       ;}
      const std::vector<double>& vecIlyPipePhi   () const { return m_vecIlyPipePhi        ;}
      const std::vector<double>& vecIlyPipeZ     () const { return m_vecIlyPipeZ        ;}

      DDName                     ilyPTMName   () const { return ddname(m_IlyPTMName)  ;}
      double                     ilyPTMHere   () const { return m_IlyPTMHere   ;}
      DDMaterial                 ilyPTMMat    () const { return ddmat(m_IlyPTMMat)    ;}
      double                     ilyPTMWidth  () const { return m_IlyPTMWidth  ;}
      double                     ilyPTMLength () const { return m_IlyPTMLength ;}
      double                     ilyPTMHeight () const { return m_IlyPTMHeight ;}
      const std::vector<double>& vecIlyPTMZ   () const { return m_vecIlyPTMZ   ;}
      const std::vector<double>& vecIlyPTMPhi () const { return m_vecIlyPTMPhi ;}

      DDName                   ilyFanOutName  () const { return ddname(m_IlyFanOutName)  ;}
      double                   ilyFanOutHere  () const { return m_IlyFanOutHere  ;}
      DDMaterial               ilyFanOutMat   () const { return ddmat(m_IlyFanOutMat)   ;}
      double                   ilyFanOutWidth () const { return m_IlyFanOutWidth ;}
      double                   ilyFanOutLength() const { return m_IlyFanOutLength;}
      double                   ilyFanOutHeight() const { return m_IlyFanOutHeight;}
      const std::vector<double>& vecIlyFanOutZ  () const { return m_vecIlyFanOutZ  ;}
      const std::vector<double>& vecIlyFanOutPhi() const { return m_vecIlyFanOutPhi;}
      DDName                   ilyDiffName    () const { return ddname(m_IlyDiffName)    ;}
      DDMaterial               ilyDiffMat     () const { return ddmat(m_IlyDiffMat)    ;}
      double                   ilyDiffOff     () const { return m_IlyDiffOff     ;}
      double                   ilyDiffLength  () const { return m_IlyDiffLength  ;}
      DDName                   ilyBndlName    () const { return ddname(m_IlyBndlName)    ;}
      DDMaterial               ilyBndlMat     () const { return ddmat(m_IlyBndlMat)    ;}
      double                   ilyBndlOff     () const { return m_IlyBndlOff     ;}
      double                   ilyBndlLength  () const { return m_IlyBndlLength  ;}
      DDName                   ilyFEMName     () const { return ddname(m_IlyFEMName)     ;}
      DDMaterial               ilyFEMMat      () const { return ddmat(m_IlyFEMMat)      ;}
      double                   ilyFEMWidth    () const { return m_IlyFEMWidth    ;}
      double                   ilyFEMLength   () const { return m_IlyFEMLength   ;}
      double                   ilyFEMHeight   () const { return m_IlyFEMHeight   ;}
      const std::vector<double>& vecIlyFEMZ     () const { return m_vecIlyFEMZ     ;}
      const std::vector<double>& vecIlyFEMPhi   () const { return m_vecIlyFEMPhi   ;}

      DDName              hawRName() const { return ddname( m_HawRName ) ; }
      DDName              fawName()  const { return ddname( m_FawName  ) ; }
      double              fawHere( ) const { return m_FawHere ; }
      double              hawRHBIG() const { return m_HawRHBIG ; }
      double              hawRhsml() const { return m_HawRhsml ; }
      double              hawRCutY() const { return m_HawRCutY ; }
      double              hawRCutZ() const { return m_HawRCutZ ; }
      double              hawRCutDelY() const { return m_HawRCutDelY ; }
      double              hawYOffCry() const { return m_HawYOffCry ; }

      unsigned int        nFawPerSupm() const { return m_NFawPerSupm ; }
      double              fawPhiOff() const { return m_FawPhiOff ; }
      double              fawDelPhi() const { return m_FawDelPhi ; }
      double              fawPhiRot() const { return m_FawPhiRot ; }
      double              fawRadOff() const { return m_FawRadOff ; }

      double              gridHere()  const { return m_GridHere     ; }
      DDName              gridName()  const { return ddname( m_GridName ) ; }
      DDMaterial          gridMat()   const { return ddmat(  m_GridMat  ) ; }
      double              gridThick() const { return m_GridThick ; }

      double                    backHere()        const { return m_BackHere     ; }
      double                    backXOff()        const { return m_BackXOff ; } 
      double                    backYOff()        const { return m_BackYOff ; }
      DDName                    backSideName()    const { return ddname( m_BackSideName ) ; }
      double                    backSideHere()    const { return m_BackSideHere         ; } 
      double                    backSideLength()  const { return m_BackSideLength         ; } 
      double                    backSideHeight()  const { return m_BackSideHeight         ; }
      double                    backSideWidth()   const { return m_BackSideWidth          ; }
      double                    backSideYOff1()   const { return m_BackSideYOff1         ; }
      double                    backSideYOff2()   const { return m_BackSideYOff2          ; }
      double                    backSideAngle()   const { return m_BackSideAngle          ; }
      DDMaterial                backSideMat()     const { return ddmat( m_BackSideMat )   ; }
      DDName                    backPlateName()   const { return ddname( m_BackPlateName ) ; }
      double                    backPlateHere()   const { return m_BackPlateHere         ; } 
      double                    backPlateLength() const { return m_BackPlateLength   ; }
      double                    backPlateThick()  const { return m_BackPlateThick     ; }
      double                    backPlateWidth()  const { return m_BackPlateWidth     ; }
      DDMaterial                backPlateMat()    const { return ddmat( m_BackPlateMat ) ; }
      DDName                    backPlate2Name()   const { return ddname( m_BackPlate2Name ) ; }
      double                    backPlate2Thick()  const { return m_BackPlate2Thick     ; }
      DDMaterial                backPlate2Mat()    const { return ddmat( m_BackPlate2Mat ) ; }
      const std::string&        grilleName()      const { return m_GrilleName ; }
      double                    grilleThick()     const { return m_GrilleThick         ; }
      double                    grilleHere()      const { return m_GrilleHere          ; }
      double                    grilleWidth()     const { return m_GrilleWidth         ; }
      double                    grilleZSpace()    const { return m_GrilleZSpace         ; }
      DDMaterial                grilleMat()       const { return ddmat( m_GrilleMat )    ; }
      const std::vector<double>& vecGrilleHeight() const { return m_vecGrilleHeight      ; }
      const std::vector<double>& vecGrilleZOff()  const { return m_vecGrilleZOff        ; }

      DDName                   grEdgeSlotName    () const { return ddname(m_GrEdgeSlotName) ; }
      DDMaterial               grEdgeSlotMat     () const { return ddmat(m_GrEdgeSlotMat) ; }
      double                   grEdgeSlotHere    () const { return m_GrEdgeSlotHere    ; }
      double                   grEdgeSlotHeight  () const { return m_GrEdgeSlotHeight  ; }
      double                   grEdgeSlotWidth   () const { return m_GrEdgeSlotWidth   ; }
      const std::string&       grMidSlotName     () const { return m_GrMidSlotName     ; }
      DDMaterial               grMidSlotMat      () const { return ddmat(m_GrMidSlotMat) ; }
      double                   grMidSlotHere     () const { return m_GrMidSlotHere     ; }
      double                   grMidSlotWidth    () const { return m_GrMidSlotWidth    ; }
      double                   grMidSlotXOff     () const { return m_GrMidSlotXOff     ; }
      const std::vector<double>& vecGrMidSlotHeight() const { return m_vecGrMidSlotHeight; }

      double                    backPipeHere()    const { return m_BackPipeHere     ; }
      const std::string&        backPipeName()   const { return m_BackPipeName ; }
      const std::vector<double>& vecBackPipeDiam() const { return m_vecBackPipeDiam    ; }
      const std::vector<double>& vecBackPipeThick() const { return m_vecBackPipeThick    ; }
      DDMaterial                backPipeMat()    const { return ddmat( m_BackPipeMat ) ; }
      DDMaterial                backPipeWaterMat() const { return ddmat( m_BackPipeWaterMat ) ; }
      double                   backMiscHere()       const { return m_BackMiscHere     ; }
      const std::vector<double>& vecBackMiscThick() const { return m_vecBackMiscThick ; }
      const std::vector<std::string>& vecBackMiscName() const
                                  { return m_vecBackMiscName ; }
      const std::vector<std::string>& vecBackMiscMat() const
	                          { return m_vecBackMiscMat ; }
      double                     patchPanelHere()     const { return m_PatchPanelHere     ; }
      const std::vector<double>& vecPatchPanelThick() const { return m_vecPatchPanelThick ; }
      const std::vector<std::string>& vecPatchPanelNames() const
                                  { return m_vecPatchPanelNames ; }
      const std::vector<std::string>& vecPatchPanelMat() const
	                          { return m_vecPatchPanelMat ; }
      DDName                    patchPanelName()   const { return ddname( m_PatchPanelName ) ; }

      const std::vector<std::string>& vecBackCoolName() const { return m_vecBackCoolName       ;}
      double                   backCoolHere()       const { return m_BackCoolHere     ; }
      double                   backCoolBarWidth  () const { return m_BackCoolBarWidth      ;}
      double                   backCoolBarHeight () const { return m_BackCoolBarHeight     ;}
      DDMaterial               backCoolMat       () const { return ddmat(m_BackCoolMat)    ;}
      double                   backCoolBarHere()    const { return m_BackCoolBarHere     ; }
      DDName                   backCoolBarName   () const { return ddname(m_BackCoolBarName);}
      double                   backCoolBarThick  () const { return m_BackCoolBarThick      ;}
      DDMaterial               backCoolBarMat    () const { return ddmat(m_BackCoolBarMat) ;}
      DDName                   backCoolBarSSName () const { return ddname(m_BackCoolBarSSName);}
      double                   backCoolBarSSThick() const { return m_BackCoolBarSSThick    ;}
      DDMaterial               backCoolBarSSMat  () const { return ddmat(m_BackCoolBarSSMat) ;}
      DDName                   backCoolBarWaName () const { return ddname(m_BackCoolBarWaName);}
      double                   backCoolBarWaThick() const { return m_BackCoolBarWaThick    ;}
      DDMaterial               backCoolBarWaMat  () const { return ddmat(m_BackCoolBarWaMat) ;}
      double                   backCoolVFEHere()    const { return m_BackCoolVFEHere     ; }
      DDName                   backCoolVFEName   () const { return ddname(m_BackCoolVFEName) ;}
      DDMaterial               backCoolVFEMat    () const { return ddmat(m_BackCoolVFEMat) ;}
      DDName                   backVFEName       () const { return ddname(m_BackVFEName)   ;}
      DDMaterial               backVFEMat        () const { return ddmat(m_BackVFEMat) ;}
      const std::vector<double>& vecBackVFELyrThick() const { return m_vecBackVFELyrThick    ;}
      const std::vector<std::string>& vecBackVFELyrName () const { return m_vecBackVFELyrName     ;}
      const std::vector<std::string>& vecBackVFELyrMat  () const { return m_vecBackVFELyrMat      ;}
      const std::vector<double>& vecBackCoolNSec   () const { return m_vecBackCoolNSec       ;}
      const std::vector<double>& vecBackCoolSecSep () const { return m_vecBackCoolSecSep     ;}
      const std::vector<double>& vecBackCoolNPerSec() const { return m_vecBackCoolNPerSec    ;}
      double                   backCBStdSep      () const { return m_BackCBStdSep          ;}

      double                   backCoolTankHere()    const { return m_BackCoolTankHere     ; }
      const std::string&       backCoolTankName   () const { return m_BackCoolTankName ;}
      double                   backCoolTankWidth  () const { return m_BackCoolTankWidth  ;}
      double                   backCoolTankThick  () const { return m_BackCoolTankThick  ;}
      DDMaterial               backCoolTankMat    () const { return ddmat(m_BackCoolTankMat) ;}
      const std::string&       backCoolTankWaName () const { return m_BackCoolTankWaName ;}
      double                   backCoolTankWaWidth() const { return m_BackCoolTankWaWidth;}
      DDMaterial               backCoolTankWaMat  () const { return ddmat(m_BackCoolTankWaMat) ;}
      const std::string&       backBracketName    () const { return m_BackBracketName  ;}
      double                   backBracketHeight  () const { return m_BackBracketHeight  ;}
      DDMaterial               backBracketMat     () const { return ddmat(m_BackBracketMat)    ;}
      
      double                   dryAirTubeHere()     const { return m_DryAirTubeHere     ; }
      const std::string&       dryAirTubeName    () const { return m_DryAirTubeName   ;}
      double                   mBCoolTubeNum     () const { return m_MBCoolTubeNum   ;}
      double                   dryAirTubeInnDiam () const { return m_DryAirTubeInnDiam   ;}
      double                   dryAirTubeOutDiam () const { return m_DryAirTubeOutDiam   ;}
      DDMaterial               dryAirTubeMat     () const { return ddmat(m_DryAirTubeMat)      ;}
      double                   mBCoolTubeHere()     const { return m_MBCoolTubeHere     ; }
      const std::string&       mBCoolTubeName    () const { return m_MBCoolTubeName   ;}
      double                   mBCoolTubeInnDiam () const { return m_MBCoolTubeInnDiam   ;}
      double                   mBCoolTubeOutDiam () const { return m_MBCoolTubeOutDiam   ;}
      DDMaterial               mBCoolTubeMat     () const { return ddmat(m_MBCoolTubeMat)      ;}
      double                   mBManifHere()        const { return m_MBManifHere     ; }
      DDName                   mBManifName       () const { return ddname(m_MBManifName)        ;}
      double                   mBManifInnDiam    () const { return m_MBManifInnDiam      ;}
      double                   mBManifOutDiam    () const { return m_MBManifOutDiam      ;}
      DDMaterial               mBManifMat        () const { return ddmat(m_MBManifMat)         ;}
      double                   mBLyrHere()          const { return m_MBLyrHere     ; }
      const std::vector<double>&      vecMBLyrThick() const { return m_vecMBLyrThick       ;}
      const std::vector<std::string>& vecMBLyrName () const { return m_vecMBLyrName        ;}
      const std::vector<std::string>& vecMBLyrMat  () const { return m_vecMBLyrMat         ;}

//----------

      double                   pincerRodHere      () const { return m_PincerRodHere      ;}
      DDName                   pincerRodName      () const { return ddname(m_PincerRodName)      ;}
      DDMaterial               pincerRodMat       () const { return ddmat(m_PincerRodMat)       ;}
      std::vector<double>      vecPincerRodAzimuth() const { return m_vecPincerRodAzimuth;}
      DDName                   pincerEnvName      () const { return ddname(m_PincerEnvName)      ;}
      DDMaterial               pincerEnvMat       () const { return ddmat(m_PincerEnvMat)       ;}
      double                   pincerEnvWidth     () const { return m_PincerEnvWidth     ;}
      double                   pincerEnvHeight    () const { return m_PincerEnvHeight    ;}
      double                   pincerEnvLength    () const { return m_PincerEnvLength    ;}
      std::vector<double>      vecPincerEnvZOff   () const { return m_vecPincerEnvZOff   ;}
       		     						     			     
      DDName                   pincerBlkName      () const { return ddname(m_PincerBlkName)      ;}
      DDMaterial               pincerBlkMat       () const { return ddmat(m_PincerBlkMat)       ;}
      double                   pincerBlkLength    () const { return m_PincerBlkLength    ;}
			       		     						   
      DDName                   pincerShim1Name    () const { return ddname(m_PincerShim1Name)    ;}
      double                   pincerShimHeight   () const { return m_PincerShimHeight   ;}
      DDName                   pincerShim2Name    () const { return ddname(m_PincerShim2Name)    ;}
      DDMaterial               pincerShimMat      () const { return ddmat(m_PincerShimMat)      ;}
      double                   pincerShim1Width   () const { return m_PincerShim1Width   ;}
      double                   pincerShim2Width   () const { return m_PincerShim2Width   ;}
			       		     						 
      DDName                   pincerCutName      () const { return ddname(m_PincerCutName)      ;}
      DDMaterial               pincerCutMat       () const { return ddmat(m_PincerCutMat)      ;}
      double                   pincerCutWidth     () const { return m_PincerCutWidth    ;}
      double                   pincerCutHeight    () const { return m_PincerCutHeight    ;}

protected:

private:

      void web( unsigned int        iWeb,
		double              bWeb,
		double              BWeb,
		double              LWeb,
		double              theta,
		const Pt3D&         corner,
		const DDLogicalPart& logPar,
		double&             zee  ,
		double              side,
		double              front,
		double              delta,
		DDCompactView&      cpv );

      std::string         m_idNameSpace;            //Namespace of this and ALL sub-parts

      // Barrel volume
      std::string         m_BarName    ; // Barrel volume name
      std::string         m_BarMat     ; // Barrel material name
      std::vector<double> m_vecBarZPts ; // Barrel list of z pts
      std::vector<double> m_vecBarRMin ; // Barrel list of rMin pts
      std::vector<double> m_vecBarRMax ; // Barrel list of rMax pts
      std::vector<double> m_vecBarTran ; // Barrel translation
      std::vector<double> m_vecBarRota ; // Barrel rotation
      std::vector<double> m_vecBarRota2; // 2nd Barrel rotation
      std::vector<double> m_vecBarRota3; // 2nd Barrel rotation
      double              m_BarPhiLo   ; // Barrel phi lo
      double              m_BarPhiHi   ; // Barrel phi hi
      double              m_BarHere    ; // Barrel presence flag
      
      // Supermodule volume
      std::string         m_SpmName     ; // Supermodule volume name
      std::string         m_SpmMat      ; // Supermodule material name
      std::vector<double> m_vecSpmZPts  ; // Supermodule list of z pts
      std::vector<double> m_vecSpmRMin  ; // Supermodule list of rMin pts
      std::vector<double> m_vecSpmRMax  ; // Supermodule list of rMax pts
      std::vector<double> m_vecSpmTran  ; // Supermodule translation
      std::vector<double> m_vecSpmRota  ; // Supermodule rotation
      std::vector<double> m_vecSpmBTran ; // Base Supermodule translation
      std::vector<double> m_vecSpmBRota ; // Base Supermodule rotation
      unsigned int        m_SpmNPerHalf ; // # Supermodules per half detector
      double              m_SpmLowPhi   ; // Low   phi value of base supermodule
      double              m_SpmDelPhi   ; // Delta phi value of base supermodule
      double              m_SpmPhiOff   ; // Phi offset value supermodule
      std::vector<double> m_vecSpmHere  ; // Bit saying if a supermodule is present or not
      std::string         m_SpmCutName  ; // Name of cut box
      double              m_SpmCutThick ; // Box thickness
      int                 m_SpmCutShow  ; // Non-zero means show the box on display (testing only)
      std::vector<double> m_vecSpmCutTM ; // Translation for minus phi cut box
      std::vector<double> m_vecSpmCutTP ; // Translation for plus  phi cut box      
      double              m_SpmCutRM    ; // Rotation for minus phi cut box
      double              m_SpmCutRP    ; // Rotation for plus  phi cut box      
      double              m_SpmExpThick ; // Thickness (x) of supermodule expansion box
      double              m_SpmExpWide  ; // Width     (y) of supermodule expansion box
      double              m_SpmExpYOff  ; // Offset    (y) of supermodule expansion box
      std::string         m_SpmSideName ; // Supermodule Side Plate volume name
      std::string         m_SpmSideMat  ; // Supermodule Side Plate material name
      double              m_SpmSideHigh ; // Side plate height
      double              m_SpmSideThick; // Side plate thickness
      double              m_SpmSideYOffM; // Side plate Y offset on minus phi side
      double              m_SpmSideYOffP; // Side plate Y offset on plus  phi side

      double              m_NomCryDimAF    ; // Nominal crystal AF
      double              m_NomCryDimLZ    ; // Nominal crystal LZ
      std::vector<double> m_vecNomCryDimBF ; // Nominal crystal BF
      std::vector<double> m_vecNomCryDimCF ; // Nominal crystal CF
      std::vector<double> m_vecNomCryDimAR ; // Nominal crystal AR
      std::vector<double> m_vecNomCryDimBR ; // Nominal crystal BR
      std::vector<double> m_vecNomCryDimCR ; // Nominal crystal CR

      double              m_UnderAF ; // undershoot of AF
      double              m_UnderLZ ; // undershoot of LZ
      double              m_UnderBF ; // undershoot of BF
      double              m_UnderCF ; // undershoot of CF
      double              m_UnderAR ; // undershoot of AR
      double              m_UnderBR ; // undershoot of BR
      double              m_UnderCR ; // undershoot of CR

      double              m_WallThAlv ; // alveoli wall thickness
      double              m_WrapThAlv ; // wrapping thickness
      double              m_ClrThAlv  ; // clearance thickness (nominal)
      std::vector<double> m_vecGapAlvEta ; // Extra clearance after each alveoli perp to crystal axis

      double              m_WallFrAlv ; // alveoli wall frontage
      double              m_WrapFrAlv ; // wrapping frontage
      double              m_ClrFrAlv  ; // clearance frontage (nominal)

      double              m_WallReAlv ; // alveoli wall rearage
      double              m_WrapReAlv ; // wrapping rearage
      double              m_ClrReAlv  ; // clearance rearage (nominal)

      unsigned int        m_NCryTypes     ; // number of crystal shapes
      unsigned int        m_NCryPerAlvEta ; // number of crystals in eta per alveolus

      std::string         m_CryName  ; // string name of crystal volume
      std::string         m_ClrName  ; // string name of clearance volume
      std::string         m_WrapName ; // string name of wrap volume
      std::string         m_WallName ; // string name of wall volume

      std::string         m_CryMat  ; // string name of crystal material
      std::string         m_ClrMat  ; // string name of clearance material
      std::string         m_WrapMat ; // string name of wrap material
      std::string         m_WallMat ; // string name of wall material

      std::string              m_capName      ; // Capsule
      double                   m_capHere      ; // 
      std::string              m_capMat       ; // 
      double                   m_capXSize     ; // 
      double                   m_capYSize     ; // 
      double                   m_capThick     ; // 

      std::string              m_CERName      ; // Ceramic
      std::string              m_CERMat       ; // 
      double                   m_CERXSize     ; // 
      double                   m_CERYSize     ; // 
      double                   m_CERThick     ; // 

      std::string              m_BSiName      ; // Bulk Silicon
      std::string              m_BSiMat       ; // 
      double                   m_BSiXSize     ; // 
      double                   m_BSiYSize     ; // 
      double                   m_BSiThick     ; // 

      std::string              m_APDName      ; // APD
      std::string              m_APDMat       ; // 
      double                   m_APDSide      ; // 
      double                   m_APDThick     ; // 
      double                   m_APDZ         ; // 
      double                   m_APDX1        ; // 
      double                   m_APDX2        ; // 

      std::string              m_ATJName      ; // After-The-Junction
      std::string              m_ATJMat       ; // 
      double                   m_ATJThick     ; // 

      std::string              m_SGLName      ; // APD-Silicone glue
      std::string              m_SGLMat       ; // 
      double                   m_SGLThick     ; // 

      std::string              m_AGLName      ; // APD-Glue
      std::string              m_AGLMat       ; // 
      double                   m_AGLThick     ; // 

      std::string              m_ANDName      ; // APD-Non-Depleted
      std::string              m_ANDMat       ; // 
      double                   m_ANDThick     ; // 

      double              m_WebHere      ; // here flag
      std::string         m_WebPlName    ; // string name of web plate volume
      std::string         m_WebClrName   ; // string name of web clearance volume
      std::string         m_WebPlMat     ; // string name of web material
      std::string         m_WebClrMat    ; // string name of web clearance material
      std::vector<double> m_vecWebPlTh   ; // Thickness of web plates
      std::vector<double> m_vecWebClrTh  ; // Thickness of total web clearance
      std::vector<double> m_vecWebLength ; // Length of web plate

      double                   m_IlyHere      ; // here flag
      std::string              m_IlyName      ; // string name of inner layer volume
      double                   m_IlyPhiLow    ; // low phi of volumes
      double                   m_IlyDelPhi    ; // delta phi of ily
      std::vector<std::string> m_vecIlyMat    ; // materials of inner layer volumes
      std::vector<double>      m_vecIlyThick  ; // Thicknesses of inner layer volumes

      std::string              m_IlyPipeName          ; // Cooling pipes
      double                   m_IlyPipeHere          ; //
      std::string              m_IlyPipeMat           ; //
      double                   m_IlyPipeOD            ; //
      double                   m_IlyPipeID            ; //
      std::vector<double>      m_vecIlyPipeLength     ; //
      std::vector<double>      m_vecIlyPipeType       ; //
      std::vector<double>      m_vecIlyPipePhi        ; //
      std::vector<double>      m_vecIlyPipeZ          ; //

      std::string              m_IlyPTMName          ; // PTM
      double                   m_IlyPTMHere          ; //
      std::string              m_IlyPTMMat           ; //
      double                   m_IlyPTMWidth         ; //
      double                   m_IlyPTMLength        ; //
      double                   m_IlyPTMHeight        ; //
      std::vector<double>      m_vecIlyPTMZ          ; //
      std::vector<double>      m_vecIlyPTMPhi        ; //     

      std::string              m_IlyFanOutName          ; // FanOut
      double                   m_IlyFanOutHere          ; //
      std::string              m_IlyFanOutMat           ; //
      double                   m_IlyFanOutWidth         ; //
      double                   m_IlyFanOutLength        ; //
      double                   m_IlyFanOutHeight        ; //
      std::vector<double>      m_vecIlyFanOutZ          ; //
      std::vector<double>      m_vecIlyFanOutPhi        ; //     
      std::string              m_IlyDiffName            ; // Diffuser
      std::string              m_IlyDiffMat             ; //
      double                   m_IlyDiffOff             ; //
      double                   m_IlyDiffLength          ; //
      std::string              m_IlyBndlName            ; // Fiber bundle
      std::string              m_IlyBndlMat             ; //
      double                   m_IlyBndlOff             ; //
      double                   m_IlyBndlLength          ; //
      std::string              m_IlyFEMName             ; // FEM
      std::string              m_IlyFEMMat              ; //
      double                   m_IlyFEMWidth            ; //
      double                   m_IlyFEMLength           ; //
      double                   m_IlyFEMHeight           ; //
      std::vector<double>      m_vecIlyFEMZ             ; //
      std::vector<double>      m_vecIlyFEMPhi           ; //              


      std::string         m_HawRName ; // string name of half-alveolar wedge
      std::string         m_FawName  ; // string name of full-alveolar wedge
      double              m_FawHere  ; // here flag 
      double              m_HawRHBIG ; // height at big end of half alveolar wedge
      double              m_HawRhsml ; // height at small end of half alveolar wedge
      double              m_HawRCutY ; // x dim of hawR cut box
      double              m_HawRCutZ ; // y dim of hawR cut box
      double              m_HawRCutDelY ; // y offset of hawR cut box from top of HAW
      double              m_HawYOffCry  ; // Y offset of crystal wrt HAW at front
      unsigned int        m_NFawPerSupm ; // Number of Full Alv. Wedges per supermodule
      double              m_FawPhiOff ; // Phi offset for FAW placement
      double              m_FawDelPhi ; // Phi delta for FAW placement
      double              m_FawPhiRot ; // Phi rotation of FAW about own axis prior to placement
      double              m_FawRadOff ; // Radial offset for FAW placement

      double              m_GridHere  ; // here flag
      std::string         m_GridName  ; // Grid name
      std::string         m_GridMat   ; // Grid material
      double              m_GridThick ; // Grid Thickness

      double                   m_BackXOff        ; //
      double                   m_BackYOff        ; //

      double                   m_BackHere              ; // here flag
      std::string              m_BackSideName          ; // Back Sides
      double                   m_BackSideHere          ; //
      double                   m_BackSideLength        ; //
      double                   m_BackSideHeight        ; //
      double                   m_BackSideWidth         ; //
      double                   m_BackSideYOff1         ; //
      double                   m_BackSideYOff2         ; //
      double                   m_BackSideAngle         ; //
      std::string              m_BackSideMat           ; //
      std::string              m_BackPlateName    ; // back plate
      double                   m_BackPlateHere          ; //
      double                   m_BackPlateLength  ; //
      double                   m_BackPlateThick   ; //
      double                   m_BackPlateWidth   ; //
      std::string              m_BackPlateMat     ; //
      std::string              m_BackPlate2Name    ; // back plate2
      double                   m_BackPlate2Thick   ; //
      std::string              m_BackPlate2Mat     ; //
      std::string              m_GrilleName      ; // grille
      double                   m_GrilleHere      ; //
      double                   m_GrilleThick     ; //
      double                   m_GrilleWidth     ; //
      double                   m_GrilleZSpace    ; //
      std::string              m_GrilleMat       ; //
      std::vector<double>      m_vecGrilleHeight ; //
      std::vector<double>      m_vecGrilleZOff   ; //

      std::string              m_GrEdgeSlotName          ; // Slots in Grille
      std::string              m_GrEdgeSlotMat           ; //
      double                   m_GrEdgeSlotHere          ; //
      double                   m_GrEdgeSlotHeight        ; //
      double                   m_GrEdgeSlotWidth         ; //

      std::string              m_GrMidSlotName           ; // Slots in Grille
      std::string              m_GrMidSlotMat            ; //
      double                   m_GrMidSlotHere           ; //
      double                   m_GrMidSlotWidth          ; //
      double                   m_GrMidSlotXOff           ; //
      std::vector<double>      m_vecGrMidSlotHeight      ; //                  

      double                   m_BackPipeHere    ; // here flag
      std::string              m_BackPipeName    ; // 
      std::vector<double>      m_vecBackPipeDiam ; // pipes
      std::vector<double>      m_vecBackPipeThick ; // pipes
      std::string              m_BackPipeMat     ; //
      std::string              m_BackPipeWaterMat ; //

      std::vector<std::string> m_vecBackCoolName       ; // cooling circuits
      double                   m_BackCoolHere          ; // here flag
      double                   m_BackCoolBarHere       ; // here flag
      double                   m_BackCoolBarWidth      ; //
      double                   m_BackCoolBarHeight     ; //
      std::string              m_BackCoolMat           ;
      std::string              m_BackCoolBarName       ; // cooling bar
      double                   m_BackCoolBarThick      ; //
      std::string              m_BackCoolBarMat        ;
      std::string              m_BackCoolBarSSName     ; // cooling bar tubing
      double                   m_BackCoolBarSSThick    ; //
      std::string              m_BackCoolBarSSMat      ;
      std::string              m_BackCoolBarWaName     ; // cooling bar water
      double                   m_BackCoolBarWaThick    ; //
      std::string              m_BackCoolBarWaMat      ;
      double                   m_BackCoolVFEHere       ; // here flag
      std::string              m_BackCoolVFEName       ;
      std::string              m_BackCoolVFEMat        ;
      std::string              m_BackVFEName           ;
      std::string              m_BackVFEMat            ;
      std::vector<double>      m_vecBackVFELyrThick    ; //
      std::vector<std::string> m_vecBackVFELyrName     ; //
      std::vector<std::string> m_vecBackVFELyrMat      ; //
      std::vector<double>      m_vecBackCoolNSec       ; //
      std::vector<double>      m_vecBackCoolSecSep     ; //
      std::vector<double>      m_vecBackCoolNPerSec    ; //
      double                   m_BackMiscHere          ; // here flag
      std::vector<double>      m_vecBackMiscThick      ; // misc materials
      std::vector<std::string> m_vecBackMiscName       ; //
      std::vector<std::string> m_vecBackMiscMat        ; //
      double                   m_BackCBStdSep          ; //

      double                   m_PatchPanelHere        ; // here flag
      std::string              m_PatchPanelName        ; //
      std::vector<double>      m_vecPatchPanelThick    ; // patch panel materials
      std::vector<std::string> m_vecPatchPanelNames    ; //
      std::vector<std::string> m_vecPatchPanelMat      ; //

      double                   m_BackCoolTankHere    ; // here flag
      std::string              m_BackCoolTankName    ; // service tank
      double                   m_BackCoolTankWidth   ; //
      double                   m_BackCoolTankThick   ; //
      std::string              m_BackCoolTankMat     ; //
      std::string              m_BackCoolTankWaName  ; //
      double                   m_BackCoolTankWaWidth ; //
      std::string              m_BackCoolTankWaMat   ; //
      std::string              m_BackBracketName     ; //
      double                   m_BackBracketHeight   ; //
      std::string              m_BackBracketMat      ; //

      double                   m_DryAirTubeHere      ; // here flag
      std::string              m_DryAirTubeName      ; // dry air tube
      unsigned int             m_MBCoolTubeNum      ; //
      double                   m_DryAirTubeInnDiam   ; //
      double                   m_DryAirTubeOutDiam   ; //
      std::string              m_DryAirTubeMat       ; //
      double                   m_MBCoolTubeHere      ; // here flag
      std::string              m_MBCoolTubeName      ; // mothr bd cooling tube
      double                   m_MBCoolTubeInnDiam   ; //
      double                   m_MBCoolTubeOutDiam   ; //
      std::string              m_MBCoolTubeMat       ; //
      double                   m_MBManifHere         ; // here flag
      std::string              m_MBManifName         ; //mother bd manif
      double                   m_MBManifInnDiam      ; //
      double                   m_MBManifOutDiam      ; //
      std::string              m_MBManifMat          ; //
      double                   m_MBLyrHere           ; // here flag
      std::vector<double>      m_vecMBLyrThick       ; // mother bd lyrs
      std::vector<std::string> m_vecMBLyrName        ; //
      std::vector<std::string> m_vecMBLyrMat         ; //

//-------------------------------------------------------------------

      double                   m_PincerRodHere      ; // here flag
      std::string              m_PincerRodName      ; // pincer rod
      std::string              m_PincerRodMat       ; // 
      std::vector<double>      m_vecPincerRodAzimuth; //

      std::string              m_PincerEnvName      ; // pincer envelope
      std::string              m_PincerEnvMat       ; // 
      double                   m_PincerEnvWidth     ; //
      double                   m_PincerEnvHeight    ; //
      double                   m_PincerEnvLength    ; //
      std::vector<double>      m_vecPincerEnvZOff   ; //

      std::string              m_PincerBlkName      ; // pincer block
      std::string              m_PincerBlkMat       ; // 
      double                   m_PincerBlkLength    ; //

      std::string              m_PincerShim1Name    ; // pincer shim
      double                   m_PincerShimHeight   ; //
      std::string              m_PincerShim2Name    ; // 
      std::string              m_PincerShimMat      ; // 
      double                   m_PincerShim1Width   ; //
      double                   m_PincerShim2Width   ; //

      std::string              m_PincerCutName      ; // pincer block
      std::string              m_PincerCutMat       ; // 
      double                   m_PincerCutWidth     ; //
      double                   m_PincerCutHeight    ; //

}; 

#endif
