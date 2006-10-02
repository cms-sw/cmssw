#ifndef DD_EcalBarrelAlgo_h
#define DD_EcalBarrelAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "Geometry/CaloGeometry/interface/EcalTrapezoidParameters.h"

class DDEcalBarrelAlgo : public DDAlgorithm {
 public:

      typedef EcalTrapezoidParameters Trap ;
      typedef HepPoint3D              Pt3D ;
      typedef HepTransform3D          Tf3D ;
      typedef HepReflectZ3D           RfZ3D ;
      typedef HepTranslate3D          Tl3D ;
      typedef HepRotate3D             Ro3D ;
      typedef HepRotateZ3D            RoZ3D ;
      typedef HepRotateY3D            RoY3D ;
      typedef HepRotateX3D            RoX3D ;

      typedef Hep3Vector              Vec3 ;
      typedef HepRotation             Rota ;

      //Constructor and Destructor
      DDEcalBarrelAlgo();
      virtual ~DDEcalBarrelAlgo();

      void initialize(const DDNumericArguments      & nArgs,
		      const DDVectorArguments       & vArgs,
		      const DDMapArguments          & mArgs,
		      const DDStringArguments       & sArgs,
		      const DDStringVectorArguments & vsArgs);
      void execute();

      DDMaterial ddmat(  const std::string& s ) const ;
      DDName     ddname( const std::string& s ) const ;
      DDRotation myrot(  const std::string& s,
			 const DDRotationMatrix& r ) const ;
      DDSolid    mytrap( const std::string& s,
			 const Trap&        t ) const ;

      const std::string          idNameSpace() const { return m_idNameSpace   ; }

      // barrel parent volume
      DDName                     barName()     const { return ddname( m_BarName ) ; }
      DDMaterial                 barMat()      const { return ddmat(  m_BarMat  ) ; }
      const std::vector<double>& vecBarZPts()  const { return m_vecBarZPts        ; }
      const std::vector<double>& vecBarRMin()  const { return m_vecBarRMin        ; }
      const std::vector<double>& vecBarRMax()  const { return m_vecBarRMax        ; }
      const std::vector<double>& vecBarTran()  const { return m_vecBarTran        ; }
      const std::vector<double>& vecBarRota()  const { return m_vecBarRota        ; }
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
      double                     spmCutShow()  const { return m_SpmCutShow ; } 
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

      const std::string&         webPlName()    const { return m_WebPlName          ; }
      const std::string&         webClrName()   const { return m_WebClrName         ; }
      DDMaterial                 webPlMat()     const { return ddmat( m_WebPlMat )  ; } 
      DDMaterial                 webClrMat()    const { return ddmat( m_WebClrMat ) ; } 
      const std::vector<double>& vecWebPlTh()   const { return m_vecWebPlTh         ; }
      const std::vector<double>& vecWebClrTh()  const { return m_vecWebClrTh        ; } 
      const std::vector<double>& vecWebLength() const { return m_vecWebLength       ; } 

      const std::string&              ilyName()     const { return m_IlyName     ; }
      double                          ilyPhiLow()   const { return m_IlyPhiLow   ; }
      double                          ilyDelPhi()   const { return m_IlyDelPhi   ; }
      const std::vector<std::string>& vecIlyMat()   const { return m_vecIlyMat   ; }
      const std::vector<double>&      vecIlyThick() const { return m_vecIlyThick ; }

      DDName              hawRName() const { return ddname( m_HawRName ) ; }
      DDName              fawName()  const { return ddname( m_FawName  ) ; }
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

      DDName              gridName()  const { return ddname( m_GridName ) ; }
      DDMaterial          gridMat()   const { return ddmat(  m_GridMat  ) ; }
      double              gridThick() const { return m_GridThick ; }

      double                    backXOff()        const { return m_BackXOff ; } 
      double                    backYOff()        const { return m_BackYOff ; }
      DDName                    backSideName()    const { return ddname( m_BackSideName ) ; }
      double                    backSideLength()  const { return m_BackSideLength         ; } 
      double                    backSideHeight()  const { return m_BackSideHeight         ; }
      double                    backSideWidth()   const { return m_BackSideWidth          ; }
      double                    backSideYOff1()   const { return m_BackSideYOff1         ; }
      double                    backSideYOff2()   const { return m_BackSideYOff2          ; }
      double                    backSideAngle()   const { return m_BackSideAngle          ; }
      DDMaterial                backSideMat()     const { return ddmat( m_BackSideMat )   ; }
      DDName                    extPlateName()    const { return ddname( m_ExtPlateName ) ; }
      double                    extPlateLength()  const { return m_ExtPlateLength    ; }
      double                    extPlateThick()   const { return m_ExtPlateThick     ; }
      double                    extPlateWidth()   const { return m_ExtPlateWidth     ; }
      DDMaterial                extPlateMat()     const { return ddmat( m_ExtPlateMat ) ; }
      DDName                    extSpacerName()   const { return ddname( m_ExtSpacerName ) ; }
      double                    extSpacerThick()  const { return m_ExtSpacerThick    ; }
      double                    extSpacerWidth()  const { return m_ExtSpacerWidth    ; }
      DDMaterial                extSpacerMat()    const { return ddmat( m_ExtSpacerMat ) ; }
      DDName                    backPlateName()   const { return ddname( m_BackPlateName ) ; }
      double                    backPlateLength() const { return m_BackPlateLength   ; }
      double                    backPlateThick()  const { return m_BackPlateThick     ; }
      double                    backPlateWidth()  const { return m_BackPlateWidth     ; }
      DDMaterial                backPlateMat()    const { return ddmat( m_BackPlateMat ) ; }
      const std::string&        grilleName()      const { return m_GrilleName ; }
      double                    grilleThick()     const { return m_GrilleThick         ; }
      double                    grilleWidth()     const { return m_GrilleWidth         ; }
      double                    grilleZSpace()    const { return m_GrilleZSpace         ; }
      DDMaterial                grilleMat()       const { return ddmat( m_GrilleMat )    ; }
      const std::vector<double>& vecGrilleHeight() const { return m_vecGrilleHeight      ; }
      const std::vector<double>& vecGrilleZOff()  const { return m_vecGrilleZOff        ; }
      const std::string         backPipeName()   const { return m_BackPipeName ; }
      const std::vector<double>& vecBackPipeDiam() const { return m_vecBackPipeDiam    ; }
      double                    backPipeThick()  const { return m_BackPipeThick      ; }
      DDMaterial                backPipeMat()    const { return ddmat( m_BackPipeMat ) ; }
      DDMaterial                backPipeWaterMat() const { return ddmat( m_BackPipeWaterMat ) ; }
      const std::string&        backCoolName()    const { return m_BackCoolName ; }
      double                    backCoolWidth()   const { return m_BackCoolWidth      ; }
      const std::vector<double>& vecBackCoolLength() const { return m_vecBackCoolLength  ; }
      const std::vector<double>& vecBackCoolSSVol()  const { return m_vecBackCoolSSVol   ; }
      const std::vector<double>& vecBackCoolAlVol()  const { return m_vecBackCoolAlVol   ; }
      const std::vector<double>& vecBackCoolWaVol()  const { return m_vecBackCoolWaVol   ; }
      const std::string&         backSSName()       const { return m_BackSSName ; }
      const std::string&         backAlName()       const { return m_BackAlName ; }
      const std::string&         backWaName()       const { return m_BackWaName ; }
      DDMaterial                backSSMat()       const { return ddmat( m_BackSSMat ) ; }
      DDMaterial                backAlMat()       const { return ddmat( m_BackAlMat ) ; }
      DDMaterial                backWaMat()       const { return ddmat( m_BackWaMat ) ; }
      const std::vector<double>& vecBackMiscThick() const { return m_vecBackMiscThick ; }
      const std::vector<std::string>& vecBackMiscName() const
                                  { return m_vecBackMiscName ; }
      const std::vector<std::string>& vecBackMiscMat() const
	                          { return m_vecBackMiscMat ; }
      const std::vector<double>& vecPatchPanelThick() const { return m_vecPatchPanelThick ; }
      const std::vector<std::string>& vecPatchPanelNames() const
                                  { return m_vecPatchPanelNames ; }
      const std::vector<std::string>& vecPatchPanelMat() const
	                          { return m_vecPatchPanelMat ; }
      DDName                    patchPanelName()   const { return ddname( m_PatchPanelName ) ; }


protected:

private:

      void web( unsigned int        iWeb,
		double              bWeb,
		double              BWeb,
		double              LWeb,
		double              theta,
		const Pt3D&         corner,
		const DDLogicalPart logPar,
		double&             zee  ,
		double              side,
		double              front,
		double              delta ) ;

      std::string         m_idNameSpace;            //Namespace of this and ALL sub-parts

      // Barrel volume
      std::string         m_BarName    ; // Barrel volume name
      std::string         m_BarMat     ; // Barrel material name
      std::vector<double> m_vecBarZPts ; // Barrel list of z pts
      std::vector<double> m_vecBarRMin ; // Barrel list of rMin pts
      std::vector<double> m_vecBarRMax ; // Barrel list of rMax pts
      std::vector<double> m_vecBarTran ; // Barrel translation
      std::vector<double> m_vecBarRota ; // Barrel rotation
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
      double              m_SpmCutShow  ; // Non-zero means show the box on display (testing only)
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

      std::string         m_WebPlName    ; // string name of web plate volume
      std::string         m_WebClrName   ; // string name of web clearance volume
      std::string         m_WebPlMat     ; // string name of web material
      std::string         m_WebClrMat    ; // string name of web clearance material
      std::vector<double> m_vecWebPlTh   ; // Thickness of web plates
      std::vector<double> m_vecWebClrTh  ; // Thickness of total web clearance
      std::vector<double> m_vecWebLength ; // Length of web plate

      std::string              m_IlyName      ; // string name of inner layer volume
      double                   m_IlyPhiLow    ; // low phi of volumes
      double                   m_IlyDelPhi    ; // delta phi of ily
      std::vector<std::string> m_vecIlyMat    ; // materials of inner layer volumes
      std::vector<double>      m_vecIlyThick  ; // Thicknesses of inner layer volumes

      std::string         m_HawRName ; // string name of half-alveolar wedge
      std::string         m_FawName  ; // string name of full-alveolar wedge
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

      std::string         m_GridName  ; // Grid name
      std::string         m_GridMat   ; // Grid material
      double              m_GridThick ; // Grid Thickness

      double                   m_BackXOff        ; //
      double                   m_BackYOff        ; //

      std::string              m_BackSideName          ; // Back Sides
      double                   m_BackSideLength        ; //
      double                   m_BackSideHeight        ; //
      double                   m_BackSideWidth         ; //
      double                   m_BackSideYOff1         ; //
      double                   m_BackSideYOff2         ; //
      double                   m_BackSideAngle         ; //
      std::string              m_BackSideMat           ; //
      std::string              m_ExtPlateName     ; // external plat
      double                   m_ExtPlateLength   ; // 
      double                   m_ExtPlateThick    ; //
      double                   m_ExtPlateWidth    ; //
      std::string              m_ExtPlateMat      ; //
      std::string              m_ExtSpacerName    ; // external spacer
      double                   m_ExtSpacerThick   ; //
      double                   m_ExtSpacerWidth   ; //
      std::string              m_ExtSpacerMat     ; //
      std::string              m_BackPlateName    ; // back plate
      double                   m_BackPlateLength  ; //
      double                   m_BackPlateThick   ; //
      double                   m_BackPlateWidth   ; //
      std::string              m_BackPlateMat     ; //
      std::string              m_GrilleName      ; // grille
      double                   m_GrilleThick     ; //
      double                   m_GrilleWidth     ; //
      double                   m_GrilleZSpace    ; //
      std::string              m_GrilleMat       ; //
      std::vector<double>      m_vecGrilleHeight ; //
      std::vector<double>      m_vecGrilleZOff   ; //
      std::string              m_BackPipeName    ; // 
      std::vector<double>      m_vecBackPipeDiam ; // pipes
      double                   m_BackPipeThick   ; //
      std::string              m_BackPipeMat     ; //
      std::string              m_BackPipeWaterMat ; //
      std::string              m_BackCoolName     ; // cooling circuits
      double                   m_BackCoolWidth    ; //
      std::vector<double>      m_vecBackCoolLength; //
      std::vector<double>      m_vecBackCoolSSVol    ; //
      std::vector<double>      m_vecBackCoolAlVol    ; //
      std::vector<double>      m_vecBackCoolWaVol    ; //
      std::string              m_BackSSName       ; //
      std::string              m_BackAlName       ; //
      std::string              m_BackWaName       ; //
      std::string              m_BackSSMat        ; //
      std::string              m_BackAlMat        ; //
      std::string              m_BackWaMat        ; //
      std::vector<double>      m_vecBackMiscThick ; // misc materials
      std::vector<std::string> m_vecBackMiscName  ; //
      std::vector<std::string> m_vecBackMiscMat   ; //
      std::string              m_PatchPanelName    ; //
      std::vector<double>      m_vecPatchPanelThick ; // patch panel materials
      std::vector<std::string> m_vecPatchPanelNames  ; //
      std::vector<std::string> m_vecPatchPanelMat   ; //
}; 

#endif
