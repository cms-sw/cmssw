#ifndef DD_EcalEndcapAlgo_h
#define DD_EcalEndcapAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "Geometry/CaloGeometry/interface/EcalTrapezoidParameters.h"
#include "CLHEP/Geometry/Transform3D.h" 


class DDEcalEndcapAlgo : public DDAlgorithm {
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
      DDEcalEndcapAlgo();
      virtual ~DDEcalEndcapAlgo();

      void initialize(const DDNumericArguments      & nArgs,
		      const DDVectorArguments       & vArgs,
		      const DDMapArguments          & mArgs,
		      const DDStringArguments       & sArgs,
		      const DDStringVectorArguments & vsArgs);
      void execute(DDCompactView& cpv);

      //  New methods for SC geometry
      void EEPositionCRs( const DDName        pName,
			  const DDTranslation offset,
			  const int iSCType,
			  DDCompactView& cpv );

      void EECreateSC( const unsigned int iSCType, DDCompactView& cpv );

      void EECreateCR();

      void EEPosSC( const int iCol , 
		    const int iRow , 
		    DDName    EEDeeName );

      unsigned int EEGetSCType( const unsigned int iCol , 
				const unsigned int iRow  ) ;

      DDName EEGetSCName( const int iCol , 
			  const int iRow  ) ;

      std::vector<double> EEGetSCCtrs( const int iCol , 
				       const int iRow  );

      DDMaterial ddmat(  const std::string& s ) const ;
      DDName     ddname( const std::string& s ) const ;
      DDRotation myrot(  const std::string& s,
			 const DDRotationMatrix& r ) const ;

      const std::string&         idNameSpace() const { return m_idNameSpace   ; }

      // endcap parent volume
      DDMaterial                 eeMat()      const { return ddmat(  m_EEMat  ) ; }
      double                     eezOff()     const { return m_EEzOff  ; }

      DDName                     eeQuaName()     const { return ddname( m_EEQuaName ) ; }
      DDMaterial                 eeQuaMat()      const { return ddmat(  m_EEQuaMat  ) ; }

      DDMaterial                 eeCrysMat()  const { return ddmat( m_EECrysMat ) ; }
      DDMaterial                 eeWallMat()  const { return ddmat( m_EEWallMat ) ; }

      double                     eeCrysLength() const { return m_EECrysLength ; }
      double                     eeCrysRear()   const { return m_EECrysRear ; }
      double                     eeCrysFront()  const { return m_EECrysFront ; }
      double                     eeSCELength()  const { return m_EESCELength ; }
      double                     eeSCERear()    const { return m_EESCERear ; }
      double                     eeSCEFront()   const { return m_EESCEFront ; }
      double                     eeSCALength()  const { return m_EESCALength ; }
      double                     eeSCARear()    const { return m_EESCARear ; }
      double                     eeSCAFront()   const { return m_EESCAFront ; }
      double                     eeSCAWall()    const { return m_EESCAWall ; }
      double                     eeSCHLength()  const { return m_EESCHLength ; }
      double                     eeSCHSide()    const { return m_EESCHSide ; }

      double                     eenSCTypes()   const { return m_EEnSCTypes ; }
      double                     eenColumns()   const { return m_EEnColumns ; }
      double                     eenSCCutaway() const { return m_EEnSCCutaway ; }
      double                     eenSCquad()    const { return m_EEnSCquad ; }
      double                     eenCRSC()      const { return m_EEnCRSC ; }
      const std::vector<double>& eevecEESCProf() const { return m_vecEESCProf ; }
      const std::vector<double>& eevecEEShape() const { return m_vecEEShape ; }
      const std::vector<double>& eevecEESCCutaway() const { return m_vecEESCCutaway ; }
      const std::vector<double>& eevecEESCCtrs() const { return m_vecEESCCtrs ; }
      const std::vector<double>& eevecEECRCtrs() const { return m_vecEECRCtrs ; }

      DDName                     cutBoxName()    const { return ddname( m_cutBoxName ) ; }
      double                     eePFHalf()      const { return m_PFhalf ; }
      double                     eePFFifth()     const { return m_PFfifth ; }
      double                     eePF45()        const { return m_PF45 ; }

      DDName  envName( unsigned int i )    const { return ddname( m_envName + int_to_string(i)  ) ; }
      DDName  alvName( unsigned int i )    const { return ddname( m_alvName + int_to_string(i)  ) ; }
      DDName  intName( unsigned int i )    const { return ddname( m_intName + int_to_string(i)  ) ; }
      DDName                     cryName()    const { return ddname( m_cryName ) ; }

      DDName                     addTmp( DDName aName ) const { return ddname( aName.name() + "Tmp" ) ; }

      const DDTranslation& cryFCtr( unsigned int iRow,
				    unsigned int iCol  ) const { return m_cryFCtr[iRow-1][iCol-1] ; }

      const DDTranslation& cryRCtr( unsigned int iRow,
				    unsigned int iCol  ) const { return m_cryRCtr[iRow-1][iCol-1] ; }

      const DDTranslation& scrFCtr( unsigned int iRow,
				    unsigned int iCol  ) const { return m_scrFCtr[iRow-1][iCol-1] ; }

      const DDTranslation& scrRCtr( unsigned int iRow,
				    unsigned int iCol  ) const { return m_scrRCtr[iRow-1][iCol-1] ; }

      const std::vector<double>& vecEESCLims() const { return m_vecEESCLims ; }

      double                     iLength()      const { return m_iLength ; }
      double                     iXYOff()       const { return m_iXYOff ; }

protected:

private:

      std::string         m_idNameSpace;            //Namespace of this and ALL sub-parts

      // Barrel volume
      std::string         m_EEMat     ;
      double              m_EEzOff    ;

      std::string         m_EEQuaName    ; 
      std::string         m_EEQuaMat     ;

      std::string         m_EECrysMat;
      std::string         m_EEWallMat;

      double              m_EECrysLength;
      double              m_EECrysRear;
      double              m_EECrysFront;
      double              m_EESCELength;
      double              m_EESCERear;
      double              m_EESCEFront;
      double              m_EESCALength;
      double              m_EESCARear;
      double              m_EESCAFront;
      double              m_EESCAWall;
      double              m_EESCHLength;
      double              m_EESCHSide;

      double m_EEnSCTypes;
      std::vector<double> m_vecEESCProf;
      double m_EEnColumns;
      std::vector<double> m_vecEEShape;
      double m_EEnSCCutaway;
      std::vector<double> m_vecEESCCutaway;
      double m_EEnSCquad;     
      std::vector<double> m_vecEESCCtrs;
      double m_EEnCRSC;
      std::vector<double> m_vecEECRCtrs;

      const std::vector<double>* m_cutParms ;
      std::string         m_cutBoxName ;

      std::string         m_envName ;
      std::string         m_alvName ;
      std::string         m_intName ;
      std::string         m_cryName ;

      DDTranslation m_cryFCtr[5][5] ;
      DDTranslation m_cryRCtr[5][5] ;

      DDTranslation m_scrFCtr[10][10] ;
      DDTranslation m_scrRCtr[10][10] ;

      double m_PFhalf ;
      double m_PFfifth ;
      double m_PF45 ;

      std::vector<double> m_vecEESCLims;

      double m_iLength ;

      double m_iXYOff ;

}; 

#endif
