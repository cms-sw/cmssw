#ifndef DD_EcalEndcapAlgo_h
#define DD_EcalEndcapAlgo_h

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

class DDEcalEndcapAlgo : public DDAlgorithm {
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
      DDEcalEndcapAlgo();
      virtual ~DDEcalEndcapAlgo();

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

      const std::string&         idNameSpace() const { return m_idNameSpace   ; }

      // endcap parent volume
      DDName                     eeName()     const { return ddname( m_EEName ) ; }
      DDMaterial                 eeMat()      const { return ddmat(  m_EEMat  ) ; }
      double                     eedz()       const { return m_EEdz    ; }
      double                     eerMin1()    const { return m_EErMin1 ; }
      double                     eerMin2()    const { return m_EErMin2 ; }
      double                     eerMax1()    const { return m_EErMax1 ; }
      double                     eerMax2()    const { return m_EErMax2 ; }
      double                     eezOff()     const { return m_EEzOff  ; }

protected:

private:

      void front() ;
      void back()  ;


      std::string         m_idNameSpace;            //Namespace of this and ALL sub-parts

      // Barrel volume
      std::string         m_EEName    ; 
      std::string         m_EEMat     ;
      double              m_EEdz      ;
      double              m_EErMin1   ;
      double              m_EErMin2   ;
      double              m_EErMax1   ;
      double              m_EErMax2   ;
      double              m_EEzOff    ;
}; 

#endif
