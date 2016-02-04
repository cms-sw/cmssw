#ifndef EcalTestBeam_DDTBH4Algo_h
#define EcalTestBeam_DDTBH4Algo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDTransform.h"

//CLHEP 
#include <CLHEP/Geometry/Transform3D.h>

class DDTBH4Algo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTBH4Algo(); 
  virtual ~DDTBH4Algo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

      DDMaterial ddmat(  const std::string& s ) const ;
      DDName     ddname( const std::string& s ) const ;
      DDRotation myrot(  const std::string& s,
			 const CLHEP::HepRotation& r ) const ;

      const std::string&         idNameSpace() const { return m_idNameSpace   ; }

      double                   blZBeg      () const { return m_BLZBeg      ; }
      double                   blZEnd      () const { return m_BLZEnd      ; }
      double                   blZPiv      () const { return m_BLZPiv      ; }
      double                   blRadius    () const { return m_BLRadius    ; }
      std::string              vacName     () const { return m_VacName    ; }
      DDMaterial                      vacMat      () const { return ddmat(m_VacMat); }
      const std::vector<double>&      vecVacZBeg  () const { return m_vecVacZBeg  ; }
      const std::vector<double>&      vecVacZEnd  () const { return m_vecVacZEnd  ; }
      std::string                     winName     () const { return m_WinName    ; }
      const std::vector<std::string>& vecWinMat   () const { return m_vecWinMat     ; }
      const std::vector<double>&      vecWinZBeg  () const { return m_vecWinZBeg ; }
      const std::vector<double>&      vecWinThick () const { return m_vecWinThick; }

      DDMaterial                      trgMat      () const { return ddmat(m_TrgMat)  ; } 
      DDMaterial                      holeMat     () const { return ddmat(m_HoleMat)  ; } 
      double                          trgVetoHoleRadius() const { return m_TrgVetoHoleRadius; }
      const std::vector<std::string>& vecTrgName  () const { return m_vecTrgName  ; } 
      const std::vector<double>&      vecTrgSide () const { return m_vecTrgSide ; } 
      const std::vector<double>&      vecTrgThick() const { return m_vecTrgThick; } 
      const std::vector<double>&      vecTrgPhi  () const { return m_vecTrgPhi  ; } 
      const std::vector<double>&      vecTrgXOff () const { return m_vecTrgXOff ; } 
      const std::vector<double>&      vecTrgYOff () const { return m_vecTrgYOff ; } 
      const std::vector<double>&      vecTrgZPiv () const { return m_vecTrgZPiv ; } 

      DDName                         fibFibName  () const { return ddname(m_FibFibName)  ; } 
      DDName                         fibCladName () const { return ddname(m_FibCladName) ; } 
      DDMaterial                     fibFibMat   () const { return ddmat(m_FibFibMat)   ; } 
      DDMaterial                     fibCladMat  () const { return ddmat(m_FibCladMat)  ; } 
      double                         fibSide     () const { return m_FibSide  ; } 
      double                         fibCladThick() const { return m_FibCladThick; } 
      double                         fibLength   () const { return m_FibLength   ; } 
      const std::vector<double>&     vecFibPhi   () const { return m_vecFibPhi   ; } 
      const std::vector<double>&     vecFibXOff  () const { return m_vecFibXOff  ; } 
      const std::vector<double>&     vecFibYOff  () const { return m_vecFibYOff  ; } 
      const std::vector<double>&     vecFibZPiv  () const { return m_vecFibZPiv  ; }

private:
      std::string         m_idNameSpace;            //Namespace of this and ALL sub-parts

      double                   m_BLZBeg      ; //
      double                   m_BLZEnd      ; //
      double                   m_BLZPiv      ; //
      double                   m_BLRadius    ; //
      std::string              m_VacName     ; // 
      std::string              m_VacMat      ; // 
      std::vector<double>      m_vecVacZBeg  ; // 
      std::vector<double>      m_vecVacZEnd  ; // 
      std::string              m_WinName     ; // 
      std::vector<std::string> m_vecWinMat   ; // 
      std::vector<double>      m_vecWinZBeg  ; // 
      std::vector<double>      m_vecWinThick ; // 

      std::string              m_TrgMat      ; // 
      std::string              m_HoleMat      ; // 
      double                   m_TrgVetoHoleRadius    ; //
      std::vector<std::string> m_vecTrgName  ; // 
      std::vector<double>      m_vecTrgSide ; // 
      std::vector<double>      m_vecTrgThick; // 
      std::vector<double>      m_vecTrgPhi  ; // 
      std::vector<double>      m_vecTrgXOff ; // 
      std::vector<double>      m_vecTrgYOff ; // 
      std::vector<double>      m_vecTrgZPiv ; // 

      std::string              m_FibFibName  ; // 
      std::string              m_FibCladName ; // 
      std::string              m_FibFibMat   ; // 
      std::string              m_FibCladMat  ; // 
      double                   m_FibSide     ; //
      double                   m_FibCladThick; //
      double                   m_FibLength   ; //
      std::vector<double>      m_vecFibPhi   ; // 
      std::vector<double>      m_vecFibXOff  ; // 
      std::vector<double>      m_vecFibYOff  ; // 
      std::vector<double>      m_vecFibZPiv  ; // 

};

#endif
