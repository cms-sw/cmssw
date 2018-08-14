#ifndef DD_EcalAPDAlgo_h
#define DD_EcalAPDAlgo_h

#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTransform.h"

class DDEcalAPDAlgo : public DDAlgorithm {

public:

  //Constructor and Destructor
  DDEcalAPDAlgo();
  ~DDEcalAPDAlgo() override;

  void initialize(const DDNumericArguments      & nArgs,
		  const DDVectorArguments       & vArgs,
		  const DDMapArguments          & mArgs,
		  const DDStringArguments       & sArgs,
		  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:

private:

  DDName     ddname( const std::string& s ) const ;

  const std::vector<double>& vecCerPos()  const { return m_vecCerPos  ; }
  int        apdHere () const { return m_APDHere ; }
  
  DDName     capName () const { return ddname(m_capName) ; }
  DDMaterial capMat  () const { return DDMaterial( ddname(m_capMat) ) ; }
  double     capXSize() const { return m_capXSize; }
  double     capYSize() const { return m_capYSize; }
  double     capThick() const { return m_capThick; }

  DDName     cerName () const { return ddname(m_CERName) ; }
  DDMaterial cerMat  () const { return DDMaterial( ddname(m_CERMat) ) ; }
  double     cerXSize() const { return m_CERXSize; }
  double     cerYSize() const { return m_CERYSize; }
  double     cerThick() const { return m_CERThick; }

  DDName     bsiName () const { return ddname(m_BSiName) ; }
  DDMaterial bsiMat  () const { return DDMaterial( ddname(m_BSiMat) ) ; }
  double     bsiXSize() const { return m_BSiXSize; }
  double     bsiYSize() const { return m_BSiYSize; }
  double     bsiThick() const { return m_BSiThick; }

  DDName     sglName () const { return ddname(m_SGLName) ; }
  DDMaterial sglMat  () const { return DDMaterial( ddname(m_SGLMat) ) ; }
  double     sglThick() const { return m_SGLThick; }

  DDName     atjName () const { return ddname(m_ATJName) ; }
  DDMaterial atjMat  () const { return DDMaterial( ddname(m_ATJMat) ) ; }
  double     atjThick() const { return m_ATJThick; }
  
  DDName     aglName () const { return ddname(m_AGLName) ; }
  DDMaterial aglMat  () const { return DDMaterial( ddname(m_AGLMat) ) ; }
  double     aglThick() const { return m_AGLThick; }

  DDName     andName () const { return ddname(m_ANDName) ; }
  DDMaterial andMat  () const { return DDMaterial( ddname(m_ANDMat) ) ; }
  double     andThick() const { return m_ANDThick; }

  DDName     apdName () const { return ddname(m_APDName) ; }
  DDMaterial apdMat  () const { return DDMaterial( ddname(m_APDMat) ) ; }
  double     apdSide () const { return m_APDSide ; }
  double     apdThick() const { return m_APDThick; }
  double     apdZ    () const { return m_APDZ    ; }
  double     apdX1   () const { return m_APDX1   ; }
  double     apdX2   () const { return m_APDX2   ; }

private:

  std::string              m_idNameSpace; //Namespace of this and ALL sub-parts

  std::vector<double>      m_vecCerPos    ; // Translation
  int                      m_APDHere      ;

  std::string              m_capName      ; // Capsule
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

}; 

#endif
