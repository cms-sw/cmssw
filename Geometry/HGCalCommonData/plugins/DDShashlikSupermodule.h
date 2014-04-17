#ifndef HGCalCommonData_DDShashlikSupermodule_h
#define HGCalCommonData_DDShashlikSupermodule_h

#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDShashlikSupermodule : public DDAlgorithm
{
public:
  DDShashlikSupermodule( void ); 
  virtual ~DDShashlikSupermodule( void );
  
  void initialize( const DDNumericArguments & nArgs,
		   const DDVectorArguments & vArgs,
		   const DDMapArguments & mArgs,
		   const DDStringArguments & sArgs,
		   const DDStringVectorArguments & vsArgs );

  void execute( DDCompactView& cpv );

private:

  double        m_startAngle;   // Start angle 
  double        m_stepAngle;    // Step  angle
  int           m_invert;       // Inverted or forward
  double        m_rPos;         // Radial position of center
  double        m_zoffset;      // Offset in z
  double        m_SupermoduleWidthFront;
  double        m_SupermoduleWidthBack;
  double        m_SupermoduleThickness;
  double        m_SupermoduleConcaveDepth;
  int           m_n;            // Mumber of copies
  int           m_startCopyNo;  // Start copy Number
  int           m_incrCopyNo;   // Increment copy Number
  std::string   m_childName;    // Children name
  std::string   m_idNameSpace;  // Namespace of this and ALL sub-parts
};

#endif
