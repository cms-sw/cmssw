#ifndef HGCalCommonData_DDShashlikNoTaperSupermodule_h
#define HGCalCommonData_DDShashlikNoTaperSupermodule_h

#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDShashlikNoTaperSupermodule : public DDAlgorithm
{
public:
  DDShashlikNoTaperSupermodule( void ); 
  virtual ~DDShashlikNoTaperSupermodule( void );
  
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
  int           m_n;            // Mumber of copies
  int           m_startCopyNo;  // Start copy Number
  int           m_incrCopyNo;   // Increment copy Number
  std::string   m_childName;    // Children name
  std::string   m_idNameSpace;  // Namespace of this and ALL sub-parts
};

#endif
