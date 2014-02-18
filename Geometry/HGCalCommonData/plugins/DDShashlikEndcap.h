#ifndef HGCalCommonData_DDShashlikEndcap_h
#define HGCalCommonData_DDShashlikEndcap_h

#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDShashlikEndcap : public DDAlgorithm
{
public:
  DDShashlikEndcap( void ); 
  virtual ~DDShashlikEndcap( void );
  
  void initialize( const DDNumericArguments & nArgs,
		   const DDVectorArguments & vArgs,
		   const DDMapArguments & mArgs,
		   const DDStringArguments & sArgs,
		   const DDStringVectorArguments & vsArgs );

  void createQuarter( DDCompactView& cpv, int xQuadrant, int yQuadrant );

  void execute( DDCompactView& cpv );



private:

  double        m_startAngle;   // Start angle 
  double        m_stepAngle;    // Step  angle
  double        m_tiltAngle;    // Tilt  angle
  int           m_invert;       // Inverted or forward
  double        m_rMin;         // Inner radius
  double        m_rMax;         // Outer radius
  double        m_rPos;         // Radial position of center
  double        m_xyoffset;     // Offset in x or y
  double        m_zoffset;      // Offset in z
  int           m_n;            // Mumber of copies
  int           m_startCopyNo;  // Start copy Number
  int           m_incrCopyNo;   // Increment copy Number
  std::string   m_childName;    // Children name
  std::string   m_idNameSpace;  // Namespace of this and ALL sub-parts
};

#endif
