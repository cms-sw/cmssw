#ifndef HGCalCommonData_DDHGCalNoTaperEndcap_h
#define HGCalCommonData_DDHGCalNoTaperEndcap_h

#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDHGCalNoTaperEndcap : public DDAlgorithm {

public:
  DDHGCalNoTaperEndcap( void ); 
  virtual ~DDHGCalNoTaperEndcap( void );
  
  void initialize( const DDNumericArguments & nArgs,
		   const DDVectorArguments & vArgs,
		   const DDMapArguments & mArgs,
		   const DDStringArguments & sArgs,
		   const DDStringVectorArguments & vsArgs );

  void execute( DDCompactView& cpv );

private:

  int createQuarter( DDCompactView& cpv, int xQuadrant, int yQuadrant, int startCopyNo );

  double        m_startAngle;   // Start angle 
  double        m_tiltAngle;    // Tilt  angle
  int           m_invert;       // Inverted or forward
  double        m_rMin;         // Inner radius
  double        m_rMax;         // Outer radius
  double        m_zoffset;      // Offset in z
  double        m_xyoffset;     // Offset in x or y
  int           m_n;            // Mumber of copies
  int           m_startCopyNo;  // Start copy Number
  int           m_incrCopyNo;   // Increment copy Number
  std::string   m_childName;    // Children name
  std::string   m_idNameSpace;  // Namespace of this and ALL sub-parts
};

#endif
