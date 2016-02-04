#ifndef ALGORITHM_DD_ANGULAR_H
# define ALGORITHM_DD_ANGULAR_H

# include "DetectorDescription/Base/interface/DDTypes.h"
# include "DetectorDescription/Base/interface/DDRotationMatrix.h"
# include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDAngular : public DDAlgorithm
{
public:
  DDAngular( void );
  virtual ~DDAngular( void );

  void initialize( const DDNumericArguments & nArgs,
                   const DDVectorArguments & vArgs,
                   const DDMapArguments & mArgs,
                   const DDStringArguments & sArgs,
                   const DDStringVectorArguments & vsArgs );

  void execute( DDCompactView& cpv );

private:

  DD3Vector 	fUnitVector( double theta, double phi );
  int           m_n;              //Number of copies
  int           m_startCopyNo;    //Start Copy number
  int           m_incrCopyNo;     //Increment in Copy number
  double        m_startAngle;     //Start angle
  double        m_rangeAngle;     //Range in angle
  double        m_radius;         //Radius
  std::vector<double> m_center;   //Phi values
  std::vector<double> m_rotateSolid; //Rotation of the solid values
  
  double        m_delta;          //Increment in phi
  std::string   m_idNameSpace;    //Namespace of this and ALL sub-parts
  std::pair<std::string, std::string> m_childNmNs; //Child name
                                                   //Namespace of the child
  
  DDRotationMatrix m_solidRot;    //Rotation of the solid
};

#endif // ALGORITHM_DD_ANGULAR_H
