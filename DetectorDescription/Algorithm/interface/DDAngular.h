#ifndef ALGORITHM_DD_ANGULAR_H
# define ALGORITHM_DD_ANGULAR_H

# include "DetectorDescription/Base/interface/DDTypes.h"
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
 
  int           n;              //Number of copies
  int           startCopyNo;    //Start Copy number
  int           incrCopyNo;     //Increment in Copy number
  double        startAngle;     //Start anle
  double        rangeAngle;     //Range in angle
  double        radius;         //Radius
  std::vector<double> center;   //Phi values
  std::vector<double> rotateSolid; //Rotation of the solid values
  
  double        delta;          //Increment in phi
 
  std::string   idNameSpace;    //Namespace of this and ALL sub-parts
  std::string   childName;      //Child name
};

#endif // ALGORITHM_DD_ANGULAR_H
