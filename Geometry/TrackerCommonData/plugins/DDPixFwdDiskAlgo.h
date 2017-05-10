#ifndef DD_PixFwdDiskAlgo_h
#define DD_PixFwdDiskAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDPixFwdDiskAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDPixFwdDiskAlgo(); 
  virtual ~DDPixFwdDiskAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
  std::string              childName;   //Child name
  std::string              rotName;     //Name of the base rotation matrix
  std::string              flagString;  //Flag if a blade is present
  int                      nBlades;     //Number of blades
  int                      startCopyNo; //Start Copy number
  double                   bladeAngle;  //Angle of blade rotation aroung y-axis
  double                   zPlane;      //Common shift in z for all blades
  std::vector<double>      bladeZShift; //Shift in Z of individual blades
  double                   anchorR;     //Distance of beam line to anchor point
  double                   bladeTilt;   //Tilt of the blade around x-axis
};

#endif
