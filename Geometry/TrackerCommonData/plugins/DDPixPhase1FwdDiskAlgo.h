#ifndef DD_PixPhase1FwdDiskAlgo_h
#define DD_PixPhase1FwdDiskAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDPixPhase1FwdDiskAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDPixPhase1FwdDiskAlgo(); 
  ~DDPixPhase1FwdDiskAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

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
