#ifndef DD_TrackerLinear_h
#define DD_TrackerLinear_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTrackerLinear : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTrackerLinear(); 
  ~DDTrackerLinear() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  std::string              childName;   //Child name
  int                      number;      //Number of positioning
  int                      startcn;     //Start copy no index
  int                      incrcn;      //Increment of copy no.
  double                   theta;       //Direction of translation
  double                   phi;         //  ......
  double                   offset;      //Offset    along (theta,phi) direction
  double                   delta;       //Increment     ................
  std::vector<double>      centre;      //Centre
  std::string              rotMat;      //Rotation matrix
};

#endif
