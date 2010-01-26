#ifndef DD_TrackerLinear_h
#define DD_TrackerLinear_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTrackerLinear : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTrackerLinear(); 
  virtual ~DDTrackerLinear();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
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
