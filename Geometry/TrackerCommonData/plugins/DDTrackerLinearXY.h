#ifndef DD_TrackerLinearXY_h
#define DD_TrackerLinearXY_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTrackerLinearXY : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTrackerLinearXY(); 
  virtual ~DDTrackerLinearXY();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
  std::string              childName;   //Child name
  int                      numberX;     //Number of positioning along X-axis
  double                   deltaX;      //Increment               .........
  int                      numberY;     //Number of positioning along Y-axis
  double                   deltaY;      //Increment               .........
  std::vector<double>      centre;      //Centre
};

#endif
