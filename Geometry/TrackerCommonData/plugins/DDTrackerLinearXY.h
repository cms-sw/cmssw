#ifndef DD_TrackerLinearXY_h
#define DD_TrackerLinearXY_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTrackerLinearXY : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTrackerLinearXY(); 
  ~DDTrackerLinearXY() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  std::string              childName;   //Child name
  int                      numberX;     //Number of positioning along X-axis
  double                   deltaX;      //Increment               .........
  int                      numberY;     //Number of positioning along Y-axis
  double                   deltaY;      //Increment               .........
  std::vector<double>      centre;      //Centre
};

#endif
