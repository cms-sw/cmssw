#ifndef HcalAlgo_DDHCalLinearXY_h
#define HcalAlgo_DDHCalLinearXY_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHCalLinearXY : public DDAlgorithm {

public:
  //Constructor and Destructor
  DDHCalLinearXY(); 
  ~DDHCalLinearXY() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  std::vector<std::string> childName;   //Child name
  int                      numberX;     //Number of positioning along X-axis
  double                   deltaX;      //Increment               .........
  int                      numberY;     //Number of positioning along Y-axis
  double                   deltaY;      //Increment               .........
  std::vector<double>      centre;      //Centre
};

#endif
