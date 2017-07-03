#ifndef DD_TrackerAngular_h
#define DD_TrackerAngular_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTrackerAngular : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDTrackerAngular(); 
  ~DDTrackerAngular() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  int           n;              //Number of copies
  int           startCopyNo;    //Start Copy number
  int           incrCopyNo;     //Increment in Copy number
  double        rangeAngle;     //Range in angle
  double        startAngle;     //Start anle
  double        radius;         //Radius
  std::vector<double> center;   //Phi values
  double        delta;          //Increment in phi

  std::string   idNameSpace;    //Namespace of this and ALL sub-parts
  std::string   childName;      //Child name
};

#endif
