#ifndef DD_TrackerAngularV1_h
#define DD_TrackerAngularV1_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTrackerAngularV1 : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDTrackerAngularV1(); 
  virtual ~DDTrackerAngularV1();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

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
