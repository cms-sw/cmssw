#ifndef HcalAlgo_DDHCalAngular_h
#define HcalAlgo_DDHCalAngular_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDHCalAngular : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDHCalAngular(); 
  virtual ~DDHCalAngular();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  double        startAngle;   //Start angle 
  double        rangeAngle;   //Range angle
  double        shiftY;       //Shift along Y
  double        shiftX;       //Shift along X
  double        zoffset;      //Offset in z
  int           n;            //Mumber of copies
  int           startCopyNo;  //Start copy Number
  int           incrCopyNo;   //Increment copy Number

  std::string   rotns;        //Namespace for rotation matrix
  std::string   idNameSpace;  //Namespace of this and ALL sub-parts
  std::string   childName;    //Children name
};

#endif
