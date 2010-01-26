#ifndef HcalAlgo_DDHCalTBZposAlgo_h
#define HcalAlgo_DDHCalTBZposAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDHCalTBZposAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDHCalTBZposAlgo(); 
  virtual ~DDHCalTBZposAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  double        eta;         //Eta at which beam is focussed
  double        theta;       //Corresponding theta value
  double        shiftY;      //Shift along Y
  double        shiftX;      //Shift along X
  double        zoffset;     //Offset in z
  double        dist;        //Radial distance
  double        tilt;        //Tilt with respect to y-axis
  int           copyNumber;  //Copy Number

  std::string   idNameSpace; //Namespace of this and ALL sub-parts
  std::string   childName;   //Children name
};

#endif
