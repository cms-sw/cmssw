#ifndef MuonCommonData_DDGEMAngular_h
#define MuonCommonData_DDGEMAngular_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDGEMAngular : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDGEMAngular(); 
  ~DDGEMAngular() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  double        startAngle;   //Start angle 
  double        stepAngle;    //Step  angle
  int           invert;       //inverted or forward
  double        rPos;         //Radial position of center
  double        zoffset;      //Offset in z
  int           n;            //Mumber of copies
  int           startCopyNo;  //Start copy Number
  int           incrCopyNo;   //Increment copy Number

  std::string   rotns;        //Namespace for rotation matrix
  std::string   idNameSpace;  //Namespace of this and ALL sub-parts
  std::string   childName;    //Children name
};

#endif
