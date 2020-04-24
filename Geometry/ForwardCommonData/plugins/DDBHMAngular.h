#ifndef ForwardCommonData_DDBHMAngular_h
#define ForwardCommonData_DDBHMAngular_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDBHMAngular : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDBHMAngular(); 
  ~DDBHMAngular() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  int           units;        //Number of copies
  double        rr;           //Radial position of the detectors
  double        dphi;         //the distance in phi between the detectors

  std::string   rotMat;       //Name of the rotation matrix
  std::string   childName;    //Children name
};

#endif
