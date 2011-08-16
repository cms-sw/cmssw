#ifndef DD_TrackerZPosAlgo_h
#define DD_TrackerZPosAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTrackerZPosAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTrackerZPosAlgo(); 
  virtual ~DDTrackerZPosAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::vector<double>      zvec;   //Z positions
  std::vector<std::string> rotMat; //Names of rotation matrices

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
  std::string              childName;   //Child name
  int                      startCopyNo; //Start Copy number
  int                      incrCopyNo;  //Increment in Copy number
};

#endif
