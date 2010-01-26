#ifndef DD_TrackerPhiAltAlgo_h
#define DD_TrackerPhiAltAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTrackerPhiAltAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDTrackerPhiAltAlgo(); 
  virtual ~DDTrackerPhiAltAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  double        tilt;        //Tilt of the module
  double        startAngle;  //offset in phi
  double        rangeAngle;  //Maximum range in phi
  double        radiusIn;    //Inner radius
  double        radiusOut;   //Outer radius
  double        zpos;        //z position
  int           number;      //Number of copies
  int           startCopyNo; //Start copy number
  int           incrCopyNo;  //Increment in copy number

  std::string   idNameSpace; //Namespace of this and ALL sub-parts
  std::string   childName;   //Child name
};

#endif
