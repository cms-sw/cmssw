#ifndef DD_TECCoolAlgo_h
#define DD_TECCoolAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTECCoolAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTECCoolAlgo(); 
  virtual ~DDTECCoolAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs);

  void execute();

private:

  std::string              idNameSpace;    //Namespace of this and ALL parts
  int                      startCopyNo;    //Start copy number
  std::vector<std::string> petalName;      //Name of the petals
  std::vector<double>      petalRmax;      //Petal         R extent
  std::vector<double>      petalWidth;     //              width
  std::vector<std::string> coolName;       //Name of cooling pieces
  std::vector<double>      coolR;          //Cooling piece radii 
  std::vector<int>         coolInsert;     //              positioning
  double                   startAngle;     //Ring          Start angle
  double                   incrAngle;      //              increment
  double                   rmin;           //Detector      Rmin
  double                   fullHeight;     //              Height 
  double                   dlTop;          //Width at top of wafer
  double                   dlBottom;       //Width at bottom of wafer
  double                   dlHybrid;       //Width at the hybrid end
  double                   frameWidth;     //Frame         width
  double                   frameOver;      //              overlap (on sides)
  double                   topFrameHeight; //Top frame     height
  double                   hybridHeight;   //Hybrid        height
  double                   hybridWidth;    //              width

};

#endif
