#ifndef DD_TIDRingAlgo_h
#define DD_TIDRingAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTIDRingAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDTIDRingAlgo(); 
  virtual ~DDTIDRingAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute();

private:

  std::string         idNameSpace;       //Namespace of this and ALL sub-parts
  std::string         moduleName;        //Name of the module
  std::string         iccName;           //Name of the ICC
  std::string         coolName;          //Name of the Cool insert

  int                 number;            //Number of copies
  double              startAngle;        //Phi offset
  double              rModule;           //Location of module in R 
  std::vector<double> zModule;           //                   in Z
  double              rICC;              //Location of ICC    in R 
  std::vector<double> zICC;              //                   in Z
  std::vector<double> rCool;             //Location of cool insert in R 
  std::vector<double> zCool;             //                        in Z

  double              fullHeight;        //Height 
  double              dlTop;             //Width at top of wafer
  double              dlBottom;          //Width at bottom of wafer
  double              dlHybrid;          //Width at the hybrid end
  double              topFrameHeight;    //Top Frmae     height
  double              bottomFrameHeight; //Bottom of the frame
  double              bottomFrameOver;   //              overlap
  double              sideFrameWidth;    //Side Frame    width
  double              sideFrameOver;     //              overlap
  double              hybridHeight;      //Hybrid        height
  double              hybridWidth;       //              width
  double              coolWidth;         //Cool insert   width
  int                 coolSide;          //              on sides

  
};

#endif
