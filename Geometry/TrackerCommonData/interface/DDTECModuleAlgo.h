#ifndef DD_TECModuleAlgo_h
#define DD_TECModuleAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTECModuleAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTECModuleAlgo(); 
  virtual ~DDTECModuleAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs);

  void execute();

private:

  std::string              idNameSpace;    //Namespace of this and ALL parts
  std::string              genMat;         //General material name
  double                   moduleThick;    //Module thickness
  double                   detTilt;        //Tilt of stereo detector
  double                   fullHeight;     //Height 
  double                   dlTop;          //Width at top of wafer
  double                   dlBottom;       //Width at bottom of wafer
  double                   dlHybrid;       //Width at the hybrid end
  double                   frameWidth;     //Frame         width
  double                   frameThick;     //              thickness
  double                   frameOver;      //              overlap (on sides)
  std::string              topFrameMat;    //Top frame     material
  double                   topFrameHeight; //              height
  double                   topFrameThick;  //              thickness
  std::vector<double>      topFrameZ;      //              z-positions
  std::string              sideFrameMat;   //Side frame    material
  double                   sideFrameThick; //              thickness
  std::vector<double>      sideFrameZ;     //              z-positions
  std::string              waferMat;       //Wafer         material
  double                   sideWidth;      //              widths on the side
  std::vector<std::string> waferRot;       //              rotation matrix
  std::string              activeMat;      //Sensitive     material
  double                   activeHeight;   //              height
  std::vector<double>      activeThick;    //              thickness
  std::string              activeRot;      //              Rotation matrix
  std::vector<double>      activeZ;        //              z-positions
  std::string              hybridMat;      //Hybrid        material
  double                   hybridHeight;   //              height
  double                   hybridWidth;    //              width
  double                   hybridThick;    //              thickness
  std::vector<double>      hybridZ;        //              z-positions
  std::string              pitchMat;       //Pitch adapter material
  double                   pitchHeight;    //              height
  double                   pitchThick;     //              thickness
  std::vector<double>      pitchZ;         //              z-positions
  std::string              pitchRot;       //              rotation matrix
  std::string              bridgeMat;      //Bridge        material
  double                   bridgeWidth;    //              width 
  double                   bridgeThick;    //              thickness
  double                   bridgeHeight;   //              height
  double                   bridgeSep;      //              separation

};

#endif
