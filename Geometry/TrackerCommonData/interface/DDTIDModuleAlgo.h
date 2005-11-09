#ifndef DD_TIDModuleAlgo_h
#define DD_TIDModuleAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTIDModuleAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTIDModuleAlgo(); 
  virtual ~DDTIDModuleAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs);

  void execute();

private:

  std::string              genMat;            //General material name
  int                      detectorN;         //Detector planes
  double                   moduleThick;       //Module thickness
  double                   detTilt;           //Tilt of stereo detector
  double                   fullHeight;        //Height 
  double                   dlTop;             //Width at top of wafer
  double                   dlBottom;          //Width at bottom of wafer
  double                   dlHybrid;          //Width at the hybrid end
  std::string              topFrameName;      //Top frame     name
  std::string              topFrameMat;       //              material
  double                   topFrameHeight;    //              height
  double                   topFrameThick;     //              thickness
  double                   topFrameWidth;     //              extra width
  double                   bottomFrameHeight; //Bottom of the frame
  double                   bottomFrameOver;   //              overlap
  std::string              sideFrameName;     //Side frame    name
  std::string              sideFrameMat;      //              material
  double                   sideFrameWidth;    //              width
  double                   sideFrameThick;    //              thickness
  double                   sideFrameOver;     //              overlap
  std::string              dumyFrameName;     //Dummy frame   name
  std::vector<std::string> waferName;         //Wafer         name
  std::string              waferMat;          //              material
  double                   sideWidth;         //              width on the side
  std::vector<std::string> activeName;        //Sensitive     name
  std::string              activeMat;         //              material
  double                   activeHeight;      //              height
  std::vector<double>      activeThick;       //              thickness
  std::string              activeRot;         //              Rotation matrix
  std::string              hybridName;        //Hybrid        name
  std::string              hybridMat;         //              material
  double                   hybridHeight;      //              height
  double                   hybridWidth;       //              width
  double                   hybridThick;       //              thickness
  std::vector<std::string> pitchName;         //Pitch adapter name
  std::string              pitchMat;          //              material
  double                   pitchHeight;       //              height
  double                   pitchThick;        //              thickness
};

#endif
