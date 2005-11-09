#ifndef DD_TIDModulePosAlgo_h
#define DD_TIDModulePosAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTIDModulePosAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTIDModulePosAlgo(); 
  virtual ~DDTIDModulePosAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs);

  void execute();

private:

  int                      detectorN;         //Number of detectors
  double                   detTilt;           //Tilt of stereo detector
  double                   fullHeight;        //Height 
  std::string              topFrameName;      //Top frame Name
  double                   dlTop;             //Width at top of wafer
  double                   dlBottom;          //Width at bottom of wafer
  double                   dlHybrid;          //Width at the hybrid end
  double                   topFrameHeight;    //Top Frame     height
  std::vector<double>      topFrameZ;         //              z-positions
  double                   bottomFrameHeight; //Bottom of the frame
  double                   bottomFrameOver;   //              overlap
  std::string              sideFrameName;     //Side Frame    name
  std::vector<double>      sideFrameZ;        //              z-positions
  std::vector<std::string> waferName;         //Wafer         name
  std::vector<double>      waferZ;            //              z-positions
  std::vector<std::string> waferRot;          //              rotation matrix
  std::string              hybridName;        //Hybrid        name
  double                   hybridHeight;      //              height
  std::vector<double>      hybridZ;           //              z-positions
  std::vector<std::string> pitchName;         //Pitch adapter rotation matrix
  double                   pitchHeight;       //              height
  std::vector<double>      pitchZ;            //              z-positions
  std::vector<std::string> pitchRot;          //              rotation matrix
};

#endif
