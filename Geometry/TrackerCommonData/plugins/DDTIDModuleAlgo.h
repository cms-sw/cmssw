#ifndef DD_TIDModuleAlgo_h
#define DD_TIDModuleAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTIDModuleAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTIDModuleAlgo(); 
  ~DDTIDModuleAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  std::string              genMat;            //General material name
  int                      detectorN;         //Detector planes
  double                   moduleThick;       //Module thickness
  double                   detTilt;           //Tilt of stereo detector
  double                   fullHeight;        //Height 
  double                   dlTop;             //Width at top of wafer
  double                   dlBottom;          //Width at bottom of wafer
  double                   dlHybrid;          //Width at the hybrid end
  bool                     doComponents;      //Components to be made

  std::string              boxFrameName;      //Top frame     name
  std::string              boxFrameMat;       //              material
  double                   boxFrameHeight;    //              height  
  double                   boxFrameThick;     //              thickness
  double                   boxFrameWidth;     //              extra width
  double                   bottomFrameHeight; //Bottom of the frame
  double                   bottomFrameOver;   //              overlap
  double                   topFrameHeight;    //Top    of the frame
  double                   topFrameOver;      //              overlap
  std::vector<std::string> sideFrameName;     //Side frame    name
  std::string              sideFrameMat;      //              material
  double                   sideFrameWidth;    //              width
  double                   sideFrameThick;    //              thickness
  double                   sideFrameOver;     //              overlap (wrt wafer)
  std::vector<std::string> holeFrameName;     //Hole in the frame   name
  std::vector<std::string> holeFrameRot;      //              Rotation matrix

  std::vector<std::string> kaptonName;        //Kapton circuit name
  std::string              kaptonMat;         //               material
  //  double                   kaptonWidth;   //               width -> computed internally from sideFrameWidth and kaptonOver
  double                   kaptonThick;       //               thickness
  double                   kaptonOver;        //               overlap (wrt Wafer)
  std::vector<std::string> holeKaptonName;    //Hole in the kapton circuit name
  std::vector<std::string> holeKaptonRot;     //              Rotation matrix

  std::vector<std::string> waferName;         //Wafer         name
  std::string              waferMat;          //              material
  double                   sideWidthTop;      //              width on the side Top
  double                   sideWidthBottom;   //                                Bottom
  std::vector<std::string> activeName;        //Sensitive     name
  std::string              activeMat;         //              material
  double                   activeHeight;      //              height
  std::vector<double>      waferThick;        //              wafer thickness (active = wafer - backplane)
  std::string              activeRot;         //              Rotation matrix
  std::vector<double>      backplaneThick;    //              thickness
  std::string              hybridName;        //Hybrid        name
  std::string              hybridMat;         //              material
  double                   hybridHeight;      //              height
  double                   hybridWidth;       //              width
  double                   hybridThick;       //              thickness
  std::vector<std::string> pitchName;         //Pitch adapter name
  std::string              pitchMat;          //              material
  double                   pitchHeight;       //              height
  double                   pitchThick;        //              thickness
  double                   pitchStereoTol;        //              tolerance in dimensions of the stereo 
  std::string              coolName;         // Cool insert name
  std::string              coolMat;          //              material
  double                   coolHeight;       //              height
  double                   coolThick;        //              thickness
  double                   coolWidth;        //              width

};

#endif
