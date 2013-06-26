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
  void execute(DDCompactView& cpv);

private:

  int                      detectorN;         //Number of detectors
  double                   detTilt;           //Tilt of stereo detector
  double                   fullHeight;        //Height 
  std::string              boxFrameName;      //Top frame Name
  double                   boxFrameHeight;    //          height 
  double                   boxFrameWidth;     //          width
  double                   dlTop;             //Width at top of wafer
  double                   dlBottom;          //Width at bottom of wafer
  double                   dlHybrid;          //Width at the hybrid end
  std::vector<double>      boxFrameZ;         //              z-positions
  double                   bottomFrameHeight; //Bottom of the frame
  double                   bottomFrameOver;   //              overlap
  double                   topFrameHeight;    //Top    of the frame
  double                   topFrameOver;      //              overlap 

  std::vector<std::string> sideFrameName;     //Side Frame    name
  std::vector<double>      sideFrameZ;        //              z-positions
  std::vector<std::string> sideFrameRot;      //              rotation matrix (required for correct positiong of the hole in the StereoR)
  double                   sideFrameWidth;    //              width
  double                   sideFrameOver;     //              overlap (wrt wafer)

  std::vector<std::string> kaptonName;         //Kapton Circuit    name
  std::vector<double>      kaptonZ;           //              z-positions
  std::vector<std::string> kaptonRot;         //              rotation matrix (required for correct positiong of the hole in the StereoR)
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
  std::string              coolName;        //Cool Insert   name
  double                   coolHeight;      //              height
  double                   coolZ;           //              z-position
  double                   coolWidth;       //              width
  std::vector<double>      coolRadShift;    //              


  bool                     doSpacers;      //Spacers (alumina) to be made (Should be "Yes" for DS modules only)
  std::string botSpacersName;   // Spacers at the "bottom" of the module
  double botSpacersHeight;      //
  double botSpacersZ;           //              z-position
  std::string sidSpacersName;   //Spacers at the "sides" of the module
  double sidSpacersHeight;   
  double sidSpacersZ;           //              z-position
  double sidSpacersWidth;       //              width
  double sidSpacersRadShift;    //              
};

#endif
