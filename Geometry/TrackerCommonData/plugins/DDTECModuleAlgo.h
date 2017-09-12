
#ifndef DD_TECModuleAlgo_h
#define DD_TECModuleAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTECModuleAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTECModuleAlgo(); 
  ~DDTECModuleAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  //this positions  toPos in mother
  void doPos( const DDLogicalPart& toPos,  const DDLogicalPart& mother, int copyNr,
	      double x, double y, double z, 
	      const std::string& rotName, DDCompactView& cpv);                     
  void doPos( DDLogicalPart toPos,
	      double x, double y, double z, 
	      std::string rotName, DDCompactView& cpv);                     
  //variables:
  double                   noOverlapShift;
  int                      ringNo;
  bool                     isStereo; 
  bool                     isRing6;
  double                   rPos;           //Position in R relativ to the center of the TEC ( this is the coord-sys of Tubs)
  double                   posCorrectionPhi; // the Phi position of the stereo Modules has to be corrected
  std::string              standardRot;    //Rotation that aligns the mother(Tub ) coordinate System with the components
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
  double                   topFrameTopWidth;//             Width at the top
  double                   topFrameBotWidth;//             Width at the bottom
  double                   topFrame2Width;  //  Stereo:2ndPart   Width
  double                   topFrame2LHeight;//             left  height
  double                   topFrame2RHeight;//             right height
  double                   topFrameZ;      //              z-positions
  std::string              sideFrameMat;   //Side frame    material
  double                   sideFrameThick; //              thickness
  double                   sideFrameLWidth; //    Left     Width (for stereo modules upper one)
  double                   sideFrameLWidthLow;//           Width (only for stereo modules: lower Width)
  double                   sideFrameLHeight;//             Height
  double                   sideFrameLtheta;//              angle of the trapezoid shift
  double                   sideFrameRWidth; //    Right    Width (for stereo modules upper one)
  double                   sideFrameRWidthLow;//           Width (only for stereo modules: lower Width)
  double                   sideFrameRHeight;//             Height
  double                   sideFrameRtheta;//              angle of the trapezoid shift
  std::vector<double>      siFrSuppBoxWidth;//    Supp.Box Width
  std::vector<double>      siFrSuppBoxHeight;//            Height
  std::vector<double>      siFrSuppBoxYPos;//              y-position of the supplies box (with HV an thermal sensor...)
  double                   sideFrameZ;     //              z-positions
  double                   siFrSuppBoxThick;//             thickness
  std::string              siFrSuppBoxMat; //              material
  std::string              waferMat;       //Wafer         material
  double                   waferPosition;  //              position of the wafer (was formaly done by adjusting topFrameHeigt)
  double                   sideWidthTop;   //              widths on the side Top
  double                   sideWidthBottom;//                                 Bottom
  std::string              waferRot;       //              rotation matrix
  std::string              activeMat;      //Sensitive     material
  double                   activeHeight;   //              height
  double                   waferThick;     //              wafer thickness (active = wafer - backplane)
  std::string              activeRot;      //              Rotation matrix
  double                   activeZ;        //              z-positions
  double                   backplaneThick; //              thickness
  double                   inactiveDy;     //InactiveStrip  Hight of ( rings > 3)
  double                   inactivePos;    //               y-Position 
  std::string              inactiveMat;    //               material
  std::string              hybridMat;      //Hybrid        material
  double                   hybridHeight;   //              height
  double                   hybridWidth;    //              width
  double                   hybridThick;    //              thickness
  double                   hybridZ;        //              z-positions
  std::string              pitchMat;       //Pitch adapter material
  double                   pitchWidth;     //              width
  double                   pitchHeight;    //              height
  double                   pitchThick;     //              thickness
  double                   pitchZ;         //              z-positions
  std::string              pitchRot;       //              rotation matrix
  std::string              bridgeMat;      //Bridge        material
  double                   bridgeWidth;    //              width 
  double                   bridgeThick;    //              thickness
  double                   bridgeHeight;   //              height
  double                   bridgeSep;      //              separation
  std::vector<double>      siReenforceHeight;// SiReenforcement Height
  std::vector<double>      siReenforceWidth;//             Width
  std::vector<double>      siReenforceYPos;//              Y - Position
  //  double                   siReenforceZPos;//              Z - Position done by the side frames Z Position an t
  double                   siReenforceThick;//             Thick
  std::string              siReenforceMat;  //             Materieal
  //double                   posCorrectionR;  //  Correct Positions of the Stereo Modules radial coordinate
};

#endif
