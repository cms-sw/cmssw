#ifndef HGCalCommonData_DDShashlikModule_h
#define HGCalCommonData_DDShashlikModule_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDShashlikModule : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDShashlikModule(); 
  virtual ~DDShashlikModule();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string         activeMat;      //Name of the active material
  std::string         activeName;     //Base name for active layers
  int                 activeLayers;   //Number of active layers
  double              activeThick;    //Thickness of active layer
  double              absorbThick;    //Thickness of absorber layer
  double              widthFront;     //Width of the module in the front
  double              widthBack;      //Width of the module in the back
  double              moduleThick;    //Offset in z
  double              moduleTaperAngle; //Taper angle of individual module
  double              holeR;          //Radius of the hole
  std::string         fibreMat;       //Fibre Material
  std::string         fibreName;      //Fibre Name
  std::vector<double> holeX;          //x-position of the holes
  std::vector<double> holeY;          //y-position of the holes
  std::string         calibFibreName; //Calibration Fibre Name
  std::vector<double> calibFibrePars; //Parameters for calibration fibre

  std::string         idNameSpace;    //Namespace of this and ALL sub-parts
};

#endif
