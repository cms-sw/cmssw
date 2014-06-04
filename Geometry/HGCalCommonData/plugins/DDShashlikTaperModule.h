#ifndef HGCalCommonData_DDShashlikTaperModule_h
#define HGCalCommonData_DDShashlikTaperModule_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDShashlikTaperModule : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDShashlikTaperModule(); 
  virtual ~DDShashlikTaperModule();
  
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
  std::string         absorbMat;      //Name of the absorber material
  std::string         absorbName;     //Base name for absorber layers
  double              absorbThick;    //Thickness of absorber layer
  double              widthFront;     //Width of the module in the front
  double              widthBack;      //Width of the module in the back
  double              moduleThick;    //Offset in z
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
