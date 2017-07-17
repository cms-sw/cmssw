#ifndef HGCalCommonData_DDHGCalWaferAlgo_h
#define HGCalCommonData_DDHGCalWaferAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHGCalWaferAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHGCalWaferAlgo();
  ~DDHGCalWaferAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:

  double                   cellSize;      //Cell Size
  int                      cellType;      //Type (1 fine; 2 coarse)
  std::vector<std::string> childNames;    //Names of children
  std::vector<int>         positionX;     //Position in X
  std::vector<int>         positionY;     //Position in Y
  std::vector<double>      angles;        //Rotation angle
  std::vector<int>         detectorType;  //Detector type
  std::string              idName;        //Name of the "parent" volume.  
  std::string              rotns;         //Namespace for rotation matrix
  std::string              idNameSpace;   //Namespace of this and ALL sub-parts
  DDName                   parentName;    //Parent name
};

#endif
