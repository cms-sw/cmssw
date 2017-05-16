#ifndef HGCalCommonData_DDHGCalWafer_h
#define HGCalCommonData_DDHGCalWafer_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDHGCalWafer : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHGCalWafer();
  virtual ~DDHGCalWafer();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);
  void execute(DDCompactView& cpv);

private:

  double                   waferSize_;    //Wafer Size
  int                      cellType_;     //Type (1 fine; 2 coarse)
  int                      nColumns_;     //Maximum number of columns
  int                      nBottomY_;     //Index of cell position of bottom row
  std::vector<std::string> childNames_;   //Names of children
  std::vector<int>         nCellsRow_;    //Number of cells in a row
  std::vector<int>         angleEdges_;   //Rotation angles to be used for edges
  std::vector<int>         detectorType_; //Detector type of edge cells
  std::string              idNameSpace_;  //Namespace of this and ALL sub-parts
  DDName                   parentName_;   //Parent name
};

#endif
