#ifndef HGCalCommonData_DDHGCalCell_h
#define HGCalCommonData_DDHGCalCell_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHGCalCell : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHGCalCell();
  ~DDHGCalCell() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:
  double                   waferSize_;             //Wafer Size
  double                   waferT_;                //Wafer Thickness
  double                   cellT_;                 //Cell Thickness
  int                      nCells_;                //Number of columns (8:12)
  int                      posSens_;               //Position depleted layer
  std::string              material_;              //Name of the material
  std::string              fullCN_, fullSensN_;    //Name of full cell
  std::vector<std::string> truncCN_, truncSensN_;  //Names of truncated cells
  std::vector<std::string> extenCN_, extenSensN_;  //Names of extended  cells
  std::vector<std::string> cornrCN_, cornrSensN_;  //Names of corner    cells
  std::string              nameSpace_;             //Namespace to be used
};

#endif
