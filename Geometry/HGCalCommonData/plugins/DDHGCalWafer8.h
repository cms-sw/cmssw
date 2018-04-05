#ifndef HGCalCommonData_DDHGCalWafer8_h
#define HGCalCommonData_DDHGCalWafer8_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHGCalWafer8 : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHGCalWafer8();
  ~DDHGCalWafer8() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:
  double                   waferSize_;  // Wafer size
  double                   waferT_;     // Wafer thickness
  double                   waferSepar_; // Sensor separation
  double                   mouseBite_;  // MouseBite radius
  int                      nCells_;     // Half number of cells along u-v axis
  int                      cellType_;   // Cell Type (0,1,2: Fine, Course 2/3)
  std::string              material_;   // Material name for module with gap
  std::vector<std::string> cellNames_;  // Name of the cells
  std::string              nameSpace_;  // Namespace to be used
};

#endif
