#ifndef HcalAlgo_DDHCalXtalAlgo_h
#define HcalAlgo_DDHCalXtalAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHCalXtalAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDHCalXtalAlgo();
  ~DDHCalXtalAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double radius;                   //Pointing distance from front surface
  double offset;                   //Offset along Z
  double dx;                       //Half size along x
  double dz;                       //Half size along z
  double angwidth;                 //Angular width
  int iaxis;                       //Axis of rotation
  std::vector<std::string> names;  //Names for rotation matrices

  std::string idNameSpace;  //Namespace of this and ALL sub-parts
  std::string idName;       //Children name
};

#endif
