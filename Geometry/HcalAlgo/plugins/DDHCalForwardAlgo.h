#ifndef HcalAlgo_DDHCalForwardAlgo_h
#define HcalAlgo_DDHCalForwardAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHCalForwardAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDHCalForwardAlgo(); //const std::string & name);
  ~DDHCalForwardAlgo() override;

  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  std::string              cellMat;                 //Cell material 
  double                   cellDx, cellDy, cellDz;  //Cell size
  double                   startY;                  //Starting Y for Cell
  std::vector<std::string> childName;               //Children name
  std::vector<int>         number;                  //Number of cells
  std::vector<int>         size;                    //Number of children
  std::vector<int>         type;                    //First child

  std::string              idNameSpace;     //Namespace for aLL sub-parts
};

#endif
