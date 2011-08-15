#ifndef HcalAlgo_DDHCalForwardAlgo_h
#define HcalAlgo_DDHCalForwardAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDHCalForwardAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDHCalForwardAlgo(); //const std::string & name);
  virtual ~DDHCalForwardAlgo();

  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

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
