#ifndef DD_TIDRingAlgo_h
#define DD_TIDRingAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTIDRingAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDTIDRingAlgo(); 
  ~DDTIDRingAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  std::string              idNameSpace;      //Namespace of this & ALL subparts
  std::vector<std::string> moduleName;       //Name of the module
  std::string              iccName;          //Name of the ICC

  int                      number;           //Number of copies
  double                   startAngle;       //Phi offset
  double                   rModule;          //Location of module in R 
  std::vector<double>      zModule;          //                   in Z
  double                   rICC;             //Location of ICC    in R 
  double                   sICC;             //Shift of ICC       per to R
  std::vector<double>      zICC;             //                   in Z
};

#endif
