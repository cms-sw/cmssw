#ifndef DD_TECCoolAlgo_h
#define DD_TECCoolAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTECCoolAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTECCoolAlgo(); 
  ~DDTECCoolAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  std::string              idNameSpace;    //Namespace of this and ALL parts
  int                      startCopyNo;    //Start copy number
  double                   rPosition;      // Position of the Inserts in R
  std::vector<double>      phiPosition;    // Position of the Inserts in Phi
  std::vector<std::string> coolInsert;       //Name of cooling pieces
};

#endif
