#ifndef DD_TECOptoHybAlgo_h
#define DD_TECOptoHybAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTECOptoHybAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTECOptoHybAlgo(); 
  ~DDTECOptoHybAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  std::string              idNameSpace;    //Namespace of this and ALL parts
  std::string              childName;      //Child name
  double                   rpos;           //r Position
  double                   zpos;           //Z position of the OptoHybrid
  double                   optoHeight;     // Height of the OptoHybrid
  double                   optoWidth;     // Width of the OptoHybrid
  int                      startCopyNo;    //Start copy number
  std::vector<double>      angles;         //Angular position of Hybrid
};

#endif
