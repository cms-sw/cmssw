#ifndef MCJetHelper_h
#define MCJetHelper_h
#include "JetMETCorrections/Utilities/interface/GetParameters.h"
#include <map>
#include <string>
#include <vector>


class MCJetHelper{

 public:
  MCJetHelper(){}
  ~MCJetHelper(){}
  MCJetHelper(JetParameters theParam)
  {
    theCalibrationType = getCalibrationType(theParam); 
  };
  
  std::string getCalibrationType(JetParameters);
  std::string getCalibrationType(){return theCalibrationType;}

 private:

  std::string theCalibrationType;
  
};
#endif
