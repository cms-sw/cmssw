#ifndef MCJetHelper_h
#define MCJetHelper_h
#include "JetMETCorrections/Utilities/interface/GetParameters.h"
#include <map>
#include <string>
#include <vector>

using namespace std;

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

  string theCalibrationType;
  
};
#endif
