#ifndef TauJetHelper_h
#define TauJetHelper_h
#include "JetMETCorrections/Utilities/interface/GetParameters.h"
#include <map>
#include <string>
#include <vector>


class TauJetHelper{

 public:
  TauJetHelper(){}
  ~TauJetHelper(){}
  TauJetHelper(JetParameters theParam)
  {
    theCalibrationType = getCalibrationType(theParam); 
  };
  
  std::string getCalibrationType(JetParameters);
  std::string getCalibrationType(){return theCalibrationType;}

 private:

  std::string theCalibrationType;
  
};
#endif
