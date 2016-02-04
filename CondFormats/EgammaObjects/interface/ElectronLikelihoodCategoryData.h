#ifndef ElectronLikelihoodCategoryData_h
#define ElectronLikelihoodCategoryData_h
#include <string>

struct ElectronLikelihoodCategoryData {

  //! ECAL subdet: 0=Barrel, 1=Endcap
  int ecaldet;
  //! pT bin: 0= <15GeV, 1= >15GeV
  int ptbin;
  //! electron class 0=non-showering 1=showering
  int iclass;
  //! electron detailed class 
  //! 0=golden, 1=bigbrem, 2=narrow, 3=showering, 4=cracks
  int ifullclass;
  //! summary label: <variablename>_<species>_<ecalsubdet>_<ptbin>_<iclass>
  std::string label;

};

#endif // ElectronLikelihoodCategoryData_h
