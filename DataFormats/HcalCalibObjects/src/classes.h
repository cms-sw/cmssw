#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"
#include "DataFormats/HcalCalibObjects/interface/HEDarkening.h"
// #include "DataFormats/HOCalibHit/interface/HOCalibVariableCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
   HEDarkening                                  hed;       
   HOCalibVariables                             rv1;
   std::vector<HOCalibVariables>                v1;
   edm::Wrapper<std::vector<HOCalibVariables> > wc1;
   //    HOCalibVariableCollection dummy0;
   //    edm::Wrapper< HOCalibVariableCollection > dummy1;
  };
}
