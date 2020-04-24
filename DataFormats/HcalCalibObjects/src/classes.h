#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"
// #include "DataFormats/HOCalibHit/interface/HOCalibVariableCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace DataFormats_HcalCalibObjects {
  struct dictionary {
   HOCalibVariables                             rv1;
   std::vector<HOCalibVariables>                v1;
   edm::Wrapper<std::vector<HOCalibVariables> > wc1;
   //    HOCalibVariableCollection dummy0;
   //    edm::Wrapper< HOCalibVariableCollection > dummy1;
  };
}
