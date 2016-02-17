
#include "DataFormats/L1Trigger/interface/EtSumHelper.h"

using namespace l1t;


double EtSumHelper::MissingEt(){
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kMissingEt) return it->et();
  }
  return -999.0;
}  

double EtSumHelper::MissingEtPhi(){
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kMissingEt) return it->hwPhi();
  }
  return -999.0;
}  

double EtSumHelper::MissingHt(){
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kMissingHt) return it->et();
  }
  return -999.0;
}  

double EtSumHelper::MissingHtPhi(){
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kMissingHt) return it->hwPhi();
  }
  return -999.0;
}  

double EtSumHelper::TotalEt(){
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kTotalEt) return it->et();
  }
  return -999.0;
}  

double EtSumHelper::TotalHt(){
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kTotalHt) return it->et();
  }
  return -999.0;
}  

