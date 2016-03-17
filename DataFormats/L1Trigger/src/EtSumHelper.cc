
#include "DataFormats/L1Trigger/interface/EtSumHelper.h"

using namespace l1t;


double EtSumHelper::MissingEt() const {
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kMissingEt) return it->et();
  }
  return -999.0;
}  

double EtSumHelper::MissingEtPhi() const {
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kMissingEt) return it->phi();
  }
  return -999.0;
}  

double EtSumHelper::MissingHt() const {
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kMissingHt) return it->et();
  }
  return -999.0;
}  

double EtSumHelper::MissingHtPhi() const {
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kMissingHt) return it->phi();
  }
  return -999.0;
}  

double EtSumHelper::TotalEt() const {
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kTotalEt) return it->et();
  }
  return -999.0;
}  

double EtSumHelper::TotalHt() const {
  for (auto it=sum_->begin(0); it!=sum_->end(0); it++){      
    if (it->getType() == EtSum::kTotalHt) return it->et();
  }
  return -999.0;
}  

