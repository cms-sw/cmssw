#ifndef DataFormats_ForwardDetId_HGCSiliconDetIdToROC_H
#define DataFormats_ForwardDetId_HGCSiliconDetIdToROC_H 1

#include <iostream>
#include <map>
#include <utility>
#include <vector>
#include <functional>

class HGCSiliconDetIdToROC {

public:

  /** This translated TriggerDetId to ROC and viceversa for HGCSilicon*/
  HGCSiliconDetIdToROC();
   
  int getROCNumber(int triggerCellU, int triggerCellV) const {
    auto itr = triggerIdToROC_.find(std::make_pair(triggerCellU,triggerCellV));
    return ((itr == triggerIdToROC_.end()) ? -1 : itr->second);
  }
  std::vector<std::pair<int,int> > getTriggerId(int roc) const {
    auto itr = triggerIdFromROC_.find(roc);
    if (itr != triggerIdFromROC_.end()) { 
      return itr-> second;
    } else {
      std::vector<std::pair<int,int> > list;
      return list;
    }
  }
  void print() const;
    
private:

  std::map<std::pair<int,int>,int>                triggerIdToROC_;
  std::map<int,std::vector<std::pair<int,int> > > triggerIdFromROC_;
};
#endif
