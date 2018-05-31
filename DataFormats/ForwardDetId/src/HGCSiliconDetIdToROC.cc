#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToROC.h"

HGCSiliconDetIdToROC::HGCSiliconDetIdToROC() {
  const int rocMax(6), cellRows(4), cellColumn(2);
  const int type[rocMax]               = {0,0,1,1,2,2};
  const int start1[rocMax][cellColumn] = {{3,2},{1,0},{1,2},{3,4},{7,6},{5,4}};
  const int start2[rocMax]             = {0,0,0,1,4,4};

  // key = ROC #; value = vector of trigger cell U.V
  for (int roc=0; roc<rocMax; ++roc) {
    std::vector<std::pair<int,int> > cells;
    if (type[roc] == 1) {
      for (int i1=0; i1<cellColumn; ++i1) {
	int u = start1[roc][i1];
	int v = start2[roc] + u/2;
	for (int i2=0; i2<cellRows; ++i2) {
	  cells.emplace_back(u,v);
	  ++u;
	}
      }
    } else if (type[roc] == 2) {
      for (int i1=0; i1<cellColumn; ++i1) {
	int v = start2[roc];
	int u = start1[roc][i1];
	for (int i2=0; i2<cellRows; ++i2) {
	  cells.emplace_back(u,v);
	  ++v;
	}
      }
    } else {
      for (int i1=0; i1<cellColumn; ++i1) {
	int u = start2[roc];
	int v = start1[roc][i1];
	for (int i2=0; i2<cellRows; ++i2) {
	  cells.emplace_back(u,v);
	  ++u; ++v;
	}
      }
    }
    triggerIdFromROC_[roc+1] = cells;
  }

  // key = trigger cell U,V; value = roc #
  for (auto const & itr : triggerIdFromROC_) {
    for (auto cell : itr.second)
      triggerIdToROC_[cell] = itr.first;
  }
}

int HGCSiliconDetIdToROC::getROCNumber(int triggerCellU, int triggerCellV,
				       int type) const {
  auto itr = triggerIdToROC_.find(std::make_pair(triggerCellU,triggerCellV));
  int  roc = ((itr == triggerIdToROC_.end()) ? -1 : 
	      ((type == 0) ? itr->second : (1+itr->second)/2));
  return roc;
}

std::vector<std::pair<int,int> > HGCSiliconDetIdToROC::getTriggerId(int roc, int type) const {
  
  std::vector<std::pair<int,int> > list;
  if (type == 0) {
    auto itr = triggerIdFromROC_.find(roc);
    return ((itr == triggerIdFromROC_.end()) ? list : itr->second);
  } else {
    for (int k=0; k<2; ++k) {
      int rocx = 2*roc + k - 1;
      auto itr = triggerIdFromROC_.find(rocx);
      if (itr != triggerIdFromROC_.end()) {
	for (auto cell : itr->second)
	  list.emplace_back(cell);
      }
    }
    return list;
  }
}

void HGCSiliconDetIdToROC::print() const {
 
  for (auto const & itr : triggerIdFromROC_) {
    std::cout << "ROC " << itr.first << " with " << (itr.second).size()
	      << " trigger cells:";
    for (auto cell : itr.second)
      std::cout << " (" << cell.first << "," << cell.second << ")";
    std::cout << std::endl;
  }
  for (auto const & itr : triggerIdToROC_)
    std::cout << "Trigger cell (" << (itr.first).first << "," 
	      << (itr.first).second << ") : ROC " << itr.second <<std::endl;
}
