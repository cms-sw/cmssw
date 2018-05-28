#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToROC.h"

HGCSiliconDetIdToROC::HGCSiliconDetIdToROC() {
  const int rocMax(6), cellRows(4), cellColumn(2);
  const int type[rocMax]               = {0,0,1,1,2,2};
  const int start1[rocMax][cellColumn] = {{3,2},{1,0},{1,2},{3,4},{7,6},{5,4}};
  const int start2[rocMax]             = {0,0,0,0,4,4};

  // key = ROC #; value = vector of trigger cell U.V
  for (int roc=0; roc<rocMax; ++roc) {
    std::vector<std::pair<int,int> > cells;
    if (type[roc] == 1) {
      for (int i1=0; i1<cellColumn; ++i1) {
	int v = start2[roc];
	int u = start1[roc][i1];
	for (int i2=0; i2<cellRows; ++i2) {
	  cells.emplace_back(u,v);
	  ++u; ++v;
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
