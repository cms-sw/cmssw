#ifndef RPCClusterSize_h
#define RPCClusterSize_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#include <iostream>

class RPCClusterSize {
public:
  //structure suitable for cluster size
  struct ClusterSizeItem {
    int dpid;
    float clusterSize;

    COND_SERIALIZABLE;
  };

  RPCClusterSize() {}
  ~RPCClusterSize() {}

  std::vector<ClusterSizeItem> const& getCls() const { return v_cls; }
  std::vector<ClusterSizeItem> v_cls;

  COND_SERIALIZABLE;
};

#endif  //RPCClusterSize_h
