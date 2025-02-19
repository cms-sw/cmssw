#ifndef RPCClusterSize_h
#define RPCClusterSize_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

class RPCClusterSize {

 public:

  
  //structure suitable for cluster size
  struct ClusterSizeItem {
    int dpid;
    float clusterSize;
  };
  
  
  RPCClusterSize(){}
  ~RPCClusterSize(){}
  

  std::vector<ClusterSizeItem>  const & getCls() const {return v_cls;}
  std::vector<ClusterSizeItem>  v_cls; 

};

#endif  //RPCClusterSize_h
