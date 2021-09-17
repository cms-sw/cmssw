#ifndef RPCDQMObject_h
#define RPCDQMObject_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#include <iostream>

class RPCDQMObject {
public:
  int dqmv;
  int run;

  //structure suitable for cluster size
  struct DQMObjectItem {
    int dpid;
    float clusterSize;
    float bx;
    float bxrms;
    float efficiency;
    float numdigi;
    float numcluster;
    int status;
    float weight;

    COND_SERIALIZABLE;
  };

  RPCDQMObject() {}
  ~RPCDQMObject() {}

  RPCDQMObject* Fake_RPCDQMObject();

  std::vector<DQMObjectItem> const& getCls() const { return v_cls; }
  std::vector<DQMObjectItem> v_cls;

private:
  COND_SERIALIZABLE;
};

#endif  //RPCDQMObject_h
