#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include "CondFormats/Serialization/interface/Serializable.h"

class DYTThrObject {
public:
  struct DytThrStruct {
    DetId id;
    double thr;

    COND_SERIALIZABLE;
  };

  DYTThrObject() {}
  ~DYTThrObject() {}
  std::vector<DytThrStruct> thrsVec;

  COND_SERIALIZABLE;
};
