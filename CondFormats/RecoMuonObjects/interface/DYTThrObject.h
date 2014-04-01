#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

class DYTThrObject {
 public:
  struct DytThrStruct {
    DetId  id;
    double thr;
  };
  DYTThrObject(){}
  ~DYTThrObject(){}
  std::vector<DytThrStruct> thrsVec;
};
