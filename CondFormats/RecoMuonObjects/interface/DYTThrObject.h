#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

class DYTThrObject {
 public:
  struct dytThrStruct {
    DetId  id;
    double thr;
  };
  DYTThrObject(){}
  ~DYTThrObject(){}
  std::vector<dytThrStruct> thrsVec;
};
