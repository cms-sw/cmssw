#ifndef ClusterLocalMax_h
#define ClusterLocalMax_h
#include <TObject.h>

namespace cscdqm {

class ClusterLocalMax {
 public:
  int Time;
  int Strip;
  ClusterLocalMax();
  virtual ~ClusterLocalMax();
//  ClassDef(ClusterLocalMax,1) //ClusterLocalMax

};

}

#endif
