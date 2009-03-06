#ifndef CSCDQM_ClusterLocalMax_h
#define CSCDQM_ClusterLocalMax_h

#include <TObject.h>

namespace cscdqm {

  /**
   * @class ClusterLocalMax
   * @brief Local Maximum of the Cluster
   */
  class ClusterLocalMax {

    public:
      int Time;
      int Strip;

      ClusterLocalMax();
      virtual ~ClusterLocalMax();
      // ClassDef(ClusterLocalMax,1) //ClusterLocalMax

  };

}

#endif
