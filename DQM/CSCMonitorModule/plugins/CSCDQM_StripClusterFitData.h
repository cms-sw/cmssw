#ifndef CSCDQM_StripClusterFitData_h
#define CSCDQM_StripClusterFitData_h

#include <TObject.h>

namespace cscdqm {

  /**
   * @class StripClusterFitData
   * @brief Strip Cluster Fit Data Object
   */
class StripClusterFitData {
 public:
  int channel()  {return channel_;}
  float height(int i)  {return height_[i];}
  float   bx() {return bx_;}
  
  //private:
  int channel_;
  float height_[16]; //or 16 for Cosmic Test
  float   bx_;
  StripClusterFitData();
  virtual ~StripClusterFitData();
//  ClassDef(StripClusterFitData,1) //StripClusterFitData
};

}

#endif
