#ifndef CSCDQM_CSCStripClusterFitData_h
#define CSCDQM_CSCStripClusterFitData_h

#include <TObject.h>

namespace cscdqm {

class CSCStripClusterFitData {
 public:
  int channel()  {return channel_;}
  float height(int i)  {return height_[i];}
  float   bx() {return bx_;}
  
  //private:
  int channel_;
  float height_[16]; //or 16 for Cosmic Test
  float   bx_;
  CSCStripClusterFitData();
  virtual ~CSCStripClusterFitData();
//  ClassDef(CSCStripClusterFitData,1) //CSCStripClusterFitData
};

}

#endif
