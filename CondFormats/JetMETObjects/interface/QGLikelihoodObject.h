#ifndef QGLikelihoodObject_h
#define QGLikelihoodObject_h

#include "CondFormats/BTauObjects/interface/CombinedSVCategoryData.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"



#include <vector>


struct QGLikelihoodCategory {
  float RhoVal, PtMin, PtMax;
  int EtaBin;  
  int QGIndex;
  int VarIndex;
};



struct QGLikelihoodObject
{
  typedef PhysicsTools::Calibration::HistogramF Histogram;

  struct Entry
  {
    QGLikelihoodCategory category;
    Histogram histogram;
  };

 std::vector<Entry> data;
  
};

#endif //QGLikelihoodObject_h
