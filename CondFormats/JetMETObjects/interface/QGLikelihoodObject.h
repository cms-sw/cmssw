#ifndef QGLikelihoodObject_h
#define QGLikelihoodObject_h

#include "CondFormats/BTauObjects/interface/CombinedSVCategoryData.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>


struct QGLikelihoodCategory {
  float RhoVal, PtMin, PtMax;
  int EtaBin;  
  int QGIndex;
  int VarIndex;

  COND_SERIALIZABLE;  
};



struct QGLikelihoodObject
{
  typedef PhysicsTools::Calibration::HistogramF Histogram;

  struct Entry
  {
    QGLikelihoodCategory category;
    Histogram histogram;

    COND_SERIALIZABLE;  
  };

  std::vector<Entry> data;

  COND_SERIALIZABLE;  
};

#endif //QGLikelihoodObject_h
