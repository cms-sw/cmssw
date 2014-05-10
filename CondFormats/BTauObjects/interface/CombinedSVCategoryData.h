#ifndef CombinedSVCategoryData_H
#define CombinedSVCategoryData_H

#include "CondFormats/Serialization/interface/Serializable.h"

struct CombinedSVCategoryData {
  float JetEtaMax, JetEtaMin, JetPtMax, JetPtMin;
  int PartonType;
  int TaggingVariable;
  int VertexType;

  COND_SERIALIZABLE;
};

#endif
