#ifndef QGLikelihoodObject_h
#define QGLikelihoodObject_h

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

/// Parameters structure
struct QGLikelihoodParameters{
  float Rho, Pt, Eta;
  int QGIndex, VarIndex;
};

/// QGLikelihoodObject containing valid range and entries with category, histogram and mean
struct QGLikelihoodObject{
  typedef PhysicsTools::Calibration::HistogramF Histogram;

  struct Entry{
    QGLikelihoodCategory category;
    Histogram histogram;

    COND_SERIALIZABLE;  
  };

  std::vector<Entry> data;

  COND_SERIALIZABLE;  
};

/// Test if parameters are compatible with category
inline bool operator==(const QGLikelihoodParameters& lhs, const QGLikelihoodCategory& rhs){
  if(lhs.QGIndex != rhs.QGIndex)	return false;
  if(lhs.VarIndex != rhs.VarIndex)	return false;
  if(lhs.Eta < rhs.EtaMin)		return false;
  if(lhs.Eta > rhs.EtaMax)		return false;
  if(lhs.Rho < rhs.RhoMin)		return false;
  if(lhs.Rho > rhs.RhoMax)		return false;
  if(lhs.Pt < rhs.PtMin)		return false;
  if(lhs.Pt > rhs.PtMax)		return false;
  return true;
}

#endif
