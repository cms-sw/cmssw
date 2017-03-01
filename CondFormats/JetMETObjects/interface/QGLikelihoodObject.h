#ifndef QGLikelihoodObject_h
#define QGLikelihoodObject_h

#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

/// Category structure: ranges associated with QGLikelihood histograms
struct QGLikelihoodCategory{
  float RhoMin, RhoMax, PtMin, PtMax, EtaMin, EtaMax;
  int QGIndex, VarIndex;
  COND_SERIALIZABLE;  
};

/// Parameters structure
struct QGLikelihoodParameters{
  float Rho, Pt, Eta;
  int QGIndex, VarIndex;
};

/// QGLikelihoodObject containing valid range and entries with category and histogram (mean is not used anymore, only for backward backward compatibility with older DB constructs)
struct QGLikelihoodObject{
  typedef PhysicsTools::Calibration::HistogramF Histogram;

  struct Entry{
    QGLikelihoodCategory category;
    Histogram histogram;
    float mean;
    COND_SERIALIZABLE;
  };

  QGLikelihoodCategory qgValidRange;
  std::vector<Entry> data;
  COND_SERIALIZABLE;
};

/// QGLikelihoodSystematicsObject containing the parameters for the systematic smearing
struct QGLikelihoodSystematicsObject{
  struct Entry{
    QGLikelihoodCategory systCategory;
    float a, b, lmin, lmax;
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
