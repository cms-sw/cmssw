#ifndef _PRINCIPALTRACKFITTER_H_
#define _PRINCIPALTRACKFITTER_H_

#include "TrackFitter.h"
#include "FitParams.h"
#include <iomanip>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

/**
   \brief Implementation of a Principal Components Analysis fitter
**/
class PrincipalTrackFitter:public TrackFitter{

 private:
  map<string, FitParams*> params;
  int threshold;

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const
    {
      ar << boost::serialization::base_object<TrackFitter>(*this);
      ar << threshold;
      ar << params;
    }
  
  template<class Archive> void load(Archive & ar, const unsigned int version)
    {
      ar >> boost::serialization::base_object<TrackFitter>(*this);
      ar >> threshold;
      ar >> params;
    }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

 public:
  PrincipalTrackFitter();
  PrincipalTrackFitter(int nb, int t);
  ~PrincipalTrackFitter();

  void initialize();
  void mergePatterns();
  void mergeTracks();
  void fit();
  void fit(vector<Hit*> hits);
  TrackFitter* clone();

  void addTrackForPrincipal(int* tracker, double* coord);
  bool hasPrincipalParams();
  void forcePrincipalParamsComputing();

  void addTrackForMultiDimFit(int* tracker, double* coord, double* pt);
  bool hasMultiDimFitParams();
  void forceMultiDimFitParamsComputing();

};
#endif
