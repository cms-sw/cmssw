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
  /**
    \brief Constructor of the class
    \param nb Number of layers in the sector
    \param t Number of tracks needed to compute the parameters
  **/
  PrincipalTrackFitter(int nb, int t);
  ~PrincipalTrackFitter();

  void initialize();
  void mergePatterns();
  void mergeTracks();
  void fit();
  void fit(vector<Hit*> hits);
  TrackFitter* clone();

  /**
     \brief Add a track for the PCA parameters computing
     \param tracker Layer/ladder/module of each stub of the track [layer0,ladder0,module0,layer1,ladder1,module1]
     \param coord X/Y/Z of each stub of the track [X0,Y0,Z0,X1,Y1,Z1]
   **/
  void addTrackForPrincipal(int* tracker, double* coord);
  /**
     Checks that parameters for the PCA are known for every sub-sector
   **/
  bool hasPrincipalParams();
  /**
     \brief Force the computing of the PCA parameters, even if we have only few tracks in the sub-sectors
   **/
  void forcePrincipalParamsComputing();
  /**
     \brief Add a track for the MDF parameters computing
     \param tracker Layer/ladder/module of each stub of the track [layer0,ladder0,module0,layer1,ladder1,module1]
     \param coord X/Y/Z of each stub of the track [X0,Y0,Z0,X1,Y1,Z1]
   **/
  void addTrackForMultiDimFit(int* tracker, double* coord, double* pt);
  /**
     Checks that parameters for the MDF are known for every sub-sector
   **/
  bool hasMultiDimFitParams();
  /**
     \brief Force the computing of the MDF parameters, even if we have only few tracks in the sub-sectors
  **/
  void forceMultiDimFitParamsComputing();

};
#endif
