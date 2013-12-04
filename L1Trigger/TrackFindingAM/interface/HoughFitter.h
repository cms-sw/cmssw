#ifndef _HOUGHFITTER_H_
#define _HOUGHFITTER_H_

#include "TrackFitter.h"
#include "HoughLocal.h"
#include <iomanip>
#include <set>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

/**
   \brief Implementation of a Hough fitter
**/
class HoughFitter:public TrackFitter{

 private:

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const
    {
      cout<<"sauvegarde du HoughFitter"<<endl;
      ar << boost::serialization::base_object<TrackFitter>(*this);
      ar << sec_phi;
    }
  
  template<class Archive> void load(Archive & ar, const unsigned int version)
    {
      cout<<"chargement de HoughFitter"<<endl;
      ar >> boost::serialization::base_object<TrackFitter>(*this);
      ar >> sec_phi;
    }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

 public:
  /**
     \brief Default constructor   
  **/
  HoughFitter();
  /**
     \brief Constructor   
     \param nb Layers number
  **/  
  HoughFitter(int nb);
  ~HoughFitter();

  void initialize();
  void mergePatterns();
  void mergeTracks();
  void fit();
  TrackFitter* clone();
};
#endif
