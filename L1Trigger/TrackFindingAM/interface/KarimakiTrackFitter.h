#ifndef _KARIMAKITRACKFITTER_H_
#define _KARIMAKITRACKFITTER_H_

#include "TrackFitter.h"
#include <iomanip>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

/**
   \brief Implementation of a Karimaki fitter
**/
class KarimakiTrackFitter:public TrackFitter{

 private:

  void fitPattern(Pattern* p);

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const//const boost::serialization::version_type& version) const 
    {
      cout<<"sauvegarde du KarimakiTrackFitter"<<endl;
      ar << boost::serialization::base_object<TrackFitter>(*this);
      ar << sec_phi;
    }
  
  template<class Archive> void load(Archive & ar, const unsigned int version)
    {
      cout<<"chargement de KarimakiTrackFitter"<<endl;
      ar >> boost::serialization::base_object<TrackFitter>(*this);
      ar >> sec_phi;
    }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

 public:
  /**
     \brief Default constructor   
  **/
  KarimakiTrackFitter();
  /**
     \brief Constructor   
     \param nb Layers number
  **/  
  KarimakiTrackFitter(int nb);
  ~KarimakiTrackFitter();

  void initialize();
  void mergePatterns();
  void mergeTracks();
  void fit();
  void fit(vector<Hit*> hits);
  TrackFitter* clone();
};
#endif
