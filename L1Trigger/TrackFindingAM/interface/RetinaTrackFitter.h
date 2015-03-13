#ifndef _RETINATRACKFITTER_H_
#define _RETINATRACKFITTER_H_

#include "TrackFitter.h"
#include "Retina.h"

#include <set>
#include <map>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

/**
   \brief Implementation of a track fitter based on an artificial retina
**/
class RetinaTrackFitter:public TrackFitter{

 private:

  unsigned int verboseLevel; 
  unsigned int event_counter;
  unsigned int road_id;

  // This contains the retina configuration parameters,
  // it is filled in initialize():
  std::map <const char*,double> config[3];

  // This is used to rotate the trigger tower into the first quadrant:
  const static double rot_angle[8];

  void initialize();
  void rotateHits(vector<Hit_t*> hits, double angle);
  void confTrans(vector<Hit_t*> hits);

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const
    {
      ar << boost::serialization::base_object<TrackFitter>(*this);
      ar << sec_phi;
    }
  
  template<class Archive> void load(Archive & ar, const unsigned int version)
    {
      ar >> boost::serialization::base_object<TrackFitter>(*this);
      ar >> sec_phi;
    }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

 public:

  /**
     \brief Default constructor   
  **/
  RetinaTrackFitter();

  /**
     \brief Constructor   
     \param nb Layers number
  **/  
  RetinaTrackFitter(int nb);

  ~RetinaTrackFitter();

  void mergePatterns();
  void mergeTracks();
  void fit();
  void fit(std::vector<Hit*> hits);
  TrackFitter* clone();

  void setEventCounter(unsigned int event_counter_){
    event_counter = event_counter_;
  };
  void setRoadID(unsigned int road_id_){
    road_id = road_id_;
  };
  void setVerboseLevel(unsigned int verboseLevel_){
    verboseLevel = verboseLevel_;
  }; 

};
#endif
