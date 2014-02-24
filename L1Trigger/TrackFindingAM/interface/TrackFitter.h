#ifndef _TRACKFITTER_H_
#define _TRACKFITTER_H_

#include <vector>
#include "Track.h"
#include "Pattern.h"

#include <boost/serialization/split_member.hpp>

using namespace std;

/**
   \brief Generic class for a fitter. A fitter is associated with a Sector and is able to delete redundancy among patterns or tracks and compute tracks from patterns.
**/
class TrackFitter{

 protected:
  vector<Pattern*> patterns;
  vector<Track*> tracks;
  int nb_layers;
  double sec_phi;//used for sector rotation
  int sector_id;

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const{
    ar << nb_layers;
    ar << sec_phi;
  }
  
  template<class Archive> void load(Archive & ar, const unsigned int version){
    ar >> nb_layers;
    ar >> sec_phi;
    sector_id=0;
  }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

 public:
  /**
  \brief Default constructor   
  **/
  TrackFitter();
  /**
  \brief Constructor   
  \param nb Layers number
  **/
  TrackFitter(int nb);
  virtual ~TrackFitter();
  /**
     \brief Performs any initialization if needed.
  **/
  virtual void initialize()=0;
  /**
     \brief Add a pattern to the existing list.
     \param p The pattern to add
   **/
  void addPattern(Pattern* p);
  /**
     \brief Delete redondant patterns among the existing list.
   **/
  virtual void mergePatterns()=0;
  /**
     \brief Delete redondant tracks among the existing list.
   **/
  virtual void mergeTracks()=0;
  /**
     \brief Create tracks from the list of existing patterns. The tracks are stored inside the TrackFitter object.
   **/
  virtual void fit()=0;
  /**
     \brief Create tracks from a list of hits. The tracks are stored inside the TrackFitter object. If patterns have previously been added to the TrackFitter object, they are not used in this method.
     \param hits A list of hits to use.
   **/
  virtual void fit(vector<Hit*> hits)=0;
  /**
     \brief Create a copy of the existing object.
     \return A pointer on the copy : you will have to delete this object.
   **/
  virtual TrackFitter* clone()=0;
  /**
     \brief Delete all patterns and tracks from the current object.
  **/
  void clean();

  /**
     \brief Returns a copy of the patterns inside the TrackFitter object.
     \return A vector containing a copy of the patterns. You have to delete these objects.
  **/
  vector<Pattern*> getFilteredPatterns();
  /**
     \brief Returns a copy of the fitted tracks.
     \return A vector containing a copy of the fitted tracks. You have to delete these objects.
  **/
  vector<Track*> getTracks();

  void setPhiRotation(double rot);
  double getPhiRotation();

  /**
     \brief Set the sector used for the fitter
     \param id The ID of the sector (from 0 to ...)
  **/
  void setSectorID(int id);

};
#endif
