#ifndef _PATTERN_H_
#define _PATTERN_H_

#include <iostream>
#include <sstream>
#include <bitset>
#include "SuperStrip.h"
#include "PatternLayer.h"
#include "Detector.h"

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

/**
   \brief Representation of a pattern, a Pattern contains one PatternLayer per layer.
**/

class Pattern{

 public:
 
  /**
     \brief Constructor
     \param nb The number of layers
  **/
  Pattern(int nb);
  /**
     \brief Default Constructor
  **/
  Pattern();
  /**
     \brief Copy constructor. The patternLayers are copied, the superstrips are copies of pointers (not real copies).
     \param p The pattern to copy
  **/
  Pattern(const Pattern& p);
  ~Pattern();
  /**
     \brief Set one of the PatternLayer
     \param layer The layer of the new PatternLayer (will be copied)
     \param strip The PatternLayer
  **/
  void setLayerStrip(int layer, PatternLayer* strip);
  /**
     \brief Retrieves the PatternLayer of the given layer
     \param layer The layer of the PatternLayer you want
     \return A pointer on the PatternLayer (this is not a copy)
  **/
  PatternLayer* getLayerStrip(int layer);
  /**
     \brief Retrieves the PatternLayer of the given layer
     \param layer The layer of the PatternLayer you want
     \return A pointer on the PatternLayer (this is not a copy)
  **/
  PatternLayer* operator[](int layer);
  /**
     \brief Create links (pointers) between the PaternLayers and the Detector structure
     \param d The detector structure on which we link the pattern
     \param sec The ladders in the sector
     \param modules The modules in the sector (one vector per ladder)
  **/
  void link(Detector& d, const vector< vector<int> >& sec, const vector<map<int, vector<int> > >& modules);
  /**
     \brief Reset the links between the pattern layers and the super strips
  **/
  void unlink();
  /**
     \brief Check if the pattern is active
     \param active_threshold Minimum number of hit super strips
     \return True if the pattern is active, false otherwise.
  **/
  bool isActive(int active_threshold);
  /**
     \brief Created a unique key for this pattern
  **/
  string getKey();
  /**
     \brief Copy constructor
     \return The number of layers for this Pattern
  **/
  int getNbLayers() const;
  /**
     \brief Get a vectors of pointers on the Hits contained in this pattern (those are no copies)
     \return A vector of Hits (must be used before the superstrips are cleared)
  **/
  vector<Hit*> getHits();
  /**
     \brief Get a vectors of pointers on the Hits contained in this pattern layer (those are no copies)
     \param layerPosition The index of the layer
     \return A vector of Hits (must be used before the superstrips are cleared)
  **/
  vector<Hit*> getHits(int layerPosition);

  /**
     \brief Check if hdp is fully included in the pattern
     \param hdp The pattern to check
     \return True if hdp is included in the pattern
   **/
  bool contains(Pattern* hdp);

  /**
     \brief Allows to display a Pattern as a string
  **/
  friend ostream& operator<<(ostream& out, const Pattern& s);

 private:
  int nb_layer;
  SuperStrip*** strips;
  char* nb_strips;
  vector<PatternLayer*> layer_strips;

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const{
    ar << nb_layer;
    ar << layer_strips;
  }
  
  template<class Archive> void load(Archive & ar, const unsigned int version){
    ar >> nb_layer;
    ar >> layer_strips; 
    strips=NULL;
    nb_strips=NULL;
  }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};
#endif
