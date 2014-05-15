#ifndef _SECTOR_H_
#define _SECTOR_H_

#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <TChain.h>
#include <TFile.h>
#include <boost/serialization/map.hpp>

#include "PatternTree.h"
#include "TrackFitter.h"
#include "CMSPatternLayer.h"

using namespace std;

/**
   \brief A part of the detector (1 to N ladders per layer)
   A Sector is a part of the detector (1 to N ladders per layer)
   The sector object can also contain the list of associated Patterns
**/

class Sector{

 private:

  map<int, map<int, vector<int> > > m_modules; // map<layer_id, map<ladder_id, vector<module_id>>>
  map<int, vector<int> > m_ladders; // map<layer_id, vector<ladder_id>> : needed to keep the order of the ladders (order not kept in maps)
  PatternTree* patterns;
  TrackFitter* fitter;
  int officialID;
  void getRecKeys(vector< vector<int> > &v, int level, string temp, vector<string> &res);

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const{
    ar << m_modules;
    ar << m_ladders;
    ar << patterns;
    ar << officialID;
    int exists;
    if(fitter==NULL){
      exists = 0;
      ar << exists;
    }
    else{
      exists = 1;
      ar << exists;
      ar << fitter;
    }
  }
  
  template<class Archive> void load(Archive & ar, const unsigned int version){
    ar >> m_modules;
    ar >> m_ladders;
    ar >> patterns;
    if(version>0)// Add support of the Sector ID coming from the TKLayout
      ar >> officialID;
    else
      officialID=-1;
    int exists;
    ar >> exists;
    if(exists==1){
      ar >> fitter;
    }
  }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

 public:
  /**
     \brief Default constructor (3 layers)
  **/
  Sector();
  /**
     \brief Copy constructor. Be carreful : the list of patterns is not copied!
     \param s The sector to be copied.
  **/
  Sector(const Sector& s);

  /**
     \brief Constructor. 
     \param layersID The list of layers in the sector (the layers tracker IDs : from 1 to 10)
  **/
  Sector(vector<int> layersID);

  /**
     \brief Destructor : destroy the list of patterns
  **/
  ~Sector();

  /**
     \brief Be carreful : the list of patterns is not copied!
  **/
  Sector& operator=(Sector& s);

  /**
     Check if the 2 sectors have the same layers and ladders
  **/
  bool operator==(Sector& s);

  /**
     \brief Add a layer to the sector
     \param layer The layer number
  **/
  void addLayer(int layer);

  /**
     \brief Add ladders to the sector
     \param layer The layer ID
     \param firstLadder The starting ladder ID
     \param nbLadders The number of consecutive ladders on this layer
  **/
  void addLadders(int layer, int firstLadder, int nbLadders);

 /**
     \brief Add modules to the ladder
     \param layer The layer ID
     \param ladder The ladder ID
     \param firstModule The starting module ID
     \param nbModules The number of consecutive modules on this ladder
  **/
  void addModules(int layer, int ladder, int firstModule, int nbModules);

  /**
     \brief Get the ladder code in the sector (to be used in the patternLayer). This is the position of the ladder in the layer
     \param layer The layer ID
     \param ladder The ladder ID
     \return The position of the ladder in the layer which is the code to use in the pattern
  **/
  int getLadderCode(int layer, int ladder);

  /**
     \brief Get the module code in the sector (to be used in the patternLayer). This is the position of the module in the ladder
     \param layer The layer ID
     \param ladder The ladder ID
     \param module The module ID
     \return The position of the module in the ladder, which is the code to use in the pattern
  **/
  int getModuleCode(int layer, int ladder, int module);

  /**
     \brief Get the number of layers in the sector
     \return The number of layers
  **/
  int getNbLayers();

  /**
     \brief Get all the possible paths in this sector (1 ladder per layer)
     \return A list of strings. Each string uses 2 digits per ladder (01 02 03 ...). The format is 020406 for ladders 2,4 and 6 respectively on layers 0,1 and 2
  **/
  vector<string> getKeys();

  /**
     \brief Get a string representation of the sector
     \return A string representating the sector (ex : "1 7-8 8-9")
  **/
  string getIDString();

  /**
     \brief Get a unique key to identify the sector
     \return The int key of the sector
  **/
  int getKey();

  /**
     \brief Get the TKLayout ID of the sector
     \return The TKLayout ID of the sector, -1 if not known
  **/
  int getOfficialID();

  /**
     \brief Set the Sector official ID (id must be > -1)
  **/
  void setOfficialID(int id);

  /**
     \brief Does the sector contains the hit?
     \return True if the hit is in the sector, false otherwise
  **/
  bool contains(const Hit& h);


  /**
     \brief Get the ladders on the given layer
     \param l The layer index (not the ID!)
     \return A vector containing the ordered ladders id.
  **/
  vector<int> getLadders(int l) const;

  /**
     \brief Get the modules on the given layer and ladder
     \param lay The layer index (not the ID!)
     \param l The ladder ID
     \return A vector containing the ordered module IDs.
  **/
  vector<int> getModules(int lay, int l) const;


  /**
     \brief Get the layer ID
     \param i The order of the layer in the sector
     \return The layer ID
  **/
  int getLayerID(int i) const;

 /**
     \brief Get the layer index
     \param i The ID of the layer
     \return The index of the layer
  **/
  int getLayerIndex(int i) const;

  /**
     \brief Get list of patterns
     \return A pointer on the PatternTree structure
  **/
  PatternTree* getPatternTree();

  /**
     \brief Allows to display a sector as a string
  **/
  friend ostream& operator<<(ostream& out, const Sector& s);

  /**
     \brief  Returns the number of Low Definition Patterns
     \return The number of LD patterns contained in the sector
  **/
  int getLDPatternNumber();

  /**
     \brief  Returns the number of Full Definition Patterns
     \return The number of FD patterns contained in the sector
  **/
  int getFDPatternNumber();
  /**
     \brief Replace all LD patterns with adapatative patterns. All FD patterns are removed.
     \param r The number of DC bits used between FD and LD
  **/
  void computeAdaptativePatterns(short r);
  /**
     Link all the patterns contained in the sector to the super strips contained in the Detector object
     \param d The Detector object
  **/
  void link(Detector& d);
  /**
     \brief Get the active patterns of the sector
     \param active_threshold The minimum number of hit super strips to activate the pattern
     \return A vector containing pointers on copies of the patterns
  **/
  vector<GradedPattern*> getActivePatterns(int active_threshold);

  /**
     \brief Associates a fitter to the Sector
     \param f The TrackFitter object which will be used for this sector
  **/
  void setFitter(TrackFitter* f);

  /**
     \brief Get the fitter for this Sector
     \return f The TrackFitter object which will be used for this sector. Do NOT delete this object!
  **/
  TrackFitter* getFitter();

   /**
     \brief Update the Phi Rotation value of the current fitter
   **/
  void updateFitterPhiRotation();

  static map< int, vector<int> > readConfig(string name);
};
BOOST_CLASS_VERSION(Sector, 1)
#endif
