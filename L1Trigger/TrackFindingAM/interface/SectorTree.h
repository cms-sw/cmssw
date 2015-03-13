#ifndef _SECTORTREE_H_
#define _SECTORTREE_H_

#include <iostream>
#include <map>
#include <ctime>
#include <TTree.h>
#include <TH1F.h>
#include <TFile.h>
#include "Sector.h"
#include "CMSPatternLayer.h"
#include "PrincipalTrackFitter.h"
#include "KarimakiTrackFitter.h"
#include "HoughFitter.h"
#include "RetinaTrackFitter.h"

#ifdef IPNL_USE_CUDA
#include "gpu_struct.h"
#endif

#ifndef __APPLE__
BOOST_CLASS_EXPORT_KEY(CMSPatternLayer) 
BOOST_CLASS_EXPORT_KEY(PrincipalTrackFitter) 
#endif

using namespace std;

/**
   \brief Contains a list of sector

   This class allows to search a sector by giving one ladder per layer.
**/

class SectorTree{

 private:
  multimap<string, Sector*> sectors;
  bool mapNeedsUpdate;
  vector<Sector*> sector_list;
  /**
     \brief used to know the superstrip size used for the patterns contained in this sectorTree.
     This value is not used inside the class.
  **/
  int superStripSize;

  void updateSectorMap();

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const{
    ar << sector_list;
    ar << superStripSize;
  }
  
  template<class Archive> void load(Archive & ar, const unsigned int version){
    ar >> sector_list;
    ar >> superStripSize;
  }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  
 public:
  /**
     \brief Constructor
  **/
  SectorTree();
  /**
     \brief Destructor
  **/
  ~SectorTree();
  /**
     \brief Search for one sector
     \param ladders One ladder per layer : ie [5,6,6] if the particule went through ladders 5,6,6 on layers 0,1,2.
     \param modules One module per ladder : ie [1,2,3] if the particule went through modules 1,2,3 on layers 0,1,2.
     \return NULL if no sector was found or a pointer on the Sector (do NOT delete this pointer, it is not a copy). If several sectors match, the first one is returned (smaller ladders)
  **/
  Sector* getSector(vector<int> ladders, vector<int> modules);
 /**
     \brief Search for one sector
     \param h A hit
     \return NULL if the hit is not in any sector or a pointer on the Sector (do NOT delete this pointer, it is not a copy). If several sectors match, the first one is returned (smaller ladders)
  **/
  Sector* getSector(const Hit& h);
  /**
     \brief Add a sector to the SectorTree
     \param s The sector to add
  **/
  void addSector(Sector s);
  /**
     \brief Get all the sectors
     \return A vector containing all the Sectors. Do NOT delete those pointers.
  **/
  vector<Sector*> getAllSectors();
  /**
     \brief Gets the number of Low Def patterns in the tree
  **/
  int getLDPatternNumber();
  /**
     \brief Gets the number of Full Def patterns in the tree
  **/
  int getFDPatternNumber();
  /**
     \brief Replace all LD patterns with adapatative patterns. All FD patterns are removed.
     \param r The number of DC bits used between FD and LD
  **/
  void computeAdaptativePatterns(short r);
  /**
     Link all the patterns contained in each sector to the super strips contained in the Detector object
     \param d The Detector object
  **/
  void link(Detector& d);
#ifdef IPNL_USE_CUDA
  /**
     Link all the patterns in the patternBank region to the super strips contained in the deviceDetector object
     \param d The Detector object on the device
     \param p The patterns bank on the device
  **/
  void linkCuda(patternBank* p, deviceDetector* d);
#endif
  /**
     \brief Get the active patterns in each sector
     \param active_threshold The minimum number of hit super strips to activate the pattern
     \return A vector containing pointers on copies of the sectors, each sectors containing its active patterns
  **/
  vector<Sector*> getActivePatternsPerSector(int active_threshold);

  /**
     \brief Get the active patterns in each sector
     \param max_nb_missing_hit The maximum number of non active layers to activate the pattern
     \param active_threshold The minimum number of hit super strips to activate the pattern
     \return A vector containing pointers on copies of the sectors, each sectors containing its active patterns
  **/
  vector<Sector*> getActivePatternsPerSectorUsingMissingHit(int max_nb_missing_hit, int active_threshold);

  /**
     \brief Retrieve the superstrip size used for the patterns inside the SectorTree
     \return -1 if not specified, the superStrip size otherwise.
  **/
  int getSuperStripSize();

  /**
     \brief Set the size of the superStrip used in the patterns stored in the SectorTree
     \param s The superStrip size (should be greater than 0)
  **/
  void setSuperStripSize(int s);

  /**
     \brief Get the number of sectors in the SectorTree
  **/
  int getNbSectors();

};
#endif
