#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include "Layer.h"
#include <vector>


/**
   \brief Representation of a detector.
   A detector is made of layers->ladders->modules->segments->superStrips
   This class is used for the pattern recognition step. A Detector is an organized set of superstrip objects. At the beginning of the program each pattern layer (we have nb_layers pattern layers per pattern) will be linked to its corresponding superstrip object in the Detector (a pointer to this object is added to the pattern layer). Then for each event :
     - For each stub we look for the containing superstrip in the Detector and set it to active
     - For each pattern we count the total number of active pattern layer (pattern layer with an active superstrip) -> if the number is greater than the threshold the pattern is active.
     - we reset the detector to 0
**/
class Detector{

 private:
  vector<Layer*> layers;
  vector<int> layerNumber;
  SuperStrip* dump;//used for fake superstrips in the patterns

  int getLayerPosition(int pos);
  
 public:
  /**
     \brief Constructor.
     Creates an empty detector (no layer).
  **/
  Detector();
  /**
     \brief Add a new layer to the detector
     \param lNum The id of the layer (ie 10 for the outermost layer)
     \param nbLad Number of ladders in the layer
     \param nbMod Number of modules for each ladder
     \param segmentSize Number of strips in each segment
     \param sstripSize Size of a superStrip (number of strips)
  **/
  void addLayer(int lNum, int nbLad, int nbMod, int segmentSize, int sstripSize);
  /**
     \brief Destructor
     Delete all layers.
  **/
  ~Detector();
  /**
     \brief Get one of the detector's layers
     \param pos The id of the layer (ie 10 for the outermost layer)
     \return A pointer on the Layer object (not a copy), NULL if nothing found.
  **/
  Layer* getLayer(int pos);
  /**
     \brief Get one of the detector's layers
     \param pos The position of the layer in the detector(layer num 0, layer num 1, ...)
     \return A pointer on the Layer object (not a copy), NULL if nothing found.
  **/
  Layer* getLayerFromAbsolutePosition(int pos);
  /**
     \brief Actives the super strip corresponding tho the given hit in the detector
     \param h A hit
  **/
  void receiveHit(const Hit& h);
  /**
     \brief Desactivates all superstrips in the detector
  **/
  void clear();
  /**
     Get the current number of layes in the detector
     \return The number of layers
  **/
  int getNbLayers();
  /**
     Get the dump superstrip (used for fake superstrips in patterns)
  **/
  SuperStrip* getDump();

};
#endif
