#ifndef LaserAlignment_SeedLayerPairs_H
#define LaserAlignment_SeedLayerPairs_H

/** \class SeedLayerPairs
 *  interface to access pairs of layers; used for seedgenerator
 *
 *  $Date: 2007/05/10 12:00:32 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */
 
#include <vector>
class DetLayer;
class LayerWithHits;

class SeedLayerPairs {
public:

  typedef std::pair< const LayerWithHits*, const LayerWithHits*>        LayerPair;

  SeedLayerPairs() {};
  virtual ~SeedLayerPairs() {};
    virtual std::vector<LayerPair> operator()()= 0;
  

};

#endif
