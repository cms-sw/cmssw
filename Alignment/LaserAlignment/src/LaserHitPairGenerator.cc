/** \file LaserHitPairGenerator.cc
 *  
 *
 *  $Date: 2007/03/26 06:24:18 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/LaserHitPairGenerator.h"

#include "Alignment/LaserAlignment/interface/SeedLayerPairs.h"

LaserHitPairGenerator::LaserHitPairGenerator(SeedLayerPairs & layers, const edm::EventSetup & iSetup)
{
  std::vector<SeedLayerPairs::LayerPair> layerPairs = layers();
  
  for (std::vector<SeedLayerPairs::LayerPair>::const_iterator it = layerPairs.begin(); it != layerPairs.end(); it++)
    {
      add((*it).first, (*it).second, iSetup);
    }
}

LaserHitPairGenerator::~LaserHitPairGenerator()
{
  for (Container::const_iterator it = theGenerators.begin(); it != theGenerators.end(); it++)
    {
      delete (*it);
    }
}

void LaserHitPairGenerator::add(const LayerWithHits * inner, const LayerWithHits * outer, const edm::EventSetup & iSetup)
{
  theGenerators.push_back(new LaserHitPairGeneratorFromLayerPair(inner, outer, iSetup));
}

void LaserHitPairGenerator::hitPairs(const TrackingRegion & region, OrderedHitPairs & pairs, const edm::EventSetup & iSetup)
{
  for (Container::const_iterator it = theGenerators.begin(); it != theGenerators.end(); it++)
    {
      (**it).hitPairs(region, pairs, iSetup);
    }

  theLayerCache.clear();
}
