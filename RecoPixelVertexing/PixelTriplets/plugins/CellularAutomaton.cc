
#include "CellularAutomaton.h"

template <unsigned int numberOfLayers>
void CellularAutomaton<numberOfLayers>::createAndConnectCells (std::vector<const HitDoublets*> doublets, const SeedingLayerSetsHits::SeedingLayerSet& fourLayers, const TrackingRegion& region, const float thetaCut, const float phiCut)
{
  unsigned int cellId = 0;
  constexpr unsigned int numberOfLayerPairs =   numberOfLayers - 1;
  float ptmin = region.ptMin();
  float region_origin_x = region.origin().x();
  float region_origin_y = region.origin().y();
  float region_origin_radius = region.originRBound();
  unsigned int layerPairId = 0;
  auto innerLayerId = layerPairId;
  auto outerLayerId = innerLayerId + 1;
  auto & doubletLayerPairId = doublets[layerPairId];
  auto numberOfDoublets = doubletLayerPairId->size ();

  isOuterHitOfCell[outerLayerId].resize(fourLayers[outerLayerId].hits().size());

  theFoundCellsPerLayer[layerPairId].reserve (numberOfDoublets);
  for (unsigned int i = 0; i < numberOfDoublets; ++i)
  {
    theFoundCellsPerLayer[layerPairId].emplace_back (doubletLayerPairId, i,  cellId,  doubletLayerPairId->innerHitId(i), doubletLayerPairId->outerHitId(i));

    isOuterHitOfCell[outerLayerId][doubletLayerPairId->outerHitId(i)].push_back (&(theFoundCellsPerLayer[layerPairId][i]));
    cellId++;

  }

  for (layerPairId = 1; layerPairId < numberOfLayerPairs; ++layerPairId)
  {
	doubletLayerPairId = doublets[layerPairId];
    innerLayerId = layerPairId;
    outerLayerId = innerLayerId + 1;
    numberOfDoublets =doubletLayerPairId->size ();
    isOuterHitOfCell[outerLayerId].resize(fourLayers[outerLayerId].hits().size());

    theFoundCellsPerLayer[layerPairId].reserve (numberOfDoublets);
    for (unsigned int i = 0; i < numberOfDoublets; ++i)
    {
      theFoundCellsPerLayer[layerPairId].emplace_back (doublets[layerPairId], i, cellId, doubletLayerPairId->innerHitId(i), doubletLayerPairId->outerHitId(i));
      isOuterHitOfCell[outerLayerId][doubletLayerPairId->outerHitId(i)].push_back (&(theFoundCellsPerLayer[layerPairId][i]));
      cellId++;
      for (auto neigCell : isOuterHitOfCell[innerLayerId][doubletLayerPairId->innerHitId(i)])
      {
        theFoundCellsPerLayer[layerPairId][i].checkAlignmentAndTag (neigCell, ptmin, region_origin_x, region_origin_y, region_origin_radius, thetaCut, phiCut);
      }
    }
  }
}

template <unsigned int numberOfLayers>
void
CellularAutomaton<numberOfLayers>::evolve ()
{
  constexpr unsigned int numberOfIterations = numberOfLayers - 2;
  unsigned int numberOfCellsFound ;
  for (unsigned int iteration = 0; iteration < numberOfIterations - 1; ++iteration)
  {
    for (unsigned int innerLayerId = 0; innerLayerId < numberOfIterations - iteration ; ++innerLayerId)
    {

      for (auto& cell : theFoundCellsPerLayer[innerLayerId])
      {
        cell.evolve();
      }
    }

    for (unsigned int innerLayerId = 0; innerLayerId < numberOfLayers - iteration - 2; ++innerLayerId)
    {

      for (auto& cell : theFoundCellsPerLayer[innerLayerId])
      {
        cell.updateState();
      }
    }
  }

  //last iteration 
  numberOfCellsFound = theFoundCellsPerLayer[0].size();

  for (unsigned int cellId = 0; cellId < numberOfCellsFound; ++cellId)
  {
    theFoundCellsPerLayer[0][cellId].evolve();
  }

  for (auto& cell : theFoundCellsPerLayer[0])
  {
    cell.updateState();
    if (cell.isRootCell (numberOfLayers - 2 ))
    {
      theRootCells.push_back (&cell);
    }
  }
}

template <unsigned int numberOfLayers>
void
CellularAutomaton<numberOfLayers>::findNtuplets(std::vector<CACell::CAntuplet>& foundNtuplets,  const unsigned int minHitsPerNtuplet)
{
  std::vector<CACell*> tmpNtuplet;
  tmpNtuplet.reserve(numberOfLayers);

  for (CACell* root_cell : theRootCells)
  {
    tmpNtuplet.clear();
    tmpNtuplet.push_back(root_cell);
    root_cell->findNtuplets (foundNtuplets, tmpNtuplet, minHitsPerNtuplet);
  }

}

template class CellularAutomaton<4>;


