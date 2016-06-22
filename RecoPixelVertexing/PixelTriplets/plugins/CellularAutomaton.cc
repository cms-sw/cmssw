
#include "CellularAutomaton.h"

template <unsigned int numberOfLayers>
void CellularAutomaton<numberOfLayers>::create_and_connect_cells (std::vector<const HitDoublets*> doublets, const SeedingLayerSetsHits::SeedingLayerSet& fourLayers, const TrackingRegion& region)
{
  unsigned int cellId = 0;
  constexpr unsigned int numberOfLayerPairs =   numberOfLayers - 1;
  float ptmin = region.ptMin();
  unsigned int layerPairId = 0;
  auto innerLayerId = layerPairId;
  auto outerLayerId = innerLayerId + 1;
  auto numberOfDoublets = doublets[layerPairId]->size ();

  isOuterHitOfCell[outerLayerId].resize(fourLayers[outerLayerId].hits().size());
  theFoundCellsPerLayer[layerPairId].reserve (numberOfDoublets);
  for (unsigned int i = 0; i < numberOfDoublets; ++i)
  {
    theFoundCellsPerLayer[layerPairId].emplace_back (doublets[layerPairId], i,  cellId,  doublets[layerPairId]->innerHitId(i), doublets[layerPairId]->outerHitId(i));

    isOuterHitOfCell[outerLayerId][doublets[layerPairId]->outerHitId(i)].push_back (&(theFoundCellsPerLayer[layerPairId][i]));
    cellId++;

  }

  for (layerPairId = 1; layerPairId < numberOfLayerPairs; ++layerPairId)
  {

    innerLayerId = layerPairId;
    outerLayerId = innerLayerId + 1;
    numberOfDoublets = doublets[layerPairId]->size ();
    isOuterHitOfCell[outerLayerId].resize(fourLayers[outerLayerId].hits().size());

    theFoundCellsPerLayer[layerPairId].reserve (numberOfDoublets);
    for (unsigned int i = 0; i < numberOfDoublets; ++i)
    {
      theFoundCellsPerLayer[layerPairId].emplace_back (doublets[layerPairId], i, cellId, doublets[layerPairId]->innerHitId(i), doublets[layerPairId]->outerHitId(i));
      isOuterHitOfCell[outerLayerId][doublets[layerPairId]->outerHitId(i)].push_back (&(theFoundCellsPerLayer[layerPairId][i]));
      cellId++;
      for (auto neigCell : isOuterHitOfCell[innerLayerId][doublets[layerPairId]->innerHitId(i)])
      {
        theFoundCellsPerLayer[layerPairId][i].check_alignment_and_tag (neigCell, ptmin);
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
        cell.update_state();
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
    cell.update_state();
    if (cell.is_root_cell (numberOfLayers - 2 ))
    {
      theRootCells.push_back (&cell);
    }
  }
}

template <unsigned int numberOfLayers>
void
CellularAutomaton<numberOfLayers>::find_ntuplets(std::vector<CACell::CAntuplet>& foundNtuplets,  const unsigned int minHitsPerNtuplet)
{
  std::vector<CACell*> tmpNtuplet;
  tmpNtuplet.reserve(numberOfLayers);

  for (CACell* root_cell : theRootCells)
  {
    tmpNtuplet.clear();
    tmpNtuplet.push_back(root_cell);
    root_cell->find_ntuplets (foundNtuplets, tmpNtuplet, minHitsPerNtuplet);
  }

}

template class CellularAutomaton<4>;