
#include "CellularAutomaton.h"

template <unsigned int numberOfLayers>
void CellularAutomaton<numberOfLayers>::create_and_connect_cells (std::vector<const HitDoublets*> doublets, const SeedingLayerSetsHits::SeedingLayerSet& fourLayers, const TrackingRegion& region)
{

  unsigned int cellId = 0;
  constexpr unsigned int numberOfLayerPairs =   numberOfLayers - 1;
  float ptmin = region.ptMin();

  for (unsigned int layerPairId = 0; layerPairId < numberOfLayerPairs; ++layerPairId)
  {
    auto innerLayerId = layerPairId;
    auto outerLayerId = innerLayerId + 1;
    auto numberOfDoublets = doublets[layerPairId]->size ();


    isOuterHitOfCell[outerLayerId].resize(fourLayers[outerLayerId].hits().size());

    theFoundCellsPerLayer[layerPairId].reserve (numberOfDoublets);

    if (layerPairId == 0)
    {
      for (unsigned int i = 0; i < numberOfDoublets; ++i)
      {
        theFoundCellsPerLayer[layerPairId].emplace_back (CACell(doublets[layerPairId], i,  cellId++,  doublets[layerPairId]->innerHitId(i), doublets[layerPairId]->outerHitId(i)));


        isOuterHitOfCell[outerLayerId][doublets[layerPairId]->outerHitId(i)].push_back (&(theFoundCellsPerLayer[layerPairId][i]));
      }
    }// if the layer is not the innermost one we check the compatibility between the two cells that share the same hit: one in the inner layer, previously created,
      // and the one we are about to create. If these two cells meet the neighboring conditions, they become one the neighbor of the other.
    else
    {
      for (unsigned int i = 0; i < numberOfDoublets; ++i)
      {


        theFoundCellsPerLayer[layerPairId].emplace_back (CACell(doublets[layerPairId], i, cellId++, doublets[layerPairId]->innerHitId(i), doublets[layerPairId]->outerHitId(i)));

        isOuterHitOfCell[outerLayerId][doublets[layerPairId]->outerHitId(i)].push_back (&(theFoundCellsPerLayer[layerPairId][i]));



        for (auto neigCell : isOuterHitOfCell[innerLayerId][doublets[layerPairId]->innerHitId(i)])
        {
          theFoundCellsPerLayer[layerPairId][i].check_alignment_and_tag (neigCell, ptmin);

        }

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
