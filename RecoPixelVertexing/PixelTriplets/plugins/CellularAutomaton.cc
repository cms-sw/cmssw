#include "CellularAutomaton.h"

void CellularAutomaton::createAndConnectCells(const std::vector<HitDoublets>& hitDoublets, const TrackingRegion& region,
		const float thetaCut, const float phiCut, const float hardPtCut)
{
	unsigned int cellId = 0;
	float ptmin = region.ptMin();
	float region_origin_x = region.origin().x();
	float region_origin_y = region.origin().y();
	float region_origin_radius = region.originRBound();

	std::vector<bool> alreadyVisitedLayerPairs;
	alreadyVisitedLayerPairs.resize(theLayerGraph.theLayerPairs.size());
	for (auto visited : alreadyVisitedLayerPairs)
	{
		visited = false;
	}
	for (int rootVertex : theLayerGraph.theRootLayers)
	{

		std::queue<int> LayerPairsToVisit;

		for (int LayerPair : theLayerGraph.theLayers[rootVertex].theOuterLayerPairs)
		{
			LayerPairsToVisit.push(LayerPair);

		}

		unsigned int numberOfLayerPairsToVisitAtThisDepth =
				LayerPairsToVisit.size();

		while (!LayerPairsToVisit.empty())
		{
			auto currentLayerPair = LayerPairsToVisit.front();
			auto & currentLayerPairRef = theLayerGraph.theLayerPairs[currentLayerPair];
			auto & currentInnerLayerRef = theLayerGraph.theLayers[currentLayerPairRef.theLayers[0]];
			auto & currentOuterLayerRef = theLayerGraph.theLayers[currentLayerPairRef.theLayers[1]];
			bool allInnerLayerPairsAlreadyVisited	{ true };

			for (auto innerLayerPair : currentInnerLayerRef.theInnerLayerPairs)
			{
				allInnerLayerPairsAlreadyVisited &=
						alreadyVisitedLayerPairs[innerLayerPair];
			}

			if (alreadyVisitedLayerPairs[currentLayerPair] == false
					&& allInnerLayerPairsAlreadyVisited)
			{

				const HitDoublets* doubletLayerPairId =
						&(hitDoublets[currentLayerPair]);
				auto numberOfDoublets = doubletLayerPairId->size();
				currentLayerPairRef.theFoundCells.reserve(numberOfDoublets);
				for (unsigned int i = 0; i < numberOfDoublets; ++i)
				{
					currentLayerPairRef.theFoundCells.emplace_back(
							doubletLayerPairId, i, cellId,
							doubletLayerPairId->innerHitId(i),
							doubletLayerPairId->outerHitId(i));
					currentOuterLayerRef.isOuterHitOfCell[doubletLayerPairId->outerHitId(i)].push_back(
							&(currentLayerPairRef.theFoundCells[i]));
					cellId++;

					for (auto neigCell : currentInnerLayerRef.isOuterHitOfCell[doubletLayerPairId->innerHitId(i)])
					{
						currentLayerPairRef.theFoundCells[i].checkAlignmentAndTag(
								neigCell, ptmin, region_origin_x,
								region_origin_y, region_origin_radius, thetaCut,
								phiCut, hardPtCut);
					}

				}

				for (auto outerLayerPair : currentOuterLayerRef.theOuterLayerPairs)
				{
					LayerPairsToVisit.push(outerLayerPair);
				}

				alreadyVisitedLayerPairs[currentLayerPair] = true;
			}
			LayerPairsToVisit.pop();
			numberOfLayerPairsToVisitAtThisDepth--;
			if (numberOfLayerPairsToVisitAtThisDepth == 0)
			{
				numberOfLayerPairsToVisitAtThisDepth = LayerPairsToVisit.size();
			}

		}

	}

}

void CellularAutomaton::evolve(const unsigned int minHitsPerNtuplet)
{
	unsigned int numberOfIterations = minHitsPerNtuplet - 2;
	// keeping the last iteration for later
	for (unsigned int iteration = 0; iteration < numberOfIterations - 1;
			++iteration)
	{
		for (auto& layerPair : theLayerGraph.theLayerPairs)
		{
			for (auto& cell : layerPair.theFoundCells)
			{
				cell.evolve();
			}
		}

		for (auto& layerPair : theLayerGraph.theLayerPairs)
		{
			for (auto& cell : layerPair.theFoundCells)
			{
				cell.updateState();
			}
		}

	}

	//last iteration


	for(int rootLayerId : theLayerGraph.theRootLayers)
	{
		for(int rootLayerPair: theLayerGraph.theLayers[rootLayerId].theOuterLayerPairs)
		{
			for (auto& cell : theLayerGraph.theLayerPairs[rootLayerPair].theFoundCells)
			{
				cell.evolve();
				cell.updateState();
				if (cell.isRootCell(minHitsPerNtuplet - 2))
				{
					theRootCells.push_back(&cell);
				}
			}
		}
	}

}

void CellularAutomaton::findNtuplets(
		std::vector<CACell::CAntuplet>& foundNtuplets,
		const unsigned int minHitsPerNtuplet)
{
	std::vector<CACell*> tmpNtuplet;
	tmpNtuplet.reserve(minHitsPerNtuplet);

	for (CACell* root_cell : theRootCells)
	{
		tmpNtuplet.clear();
		tmpNtuplet.push_back(root_cell);
		root_cell->findNtuplets(foundNtuplets, tmpNtuplet, minHitsPerNtuplet);
	}

}


//template<unsigned int numberOfLayers>
//void CellularAutomaton<numberOfLayers>::findTriplets(
//		std::array<const HitDoublets*, numberOfLayers - 1> doublets,
//		const SeedingLayerSetsHits::SeedingLayerSet& fourLayers,
//		std::vector<CACell::CAntuplet>& foundTriplets,
//		const TrackingRegion& region, const float thetaCut, const float phiCut)
//{
//	unsigned int cellId = 0;
//	constexpr unsigned int numberOfLayerPairs = numberOfLayers - 1;
//	float ptmin = region.ptMin();
//	float region_origin_x = region.origin().x();
//	float region_origin_y = region.origin().y();
//	float region_origin_radius = region.originRBound();
//	unsigned int layerPairId = 0;
//	auto innerLayerId = layerPairId;
//	auto outerLayerId = innerLayerId + 1;
//	auto & doubletLayerPairId = doublets[layerPairId];
//	auto numberOfDoublets = doubletLayerPairId->size();
//
//	isOuterHitOfCell[outerLayerId].resize(
//			fourLayers[outerLayerId].hits().size());
//
//	theFoundCellsPerLayer[layerPairId].reserve(numberOfDoublets);
//	for (unsigned int i = 0; i < numberOfDoublets; ++i)
//	{
//		theFoundCellsPerLayer[layerPairId].emplace_back(doubletLayerPairId, i,
//				cellId, doubletLayerPairId->innerHitId(i),
//				doubletLayerPairId->outerHitId(i));
//
//		isOuterHitOfCell[outerLayerId][doubletLayerPairId->outerHitId(i)].push_back(
//				&(theFoundCellsPerLayer[layerPairId][i]));
//		cellId++;
//
//	}
//
//	for (layerPairId = 1; layerPairId < numberOfLayerPairs; ++layerPairId)
//	{
//		doubletLayerPairId = doublets[layerPairId];
//		innerLayerId = layerPairId;
//		outerLayerId = innerLayerId + 1;
//		numberOfDoublets = doubletLayerPairId->size();
//		isOuterHitOfCell[outerLayerId].resize(
//				fourLayers[outerLayerId].hits().size());
//
//		theFoundCellsPerLayer[layerPairId].reserve(numberOfDoublets);
//		for (unsigned int i = 0; i < numberOfDoublets; ++i)
//		{
//			theFoundCellsPerLayer[layerPairId].emplace_back(
//					doublets[layerPairId], i, cellId,
//					doubletLayerPairId->innerHitId(i),
//					doubletLayerPairId->outerHitId(i));
//			isOuterHitOfCell[outerLayerId][doubletLayerPairId->outerHitId(i)].push_back(
//					&(theFoundCellsPerLayer[layerPairId][i]));
//			cellId++;
//			for (auto neigCell : isOuterHitOfCell[innerLayerId][doubletLayerPairId->innerHitId(
//					i)])
//			{
//				theFoundCellsPerLayer[layerPairId][i].checkAlignmentAndPushTriplet(
//						neigCell, foundTriplets, ptmin, region_origin_x,
//						region_origin_y, region_origin_radius, thetaCut,
//						phiCut);
//			}
//		}
//	}
//}
//
//template class CellularAutomaton<4> ;
//template class CellularAutomaton<3> ;
//
