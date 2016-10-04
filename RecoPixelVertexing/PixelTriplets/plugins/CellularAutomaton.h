#ifndef RECOPIXELVERTEXING_PIXELTRIPLETS_PLUGINS_CELLULARAUTOMATON_H_
#define RECOPIXELVERTEXING_PIXELTRIPLETS_PLUGINS_CELLULARAUTOMATON_H_
#include <array>
#include "CACell.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "CAGraph.h"
class CellularAutomaton
{
public:
	CellularAutomaton(const CAGraph& graph)
	: theLayerGraph(graph)
	{

	}

	void createAndConnectCells(const std::vector<HitDoublets>&,
			const TrackingRegion&, const float, const float, const float);

	void evolve(const unsigned int);
	void findNtuplets(std::vector<CACell::CAntuplet>&, const unsigned int);
	void findTriplets(const std::vector<HitDoublets>& hitDoublets,std::vector<CACell::CAntuplet>& foundTriplets, const TrackingRegion& region,
			const float thetaCut, const float phiCut, const float hardPtCut);

private:
	CAGraph theLayerGraph;
	std::vector<CACell*> theRootCells;
	std::vector<std::vector<CACell*> > theNtuplets;

};

#endif 
