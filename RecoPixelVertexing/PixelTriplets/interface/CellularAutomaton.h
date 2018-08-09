#ifndef RecoPixelVertexing_PixelTriplets_interface_CellularAutomaton_h
#define RecoPixelVertexing_PixelTriplets_interface_CellularAutomaton_h

#include <array>

#include "RecoPixelVertexing/PixelTriplets/interface/CACell.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CAGraph.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

class CellularAutomaton
{
public:
  CellularAutomaton(CAGraph& graph)
    : theLayerGraph(graph)
  { }
  
  std::vector<CACell> & getAllCells() { return allCells; }
  
  void createAndConnectCells(const std::vector<const HitDoublets *>&,
			     const TrackingRegion&, const float, const float, const float);
  
  void evolve(const unsigned int);
  void findNtuplets(std::vector<CACell::CAntuplet>&, const unsigned int);
  void findTriplets(const std::vector<const HitDoublets*>& hitDoublets,std::vector<CACell::CAntuplet>& foundTriplets, const TrackingRegion& region,
		    const float thetaCut, const float phiCut, const float hardPtCut);
  
private:
  CAGraph & theLayerGraph;

  std::vector<CACell> allCells;
  std::vector<CACellStatus> allStatus;

  std::vector<unsigned int> theRootCells;
  std::vector<std::vector<CACell*> > theNtuplets;
  
};

#endif // RecoPixelVertexing_PixelTriplets_interface_CellularAutomaton_h
