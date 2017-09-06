/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*   Enrico Robutti (robutti@ge.infn.it)
*   Fabio Ravera   (fabio.ravera@cern.ch)
*
*/
#ifndef RecoCTPPS_PixelLocal_RPixPlaneCombinatoryTracking_H
#define RecoCTPPS_PixelLocal_RPixPlaneCombinatoryTracking_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "RecoCTPPS/PixelLocal/interface/RPixDetTrackFinder.h"


#include <vector>
#include <map>

class RPixPlaneCombinatoryTracking : public RPixDetTrackFinder{

  public:
    RPixPlaneCombinatoryTracking(edm::ParameterSet const& parameterSet);
    ~RPixPlaneCombinatoryTracking() override;
    void initialize() override;
    void findTracks() override;

  private:
    edm::ParameterSet param_;
    int verbosity_;
    uint32_t trackMinNumberOfPoints_;
    double maximumChi2OverNDF_;
    double maximumXLocalDistanceFromTrack_;
    double maximumYLocalDistanceFromTrack_;
    std::vector<std::vector<uint32_t> > possiblePlaneCombinations_;
    
    std::vector<std::vector<uint32_t> > getPlaneCombinations(std::vector<uint32_t> inputPlaneList, uint32_t numberToExtract);
    CTPPSPixelLocalTrack fitTrack(std::vector<RPixDetPatternFinder::PointInPlane> pointList);
    void getHitCombinations(
        const std::map<CTPPSPixelDetId, std::vector<RPixDetPatternFinder::PointInPlane> > &mapOfAllHits, 
        std::map<CTPPSPixelDetId, std::vector<RPixDetPatternFinder::PointInPlane> >::iterator mapIterator,
        std::map<CTPPSPixelDetId, size_t> tmpHitPlaneMap,
        std::vector<RPixDetPatternFinder::PointInPlane> tmpHitVector,
        std::map< std::map<CTPPSPixelDetId, size_t>, std::vector<RPixDetPatternFinder::PointInPlane> > &outputMap);
    std::map< std::map<CTPPSPixelDetId, size_t>, std::vector<RPixDetPatternFinder::PointInPlane> > produceAllHitCombination(std::vector<std::vector<uint32_t> > inputPlaneCombination);
    bool calculatePointOnDetector(CTPPSPixelLocalTrack track, CTPPSPixelDetId planeId, TVector3 &planeLineIntercept);
    static bool functionForPlaneOrdering(
        std::pair<std::map<CTPPSPixelDetId, size_t>, std::vector<RPixDetPatternFinder::PointInPlane> > a,
        std::pair<std::map<CTPPSPixelDetId, size_t>, std::vector<RPixDetPatternFinder::PointInPlane> > b) { 
        return (a.second.size() > b.second.size()); }
    std::vector<std::pair <std::map<CTPPSPixelDetId, size_t>, std::vector<RPixDetPatternFinder::PointInPlane> > > orderCombinationsPerNumberOrPoints(
        std::map< std::map<CTPPSPixelDetId, size_t>, std::vector<RPixDetPatternFinder::PointInPlane> > inputMap);


    inline uint32_t Factorial(uint32_t x) {
      if(x==0) return 1;
      return (x == 1 ? x : x * Factorial(x - 1));
    }

};

#endif