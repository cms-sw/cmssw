/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*   Enrico Robutti (robutti@ge.infn.it)
*   Fabio Ravera   (fabio.ravera@cern.ch)
*
*/
#ifndef RecoPPS_Local_RPixPlaneCombinatoryTracking_H
#define RecoPPS_Local_RPixPlaneCombinatoryTracking_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "RecoPPS/Local/interface/RPixDetTrackFinder.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <vector>
#include <map>

class RPixPlaneCombinatoryTracking : public RPixDetTrackFinder {
public:
  RPixPlaneCombinatoryTracking(edm::ParameterSet const &parameterSet);
  ~RPixPlaneCombinatoryTracking() override;
  void initialize() override;
  void findTracks(int run) override;

private:
  typedef std::vector<std::vector<uint32_t> > PlaneCombinations;
  typedef std::vector<RPixDetPatternFinder::PointInPlane> PointInPlaneList;
  typedef std::map<CTPPSPixelDetId, size_t> HitReferences;
  typedef std::map<HitReferences, PointInPlaneList> PointAndReferenceMap;
  typedef std::pair<HitReferences, PointInPlaneList> PointAndReferencePair;

  int verbosity_;
  uint32_t trackMinNumberOfPoints_;
  double maximumChi2OverNDF_;
  double maximumXLocalDistanceFromTrack_;
  double maximumYLocalDistanceFromTrack_;
  PlaneCombinations possiblePlaneCombinations_;

  void getPlaneCombinations(const std::vector<uint32_t> &inputPlaneList,
                            uint32_t numberToExtract,
                            PlaneCombinations &planeCombinations) const;
  CTPPSPixelLocalTrack fitTrack(PointInPlaneList pointList);
  void getHitCombinations(const std::map<CTPPSPixelDetId, PointInPlaneList> &mapOfAllHits,
                          std::map<CTPPSPixelDetId, PointInPlaneList>::iterator mapIterator,
                          HitReferences tmpHitPlaneMap,
                          const PointInPlaneList &tmpHitVector,
                          PointAndReferenceMap &outputMap);
  PointAndReferenceMap produceAllHitCombination(PlaneCombinations inputPlaneCombination);
  bool calculatePointOnDetector(CTPPSPixelLocalTrack *track, CTPPSPixelDetId planeId, GlobalPoint &planeLineIntercept);
  static bool functionForPlaneOrdering(PointAndReferencePair a, PointAndReferencePair b) {
    return (a.second.size() > b.second.size());
  }
  std::vector<PointAndReferencePair> orderCombinationsPerNumberOrPoints(PointAndReferenceMap inputMap);

  inline uint32_t factorial(uint32_t x) const {
    if (x == 0)
      return 1;
    return (x == 1 ? x : x * factorial(x - 1));
  }
};

#endif
