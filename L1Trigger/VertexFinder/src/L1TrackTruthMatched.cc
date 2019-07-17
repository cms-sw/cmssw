
#include "L1Trigger/VertexFinder/interface/VertexFinder.h"


#include "L1Trigger/VertexFinder/interface/Stub.h"
#include "L1Trigger/VertexFinder/interface/utility.h"



namespace l1tVertexFinder {

L1TrackTruthMatched::L1TrackTruthMatched(const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>& aTrack,
                                         const AnalysisSettings& aSettings,
                                         const TrackerGeometry* trackerGeometry,
                                         const TrackerTopology* trackerTopology,
                                         const std::map<edm::Ptr<TrackingParticle>, const TP*>& translateTP,
                                         edm::Handle<TTStubAssMap> mcTruthTTStubHandle,
                                         edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle,
                                         const std::map<DetId, DetId>& geoDetIdMap) :
  L1Track(aTrack)
{
  std::vector<Stub*> stubs;
  for (const auto& stubRef : aTrack->getStubRefs()) {
    stubs.push_back(new Stub(stubRef, 0, aSettings, trackerGeometry, trackerTopology, geoDetIdMap));
    stubs.back()->fillTruth(translateTP, mcTruthTTStubHandle, mcTruthTTClusterHandle);
  }
  matchedTP_ = utility::matchingTP(aSettings, stubs, nMatchedLayers_, matchedStubs_);

  numStubs = stubs.size();
  for (const auto& stub : stubs)
    delete stub;
}

L1TrackTruthMatched::~L1TrackTruthMatched()
{
}

const TP* L1TrackTruthMatched::getMatchedTP() const
{
  return matchedTP_;
}

} // end ns l1tVertexFinder
