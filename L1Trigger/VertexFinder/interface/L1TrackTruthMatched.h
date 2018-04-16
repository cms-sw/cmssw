#ifndef __L1Trigger_VertexFinder_L1TrackTruthMatched_h__
#define __L1Trigger_VertexFinder_L1TrackTruthMatched_h__


#include <vector>

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

// TTStubAssociationMap.h forgets to two needed files, so must include them here ...
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

#include "L1Trigger/VertexFinder/interface/L1Track.h"

class TrackerGeometry;
class TrackerTopology;

namespace l1tVertexFinder {

class AnalysisSettings;
class Stub;
class TP;

typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_>           TTStubAssMap;
typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_>        TTClusterAssMap;

//! Simple wrapper class for TTTrack, with match to a tracking particle
class L1TrackTruthMatched : public L1Track {
public:
  L1TrackTruthMatched(const edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >>&, const AnalysisSettings& , const TrackerGeometry* , const TrackerTopology*, const std::map<edm::Ptr< TrackingParticle >, const TP* >& translateTP, edm::Handle<TTStubAssMap> mcTruthTTStubHandle, edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle, const std::map<DetId, DetId>& geoDetIdMap);
  ~L1TrackTruthMatched();
  unsigned int getNumStubs()  const  {return numStubs;}

  // Get best matching tracking particle (=nullptr if none).
  const TP* getMatchedTP() const;

private:
  edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >> track_;

  //--- Information about its association (if any) to a truth Tracking Particle.
  const TP*             matchedTP_;
  std::vector<const Stub*>   matchedStubs_;
  unsigned int          nMatchedLayers_;
  unsigned int          numStubs;
};

} // end ns l1tVertexFinder


#endif
