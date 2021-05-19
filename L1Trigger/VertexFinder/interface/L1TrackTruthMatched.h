#ifndef __L1Trigger_VertexFinder_L1TrackTruthMatched_h__
#define __L1Trigger_VertexFinder_L1TrackTruthMatched_h__

#include <vector>

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

// TTStubAssociationMap.h forgets to two needed files, so must include them here ...
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"

#include "L1Trigger/VertexFinder/interface/L1Track.h"
#include "L1Trigger/VertexFinder/interface/utility.h"

class TrackerGeometry;
class TrackerTopology;

namespace l1tVertexFinder {

  class AnalysisSettings;
  class TP;

  typedef TTTrackAssociationMap<Ref_Phase2TrackerDigi_> TTTrackAssMap;

  //! Simple wrapper class for TTTrack, with match to a tracking particle
  class L1TrackTruthMatched : public L1Track {
  public:
    L1TrackTruthMatched(const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>& aTrack,
                        const std::map<edm::Ptr<TrackingParticle>, const TP*>& translateTP,
                        edm::Handle<TTTrackAssMap> mcTruthTTTrackHandle)
        : L1Track(aTrack) {
      auto mcTruthTP = mcTruthTTTrackHandle->findTrackingParticlePtr(aTrack);
      if (!mcTruthTP.isNull()) {
        auto tpTranslation = translateTP.find(mcTruthTP);
        if (tpTranslation != translateTP.end()) {
          matchedTP_ = tpTranslation->second;
        } else {
          matchedTP_ = nullptr;
        }
      } else {
        matchedTP_ = nullptr;
      }
    }
    ~L1TrackTruthMatched() {}

    // Get best matching tracking particle (=nullptr if none).
    const TP* getMatchedTP() const { return matchedTP_; }

  private:
    const TP* matchedTP_;
  };

}  // namespace l1tVertexFinder

#endif
