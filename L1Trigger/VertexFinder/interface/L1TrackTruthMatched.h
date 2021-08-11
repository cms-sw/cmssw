#ifndef __L1Trigger_VertexFinder_L1TrackTruthMatched_h__
#define __L1Trigger_VertexFinder_L1TrackTruthMatched_h__

#include <vector>

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/VertexFinder/interface/L1Track.h"
#include "L1Trigger/VertexFinder/interface/TP.h"
// TTStubAssociationMap.h forgets to two needed files, so must include them here ...
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"

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
                        const std::map<edm::Ptr<TrackingParticle>, edm::RefToBase<TrackingParticle>>& tpPtrToRefMap,
                        const edm::ValueMap<TP>& tpValueMap,
                        edm::Handle<TTTrackAssMap> mcTruthTTTrackHandle)
        : L1Track(aTrack), matchedTPidx_(-1) {
      auto mcTruthTP = mcTruthTTTrackHandle->findTrackingParticlePtr(aTrack);
      if (!mcTruthTP.isNull()) {
        auto tpTranslation = tpPtrToRefMap.find(mcTruthTP);
        if (tpTranslation != tpPtrToRefMap.end()) {
          matchedTP_ = &tpValueMap[tpTranslation->second];
          matchedTPidx_ = std::distance(tpPtrToRefMap.begin(), tpTranslation);
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

    // Get the index of the matched TP in the map of TP particles with the use flag set
    const int getMatchedTPidx() const { return matchedTPidx_; }

  private:
    const TP* matchedTP_;
    int matchedTPidx_;
  };

}  // namespace l1tVertexFinder

#endif
