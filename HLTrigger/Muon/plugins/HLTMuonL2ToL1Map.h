#ifndef HLTMuonL2ToL1Map_h
#define HLTMuonL2ToL1Map_h

/** \class HLTMuonL2ToL1Map
 *
 *  
 *  This is a helper class to check L2 to L1 links
 *
 *  \author Z. Gecse
 *
 */

#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToMany.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/Framework/interface/Event.h"

typedef edm::AssociationMap<edm::OneToMany<std::vector<L2MuonTrajectorySeed>, std::vector<L2MuonTrajectorySeed> > >
    SeedMap;

class HLTMuonL2ToL1Map {
public:
  /// construct with the Token of the L1 filter object, the Token of the L2 seed map ("hltL2Muons") and the Event
  explicit HLTMuonL2ToL1Map(const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>& previousCandToken,
                            const edm::EDGetTokenT<SeedMap> seedMapToken,
                            const edm::Event& iEvent) {
    // get hold of muons that fired the previous level
    edm::Handle<trigger::TriggerFilterObjectWithRefs> previousLevelCands;
    iEvent.getByToken(previousCandToken, previousLevelCands);
    previousLevelCands->getObjects(trigger::TriggerL1Mu, firedL1Muons_);

    // get hold of the seed map
    iEvent.getByToken(seedMapToken, seedMapHandle_);
  }

  ~HLTMuonL2ToL1Map() {}

  /// checks if a L2 muon was seeded by a fired L1
  bool isTriggeredByL1(reco::TrackRef& l2muon) {
    bool isTriggered = false;
    const edm::RefVector<L2MuonTrajectorySeedCollection>& seeds =
        (*seedMapHandle_)[l2muon->seedRef().castTo<edm::Ref<L2MuonTrajectorySeedCollection> >()];
    for (size_t i = 0; i < seeds.size(); i++) {
      if (find(firedL1Muons_.begin(), firedL1Muons_.end(), seeds[i]->l1Particle()) != firedL1Muons_.end()) {
        isTriggered = true;
        break;
      }
    }
    return isTriggered;
  }

  /// returns the indices of L1 seeds
  std::string getL1Keys(reco::TrackRef& l2muon) {
    std::ostringstream ss;
    const edm::RefVector<L2MuonTrajectorySeedCollection>& seeds =
        (*seedMapHandle_)[l2muon->seedRef().castTo<edm::Ref<L2MuonTrajectorySeedCollection> >()];
    for (size_t i = 0; i < seeds.size(); i++) {
      ss << seeds[i]->l1Particle().key() << " ";
    }
    return ss.str();
  }

private:
  /// contains the vector of references to fired L1 candidates
  std::vector<l1extra::L1MuonParticleRef> firedL1Muons_;

  /// containes the map from a L2 seed to its sister seeds the track of which has been cleaned
  edm::Handle<SeedMap> seedMapHandle_;
};

#endif  //HLTMuonL2ToL1Map_h
