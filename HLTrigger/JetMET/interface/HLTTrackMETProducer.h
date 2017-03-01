#ifndef HLTTrackMETProducer_h_
#define HLTTrackMETProducer_h_

/** \class HLTTrackMETProducer
 *
 *  \brief  This produces a reco::MET object that stores MHT (or MET)
 *  \author Steven Lowette
 *  \author Michele de Gruttola, Jia Fu Low (Nov 2013)
 *
 *  MHT (or MET) is calculated using one of the input collections:
 *    - Calo or PF jets
 *    - tracks
 *    - PF tracks
 *    - PF charged candidates
 *    - PF candidates
 *  MHT can include or exclude the contribution from muons.
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


namespace edm {
    class ConfigurationDescriptions;
}

// Class declaration
class HLTTrackMETProducer : public edm::stream::EDProducer<> {
  public:
    explicit HLTTrackMETProducer(const edm::ParameterSet & iConfig);
    ~HLTTrackMETProducer();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:
    /// Use pt; otherwise, use et.
    /// Ignored if jets are not used as input.
    bool usePt_;

    /// Use reco tracks as input instead of jets.
    /// If true, it overrides usePFRecTracks, usePFCandidatesCharged_ & usePFCandidates_.
    bool useTracks_;

    /// Use PF tracks as input instead of jets.
    /// If true, it overrides usePFCandidatesCharged_ & usePFCandidates_.
    bool usePFRecTracks_;

    /// Use PF charged candidates as input instead of jets.
    /// If true, it overrides usePFCandidates_.
    bool usePFCandidatesCharged_;

    /// Use PF candidates as input instead of jets.
    bool usePFCandidates_;

    /// Exclude PF muons in the MHT calculation (but not HT)
    /// Ignored if pfCandidatesLabel_ is empty.
    bool excludePFMuons_;

    /// Minimum number of jets passing pt and eta requirements
    /// Ignored if jets are not used as input
    int minNJet_;

    /// Minimum pt requirement for jets (or objects used as input)
    double minPtJet_;

    /// Maximum (abs) eta requirement for jets (or objects used as input)
    double maxEtaJet_;

    /// Input jet, track, PFRecTrack, PFCandidate collections
    edm::InputTag jetsLabel_;
    edm::InputTag tracksLabel_;
    edm::InputTag pfRecTracksLabel_;
    edm::InputTag pfCandidatesLabel_;

    edm::EDGetTokenT<reco::JetView> m_theJetToken;
    edm::EDGetTokenT<reco::TrackCollection> m_theTrackToken;
    edm::EDGetTokenT<reco::PFRecTrackCollection> m_theRecTrackToken;
    edm::EDGetTokenT<reco::PFCandidateCollection> m_thePFCandidateToken;
};

#endif  // HLTTrackMETProducer_h_

