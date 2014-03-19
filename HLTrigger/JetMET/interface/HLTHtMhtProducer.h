#ifndef HLTHtMhtProducer_h_
#define HLTHtMhtProducer_h_

/** \class HLTHtMhtProducer
 *
 *  \brief  This produces a reco::MET object that stores HT and MHT
 *  \author Steven Lowette
 *  \author Michele de Gruttola, Jia Fu Low (Nov 2013)
 *
 *  HT & MHT are calculated using input CaloJet or PFJet collection.
 *  MHT can include or exclude the contribution from muons.
 *  HT is stored as `sumet_`, MHT is stored as `p4_` in the output.
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


namespace edm {
    class ConfigurationDescriptions;
}

// Class declaration
class HLTHtMhtProducer : public edm::EDProducer {
  public:
    explicit HLTHtMhtProducer(const edm::ParameterSet & iConfig);
    ~HLTHtMhtProducer();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:
    /// Use pt; otherwise, use et.
    bool usePt_;

    /// Exclude PF muons in the MHT calculation (but not HT)
    /// Ignored if pfCandidatesLabel_ is empty.
    bool excludePFMuons_;

    /// Minimum number of jets passing pt and eta requirements
    int minNJetHt_;
    int minNJetMht_;

    /// Minimum pt requirement for jets
    double minPtJetHt_;
    double minPtJetMht_;

    /// Maximum (abs) eta requirement for jets
    double maxEtaJetHt_;
    double maxEtaJetMht_;

    /// Input jet, PFCandidate collections
    edm::InputTag jetsLabel_;
    edm::InputTag pfCandidatesLabel_;

    edm::EDGetTokenT<reco::JetView> m_theJetToken;
    edm::EDGetTokenT<reco::PFCandidateCollection> m_thePFCandidateToken;
};

#endif  // HLTHtMhtProducer_h_

