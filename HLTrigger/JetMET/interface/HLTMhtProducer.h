#ifndef HLTMhtProducer_h_
#define HLTMhtProducer_h_

/** \class HLTMhtProducer
 *
 *  \brief  This produces a reco::MET object that stores MHT (or MET)
 *  \author Steven Lowette
 *  \author Michele de Gruttola, Jia Fu Low (Nov 2013)
 *
 *  MHT (or MET) is calculated using input CaloJet or PFJet collection.
 *  MHT can include or exclude the contribution from muons.
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

// Class declaration
class HLTMhtProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTMhtProducer(const edm::ParameterSet& iConfig);
  ~HLTMhtProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  /// Use pt; otherwise, use et.
  bool usePt_;

  /// Exclude PF muons in the MHT calculation (but not HT)
  /// Ignored if pfCandidatesLabel_ is empty.
  bool excludePFMuons_;

  /// Minimum number of jets passing pt and eta requirements
  int minNJet_;

  /// Minimum pt requirement for jets
  double minPtJet_;

  /// Maximum (abs) eta requirement for jets
  double maxEtaJet_;

  /// Input jet, PFCandidate collections
  edm::InputTag jetsLabel_;
  edm::InputTag pfCandidatesLabel_;

  edm::EDGetTokenT<reco::CandidateView> m_theJetToken;
  edm::EDGetTokenT<reco::PFCandidateCollection> m_thePFCandidateToken;
};

#endif  // HLTMhtProducer_h_
