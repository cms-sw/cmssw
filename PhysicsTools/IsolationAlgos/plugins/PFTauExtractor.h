#ifndef PhysicsTools_IsolationAlgos_PFTauExtractor_H
#define PhysicsTools_IsolationAlgos_PFTauExtractor_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

class PFTauExtractor : public reco::isodeposit::IsoDepositExtractor {
public:
  explicit PFTauExtractor(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
  ~PFTauExtractor() override {}

  /// definition of pure virtual functions inherited from IsoDepositExtractor base-class
  void fillVetos(const edm::Event&, const edm::EventSetup&, const reco::TrackCollection&) override {}
  reco::IsoDeposit deposit(const edm::Event& evt, const edm::EventSetup& es, const reco::Track& track) const override {
    return depositFromObject(evt, es, track);
  }
  reco::IsoDeposit deposit(const edm::Event& evt,
                           const edm::EventSetup& es,
                           const reco::Candidate& candidate) const override {
    return depositFromObject(evt, es, candidate);
  }

private:
  /// configuration parameters
  edm::EDGetTokenT<reco::PFTauCollection> tauSourceToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > candidateSourceToken_;
  double maxDxyTrack_;
  double maxDzTrack_;
  double dRmatchPFTau_;
  double dRVetoCone_;
  double dRIsoCone_;
  double dRvetoPFTauSignalConeConstituents_;

  /// private member function for computing the IsoDeposits
  /// in case of reco::Track as well as in case of reco::Canididate input
  template <typename T>
  reco::IsoDeposit depositFromObject(const edm::Event&, const edm::EventSetup&, const T&) const;
};

#endif
