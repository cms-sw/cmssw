#include "DataFormats/Common/interface/View.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "fastjet/tools/GridMedianBackgroundEstimator.hh"

class FixedGridRhoProducerFastjet : public edm::stream::EDProducer<> {
public:
  explicit FixedGridRhoProducerFastjet(const edm::ParameterSet& iConfig);
  ~FixedGridRhoProducerFastjet() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  fastjet::GridMedianBackgroundEstimator bge_;
  const edm::InputTag pfCandidatesTag_;
  const edm::EDGetTokenT<edm::View<reco::Candidate> > input_pfcoll_token_;
};

using namespace std;

FixedGridRhoProducerFastjet::FixedGridRhoProducerFastjet(const edm::ParameterSet& iConfig)
    : bge_(iConfig.getParameter<double>("maxRapidity"), iConfig.getParameter<double>("gridSpacing")),
      pfCandidatesTag_{iConfig.getParameter<edm::InputTag>("pfCandidatesTag")},
      input_pfcoll_token_{consumes<edm::View<reco::Candidate> >(pfCandidatesTag_)} {
  produces<double>();
}

void FixedGridRhoProducerFastjet::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::Candidate> > pfColl;
  iEvent.getByToken(input_pfcoll_token_, pfColl);
  std::vector<fastjet::PseudoJet> inputs;
  for (edm::View<reco::Candidate>::const_iterator ibegin = pfColl->begin(), iend = pfColl->end(), i = ibegin; i != iend;
       ++i) {
    inputs.push_back(fastjet::PseudoJet(i->px(), i->py(), i->pz(), i->energy()));
  }
  bge_.set_particles(inputs);
  iEvent.put(std::make_unique<double>(bge_.rho()));
}

void FixedGridRhoProducerFastjet::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("maxRapidity", 5.0);
  desc.add<double>("gridSpacing", 0.55);
  desc.add<edm::InputTag>("pfCandidatesTag", edm::InputTag(""));
  descriptions.add("default_FixedGridRhoProducerFastjet", desc);
}

DEFINE_FWK_MODULE(FixedGridRhoProducerFastjet);
