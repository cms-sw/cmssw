#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace pat {
  class PATTauCandidatesRekeyer : public edm::stream::EDProducer<> {
  public:
    explicit PATTauCandidatesRekeyer(const edm::ParameterSet &iConfig);
    ~PATTauCandidatesRekeyer() override;

    void produce(edm::Event &, const edm::EventSetup &) override;

  private:
    // configurables
    edm::EDGetTokenT<std::vector<pat::Tau>> src_;
    edm::EDGetTokenT<reco::CandidateView> pcNewCandViewToken_;
    edm::EDGetTokenT<pat::PackedCandidateCollection> pcNewToken_;
  };

}  // namespace pat

using namespace pat;

PATTauCandidatesRekeyer::PATTauCandidatesRekeyer(const edm::ParameterSet &iConfig)
    : src_(consumes<std::vector<pat::Tau>>(iConfig.getParameter<edm::InputTag>("src"))),
      pcNewCandViewToken_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesNew"))),
      pcNewToken_(
          consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesNew"))) {
  produces<std::vector<pat::Tau>>();
}

PATTauCandidatesRekeyer::~PATTauCandidatesRekeyer() {}

void PATTauCandidatesRekeyer::produce(edm::Event &iEvent, edm::EventSetup const &) {
  edm::Handle<std::vector<pat::Tau>> src;
  iEvent.getByToken(src_, src);

  edm::Handle<reco::CandidateView> pcNewCandViewHandle;
  iEvent.getByToken(pcNewCandViewToken_, pcNewCandViewHandle);

  edm::Handle<pat::PackedCandidateCollection> pcNewHandle;
  iEvent.getByToken(pcNewToken_, pcNewHandle);

  auto outPtrP = std::make_unique<std::vector<pat::Tau>>();
  outPtrP->reserve(src->size());

  for (const auto &obj : *src) {
    // copy original pat object and append to vector
    outPtrP->emplace_back(obj);

    reco::CandidatePtrVector signalChHPtrs;
    for (const reco::CandidatePtr &p : outPtrP->back().signalChargedHadrCands()) {
      signalChHPtrs.push_back(edm::Ptr<reco::Candidate>(pcNewHandle, p.key()));
    }
    outPtrP->back().setSignalChargedHadrCands(signalChHPtrs);

    reco::CandidatePtrVector signalNHPtrs;
    for (const reco::CandidatePtr &p : outPtrP->back().signalNeutrHadrCands()) {
      signalNHPtrs.push_back(edm::Ptr<reco::Candidate>(pcNewHandle, p.key()));
    }
    outPtrP->back().setSignalNeutralHadrCands(signalNHPtrs);

    reco::CandidatePtrVector signalGammaPtrs;
    for (const reco::CandidatePtr &p : outPtrP->back().signalGammaCands()) {
      signalGammaPtrs.push_back(edm::Ptr<reco::Candidate>(pcNewHandle, p.key()));
    }
    outPtrP->back().setSignalGammaCands(signalGammaPtrs);

    reco::CandidatePtrVector isolationChHPtrs;
    for (const reco::CandidatePtr &p : outPtrP->back().isolationChargedHadrCands()) {
      isolationChHPtrs.push_back(edm::Ptr<reco::Candidate>(pcNewHandle, p.key()));
    }
    outPtrP->back().setIsolationChargedHadrCands(isolationChHPtrs);

    reco::CandidatePtrVector isolationNHPtrs;

    for (const reco::CandidatePtr &p : outPtrP->back().isolationNeutrHadrCands()) {
      isolationNHPtrs.push_back(edm::Ptr<reco::Candidate>(pcNewHandle, p.key()));
    }
    outPtrP->back().setIsolationNeutralHadrCands(isolationNHPtrs);

    reco::CandidatePtrVector isolationGammaPtrs;
    for (const reco::CandidatePtr &p : outPtrP->back().isolationGammaCands()) {
      isolationGammaPtrs.push_back(edm::Ptr<reco::Candidate>(pcNewHandle, p.key()));
    }
    outPtrP->back().setIsolationGammaCands(isolationGammaPtrs);
  }
  iEvent.put(std::move(outPtrP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATTauCandidatesRekeyer);
