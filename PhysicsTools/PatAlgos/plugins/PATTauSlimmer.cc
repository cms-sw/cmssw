/**
  \class    pat::PATTauSlimmer PATTauSlimmer.h "PhysicsTools/PatAlgos/interface/PATTauSlimmer.h"
  \brief    Slimmer of PAT Taus 
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "PhysicsTools/PatAlgos/interface/ObjectModifier.h"

namespace pat {

  class PATTauSlimmer : public edm::stream::EDProducer<> {
  public:
    explicit PATTauSlimmer(const edm::ParameterSet &iConfig);
    ~PATTauSlimmer() override {}

    void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

  private:
    const edm::EDGetTokenT<edm::View<pat::Tau>> src_;
    const bool linkToPackedPF_;
    const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> pf2pc_;
    const bool linkToLostTracks_;
    const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> trk2lost_;
    const bool dropPiZeroRefs_;
    const bool dropTauChargedHadronRefs_;
    const bool dropPFSpecific_;
    const bool modifyTau_;
    std::unique_ptr<pat::ObjectModifier<pat::Tau>> tauModifier_;
  };

}  // namespace pat

pat::PATTauSlimmer::PATTauSlimmer(const edm::ParameterSet &iConfig)
    : src_(consumes<edm::View<pat::Tau>>(iConfig.getParameter<edm::InputTag>("src"))),
      linkToPackedPF_(iConfig.getParameter<bool>("linkToPackedPFCandidates")),
      pf2pc_(mayConsume<edm::Association<pat::PackedCandidateCollection>>(
          iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
      linkToLostTracks_(iConfig.getParameter<bool>("linkToLostTracks")),
      trk2lost_(mayConsume<edm::Association<pat::PackedCandidateCollection>>(
          iConfig.getParameter<edm::InputTag>("lostTracks"))),
      dropPiZeroRefs_(iConfig.exists("dropPiZeroRefs") ? iConfig.getParameter<bool>("dropPiZeroRefs") : true),
      dropTauChargedHadronRefs_(
          iConfig.exists("dropTauChargedHadronRefs") ? iConfig.getParameter<bool>("dropTauChargedHadronRefs") : true),
      dropPFSpecific_(iConfig.exists("dropPFSpecific") ? iConfig.getParameter<bool>("dropPFSpecific") : true),
      modifyTau_(iConfig.getParameter<bool>("modifyTaus")) {
  if (modifyTau_) {
    const edm::ParameterSet &mod_config = iConfig.getParameter<edm::ParameterSet>("modifierConfig");
    tauModifier_ = std::make_unique<pat::ObjectModifier<pat::Tau>>(mod_config, consumesCollector());
  }
  produces<std::vector<pat::Tau>>();
}

void pat::PATTauSlimmer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace std;

  if (modifyTau_)
    tauModifier_->setEventContent(iSetup);

  Handle<View<pat::Tau>> src;
  iEvent.getByToken(src_, src);

  Handle<edm::Association<pat::PackedCandidateCollection>> pf2pc;
  if (linkToPackedPF_)
    iEvent.getByToken(pf2pc_, pf2pc);

  Handle<edm::Association<pat::PackedCandidateCollection>> trk2lost;
  if (linkToLostTracks_)
    iEvent.getByToken(trk2lost_, trk2lost);

  auto out = std::make_unique<std::vector<pat::Tau>>();
  out->reserve(src->size());

  if (modifyTau_) {
    tauModifier_->setEvent(iEvent);
  }

  for (View<pat::Tau>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
    out->push_back(*it);
    pat::Tau &tau = out->back();

    if (modifyTau_) {
      tauModifier_->modify(tau);
    }

    // clearing the pat isolation which is not used by taus
    tau.isolations_.clear();
    tau.isoDeposits_.clear();

    if (linkToPackedPF_) {
      reco::CandidatePtrVector signalChHPtrs, signalNHPtrs, signalGammaPtrs, isolationChHPtrs, isolationNHPtrs,
          isolationGammaPtrs;

      for (const reco::PFCandidatePtr &p : tau.signalPFChargedHadrCands()) {
        signalChHPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
      }
      tau.setSignalChargedHadrCands(signalChHPtrs);

      for (const reco::PFCandidatePtr &p : tau.signalPFNeutrHadrCands()) {
        signalNHPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
      }
      tau.setSignalNeutralHadrCands(signalNHPtrs);

      for (const reco::PFCandidatePtr &p : tau.signalPFGammaCands()) {
        signalGammaPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
      }
      tau.setSignalGammaCands(signalGammaPtrs);

      for (const reco::PFCandidatePtr &p : tau.isolationPFChargedHadrCands()) {
        isolationChHPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
      }
      tau.setIsolationChargedHadrCands(isolationChHPtrs);

      for (const reco::PFCandidatePtr &p : tau.isolationPFNeutrHadrCands()) {
        isolationNHPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
      }
      tau.setIsolationNeutralHadrCands(isolationNHPtrs);

      for (const reco::PFCandidatePtr &p : tau.isolationPFGammaCands()) {
        isolationGammaPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
      }
      tau.setIsolationGammaCands(isolationGammaPtrs);
    }
    if (linkToLostTracks_ && !tau.signalTracks().empty()) {
      std::vector<reco::CandidatePtr> signalLostTracks;
      for (const auto &trkRef : tau.signalTracks()) {
        const auto &lostCandRef = (*trk2lost)[edm::refToPtr(trkRef)];
        if (lostCandRef.isNonnull())
          signalLostTracks.push_back(edm::refToPtr(lostCandRef));
      }
      tau.setSignalLostTracks(signalLostTracks);
    }

    if (dropPiZeroRefs_) {
      tau.pfSpecific_[0].signalPiZeroCandidates_.clear();
      tau.pfSpecific_[0].isolationPiZeroCandidates_.clear();
    }
    if (dropTauChargedHadronRefs_) {
      tau.pfSpecific_[0].signalTauChargedHadronCandidates_.clear();
      tau.pfSpecific_[0].isolationTauChargedHadronCandidates_.clear();
    }
    if (dropPFSpecific_) {
      tau.pfSpecific_.clear();
    }
  }

  iEvent.put(std::move(out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATTauSlimmer);
