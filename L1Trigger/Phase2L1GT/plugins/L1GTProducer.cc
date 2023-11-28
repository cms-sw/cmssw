#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "L1Trigger/Phase2L1GT/interface/L1GTScales.h"

#include "DataFormats/L1Trigger/interface/TkJetWord.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TParticleFlow/interface/gt_datatypes.h"

#include "DataFormats/L1Trigger/interface/EtSum.h"

#include <vector>
#include <array>
#include <string>
#include <type_traits>

namespace l1t {

  class L1GTProducer : public edm::global::EDProducer<> {
  public:
    explicit L1GTProducer(const edm::ParameterSet &);
    ~L1GTProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions &);

  private:
    void produceGTTPromptJets(edm::Event &event) const;
    void produceGTTDisplacedJets(edm::Event &event) const;
    void produceGTTPrimaryVert(edm::Event &event) const;

    void produceGMTSaPromptMuons(edm::Event &event) const;
    void produceGMTSaDisplacedMuons(edm::Event &event) const;
    void produceGMTTkMuons(edm::Event &event) const;

    void produceCL2Jets(edm::Event &event) const;
    void produceCL2Photons(edm::Event &event) const;
    void produceCL2Electrons(edm::Event &event) const;
    void produceCL2Taus(edm::Event &event) const;
    void produceCL2EtSum(edm::Event &event) const;
    void produceCl2HtSum(edm::Event &event) const;

    void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

    const L1GTScales scales_;

    const edm::EDGetTokenT<TkJetWordCollection> gttPromptJetToken_;
    const edm::EDGetTokenT<TkJetWordCollection> gttDisplacedJetToken_;
    const edm::EDGetTokenT<VertexWordCollection> gttPrimaryVertexToken_;

    const edm::EDGetTokenT<SAMuonCollection> gmtSaPromptMuonToken_;
    const edm::EDGetTokenT<SAMuonCollection> gmtSaDisplacedMuonToken_;
    const edm::EDGetTokenT<TrackerMuonCollection> gmtTkMuonToken_;

    const edm::EDGetTokenT<PFJetCollection> cl2JetToken_;
    const edm::EDGetTokenT<TkEmCollection> cl2PhotonToken_;
    const edm::EDGetTokenT<TkElectronCollection> cl2ElectronToken_;
    const edm::EDGetTokenT<PFTauCollection> cl2TauToken_;
    const edm::EDGetTokenT<std::vector<l1t::EtSum>> cl2EtSumToken_;
    const edm::EDGetTokenT<std::vector<l1t::EtSum>> cl2HtSumToken_;
  };

  L1GTProducer::L1GTProducer(const edm::ParameterSet &config)
      : scales_(config.getParameter<edm::ParameterSet>("scales")),
        gttPromptJetToken_(consumes<TkJetWordCollection>(config.getParameter<edm::InputTag>("GTTPromptJets"))),
        gttDisplacedJetToken_(consumes<TkJetWordCollection>(config.getParameter<edm::InputTag>("GTTDisplacedJets"))),
        gttPrimaryVertexToken_(consumes<VertexWordCollection>(config.getParameter<edm::InputTag>("GTTPrimaryVert"))),
        gmtSaPromptMuonToken_(consumes<SAMuonCollection>(config.getParameter<edm::InputTag>("GMTSaPromptMuons"))),
        gmtSaDisplacedMuonToken_(consumes<SAMuonCollection>(config.getParameter<edm::InputTag>("GMTSaDisplacedMuons"))),
        gmtTkMuonToken_(consumes<TrackerMuonCollection>(config.getParameter<edm::InputTag>("GMTTkMuons"))),
        cl2JetToken_(consumes<PFJetCollection>(config.getParameter<edm::InputTag>("CL2Jets"))),
        cl2PhotonToken_(consumes<TkEmCollection>(config.getParameter<edm::InputTag>("CL2Photons"))),
        cl2ElectronToken_(consumes<TkElectronCollection>(config.getParameter<edm::InputTag>("CL2Electrons"))),
        cl2TauToken_(consumes<PFTauCollection>(config.getParameter<edm::InputTag>("CL2Taus"))),
        cl2EtSumToken_(consumes<std::vector<l1t::EtSum>>(config.getParameter<edm::InputTag>("CL2EtSum"))),
        cl2HtSumToken_(consumes<std::vector<l1t::EtSum>>(config.getParameter<edm::InputTag>("CL2HtSum"))) {
    produces<P2GTCandidateCollection>("GTTPromptJets");
    produces<P2GTCandidateCollection>("GTTDisplacedJets");
    produces<P2GTCandidateCollection>("GTTPrimaryVert");

    produces<P2GTCandidateCollection>("GMTSaPromptMuons");
    produces<P2GTCandidateCollection>("GMTSaDisplacedMuons");
    produces<P2GTCandidateCollection>("GMTTkMuons");

    produces<P2GTCandidateCollection>("CL2Jets");
    produces<P2GTCandidateCollection>("CL2Photons");
    produces<P2GTCandidateCollection>("CL2Electrons");
    produces<P2GTCandidateCollection>("CL2Taus");
    produces<P2GTCandidateCollection>("CL2EtSum");
    produces<P2GTCandidateCollection>("CL2HtSum");
  }

  void L1GTProducer::fillDescriptions(edm::ConfigurationDescriptions &description) {
    edm::ParameterSetDescription desc;

    edm::ParameterSetDescription scalesDesc;
    L1GTScales::fillPSetDescription(scalesDesc);
    desc.add<edm::ParameterSetDescription>("scales", scalesDesc);

    desc.add<edm::InputTag>("GTTPromptJets");
    desc.add<edm::InputTag>("GTTDisplacedJets");
    desc.add<edm::InputTag>("GTTPrimaryVert");

    desc.add<edm::InputTag>("GMTSaPromptMuons");
    desc.add<edm::InputTag>("GMTSaDisplacedMuons");
    desc.add<edm::InputTag>("GMTTkMuons");

    desc.add<edm::InputTag>("CL2Jets");
    desc.add<edm::InputTag>("CL2Photons");
    desc.add<edm::InputTag>("CL2Electrons");
    desc.add<edm::InputTag>("CL2Taus");
    desc.add<edm::InputTag>("CL2EtSum");
    desc.add<edm::InputTag>("CL2HtSum");

    description.addWithDefaultLabel(desc);
  }

  void L1GTProducer::produceGTTPrimaryVert(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const VertexWordCollection &collection = event.get(gttPrimaryVertexToken_);
    for (std::size_t i = 0; i < collection.size() && i < 10; i++) {
      const VertexWord &obj = collection[i];
      int hwZ0 = obj.z0Word().V.to_int() * 5;
      P2GTCandidate gtObj(
          0, reco::ParticleState::PolarLorentzVector(), reco::ParticleState::Point(0, 0, scales_.to_z0(hwZ0)));
      gtObj.hwZ0_ = hwZ0;
      gtObj.hwQual_ = obj.qualityWord().V.to_int();
      gtObj.hwSum_pT_pv_ = obj.multiplicityWord().V.to_int();
      gtObj.hwNumber_of_tracks_in_pv_ = obj.multiplicityWord().V.to_int();
      gtObj.hwNumber_of_tracks_not_in_pv_ = obj.inverseMultiplicityWord().V.to_int();
      gtObj.objectType_ = P2GTCandidate::GTTPrimaryVert;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "GTTPrimaryVert");
  }

  void L1GTProducer::produceGTTPromptJets(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const TkJetWordCollection &collection = event.get(gttPromptJetToken_);
    for (std::size_t i = 0; i < collection.size() && i < 12; i++) {
      const TkJetWord &obj = collection[i];
      int hwZ0 = obj.z0Word().V.to_int() << 7;
      P2GTCandidate gtObj(0,
                          reco::ParticleState::PolarLorentzVector(scales_.to_pT(obj.ptWord().V.to_int()),
                                                                  scales_.to_eta(obj.glbEtaWord().V.to_int()),
                                                                  scales_.to_phi(obj.glbPhiWord().V.to_int()),
                                                                  0),
                          reco::ParticleState::Point(0, 0, scales_.to_z0(hwZ0)));
      gtObj.hwPT_ = obj.ptWord().V.to_int();
      gtObj.hwPhi_ = obj.glbPhiWord().V.to_int();
      gtObj.hwEta_ = obj.glbEtaWord().V.to_int();
      gtObj.hwZ0_ = hwZ0;
      gtObj.hwNumber_of_tracks_ = obj.ntWord().V.to_int();
      gtObj.objectType_ = P2GTCandidate::GTTPromptJets;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "GTTPromptJets");
  }

  void L1GTProducer::produceGTTDisplacedJets(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const TkJetWordCollection &collection = event.get(gttDisplacedJetToken_);
    for (std::size_t i = 0; i < collection.size() && i < 12; i++) {
      const TkJetWord &obj = collection[i];
      int hwZ0 = obj.z0Word().V.to_int() << 7;
      P2GTCandidate gtObj(0,
                          reco::ParticleState::PolarLorentzVector(scales_.to_pT(obj.ptWord().V.to_int()),
                                                                  scales_.to_eta(obj.glbEtaWord().V.to_int()),
                                                                  scales_.to_phi(obj.glbPhiWord().V.to_int()),
                                                                  0),
                          reco::ParticleState::Point(0, 0, scales_.to_z0(hwZ0)));
      gtObj.hwPT_ = obj.ptWord().V.to_int();
      gtObj.hwPhi_ = obj.glbPhiWord().V.to_int();
      gtObj.hwEta_ = obj.glbEtaWord().V.to_int();
      gtObj.hwZ0_ = hwZ0;
      gtObj.hwNumber_of_tracks_ = obj.ntWord().V.to_int();
      gtObj.objectType_ = P2GTCandidate::GTTDisplacedJets;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "GTTDisplacedJets");
  }

  void L1GTProducer::produceGMTSaPromptMuons(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const SAMuonCollection &collection = event.get(gmtSaPromptMuonToken_);
    for (size_t i = 0; i < collection.size() && i < 12; i++) {
      const SAMuon &obj = collection[i];
      int hwZ0 = obj.apZ0().to_int() << 12;
      P2GTCandidate gtObj(scales_.to_chg(obj.apCharge().to_int()),
                          reco::ParticleState::PolarLorentzVector(scales_.to_pT(obj.apPt().to_int()),
                                                                  scales_.to_eta(obj.apEta().to_int()),
                                                                  scales_.to_phi(obj.apPhi().to_int()),
                                                                  0),
                          reco::ParticleState::Point(0, 0, scales_.to_z0(hwZ0)));
      gtObj.hwPT_ = obj.apPt().to_int();
      gtObj.hwPhi_ = obj.apPhi().to_int();
      gtObj.hwEta_ = obj.apEta().to_int();
      gtObj.hwZ0_ = hwZ0;
      gtObj.hwQual_ = obj.apQual().to_int();
      gtObj.hwCharge_ = obj.apCharge().to_int();
      gtObj.hwD0_ = obj.apD0().to_int();
      gtObj.objectType_ = P2GTCandidate::GMTSaPromptMuons;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "GMTSaPromptMuons");
  }

  void L1GTProducer::produceGMTSaDisplacedMuons(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const SAMuonCollection &collection = event.get(gmtSaDisplacedMuonToken_);
    for (size_t i = 0; i < collection.size() && i < 12; i++) {
      const SAMuon &obj = collection[i];
      int hwZ0 = obj.apZ0().to_int() << 12;
      P2GTCandidate gtObj(scales_.to_chg(obj.apCharge().to_int()),
                          reco::ParticleState::PolarLorentzVector(scales_.to_pT(obj.apPt().to_int()),
                                                                  scales_.to_eta(obj.apEta().to_int()),
                                                                  scales_.to_phi(obj.apPhi().to_int()),
                                                                  0),
                          reco::ParticleState::Point(0, 0, scales_.to_z0(hwZ0)));
      gtObj.hwPT_ = obj.apPt().to_int();
      gtObj.hwPhi_ = obj.apPhi().to_int();
      gtObj.hwEta_ = obj.apEta().to_int();
      gtObj.hwZ0_ = hwZ0;
      gtObj.hwQual_ = obj.apQual().to_int();
      gtObj.hwCharge_ = obj.apCharge().to_int();
      gtObj.hwD0_ = obj.apD0().to_int();
      gtObj.objectType_ = P2GTCandidate::GMTSaDisplacedMuons;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "GMTSaDisplacedMuons");
  }

  void L1GTProducer::produceGMTTkMuons(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const TrackerMuonCollection &collection = event.get(gmtTkMuonToken_);
    for (size_t i = 0; i < collection.size() && i < 12; i++) {
      const TrackerMuon &obj = collection[i];
      int hwZ0 = obj.apZ0().to_int() << 7;
      P2GTCandidate gtObj(scales_.to_chg(obj.apCharge().to_int()),
                          reco::ParticleState::PolarLorentzVector(scales_.to_pT(obj.apPt().to_int()),
                                                                  scales_.to_eta(obj.apEta().to_int()),
                                                                  scales_.to_phi(obj.apPhi().to_int()),
                                                                  0),
                          reco::ParticleState::Point(0, 0, scales_.to_z0(hwZ0)));
      gtObj.hwPT_ = obj.apPt().to_int();
      gtObj.hwPhi_ = obj.apPhi().to_int();
      gtObj.hwEta_ = obj.apEta().to_int();
      gtObj.hwZ0_ = hwZ0;
      gtObj.hwIso_ = obj.apIso().to_int();
      gtObj.hwQual_ = obj.apQual().to_int();
      gtObj.hwCharge_ = obj.apCharge().to_int();
      gtObj.hwD0_ = obj.apD0().to_int();
      gtObj.hwBeta_ = obj.apBeta().to_int();
      gtObj.objectType_ = P2GTCandidate::GMTTkMuons;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "GMTTkMuons");
  }

  void L1GTProducer::produceCL2Jets(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const PFJetCollection &collection = event.get(cl2JetToken_);
    for (size_t i = 0; i < collection.size() && i < 12; i++) {
      l1gt::Jet gtJet = l1gt::Jet::unpack(collection[i].getHWJetGT());
      int hwZ0 = gtJet.z0.V.to_int() << 7;
      P2GTCandidate gtObj(0,
                          reco::ParticleState::PolarLorentzVector(scales_.to_pT(gtJet.v3.pt.V.to_int()),
                                                                  scales_.to_eta(gtJet.v3.eta.V.to_int()),
                                                                  scales_.to_phi(gtJet.v3.phi.V.to_int()),
                                                                  0),
                          reco::ParticleState::Point(0, 0, scales_.to_z0(hwZ0)));
      gtObj.hwPT_ = gtJet.v3.pt.V.to_int();
      gtObj.hwPhi_ = gtJet.v3.phi.V.to_int();
      gtObj.hwEta_ = gtJet.v3.eta.V.to_int();
      gtObj.hwZ0_ = hwZ0;
      gtObj.objectType_ = P2GTCandidate::CL2Jets;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "CL2Jets");
  }

  void L1GTProducer::produceCL2Photons(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const TkEmCollection &collection = event.get(cl2PhotonToken_);
    for (size_t i = 0; i < collection.size() && i < 12; i++) {
      l1gt::Photon gtPhoton = l1gt::Photon::unpack_ap(const_cast<TkEm &>(collection[i]).egBinaryWord<96>());
      P2GTCandidate gtObj(0,
                          reco::ParticleState::PolarLorentzVector(scales_.to_pT(gtPhoton.v3.pt.V.to_int()),
                                                                  scales_.to_eta(gtPhoton.v3.eta.V.to_int()),
                                                                  scales_.to_phi(gtPhoton.v3.phi.V.to_int()),
                                                                  0));
      gtObj.hwPT_ = gtPhoton.v3.pt.V.to_int();
      gtObj.hwPhi_ = gtPhoton.v3.phi.V.to_int();
      gtObj.hwEta_ = gtPhoton.v3.eta.V.to_int();
      gtObj.hwIso_ = gtPhoton.isolation.V.to_int();
      gtObj.hwQual_ = gtPhoton.quality.V.to_int();
      gtObj.objectType_ = P2GTCandidate::CL2Photons;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "CL2Photons");
  }

  void L1GTProducer::produceCL2Electrons(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const TkElectronCollection &collection = event.get(cl2ElectronToken_);
    for (size_t i = 0; i < collection.size() && i < 12; i++) {
      l1gt::Electron gtElectron = l1gt::Electron::unpack_ap(const_cast<TkElectron &>(collection[i]).egBinaryWord<96>());
      int hwZ0 = gtElectron.z0.V.to_int() << 7;
      P2GTCandidate gtObj(scales_.to_chg(gtElectron.charge.V.to_int()),
                          reco::ParticleState::PolarLorentzVector(scales_.to_pT(gtElectron.v3.pt.V.to_int()),
                                                                  scales_.to_eta(gtElectron.v3.eta.V.to_int()),
                                                                  scales_.to_phi(gtElectron.v3.phi.V.to_int()),
                                                                  0),
                          reco::ParticleState::Point(0, 0, scales_.to_z0(hwZ0)));
      gtObj.hwPT_ = gtElectron.v3.pt.V.to_int();
      gtObj.hwPhi_ = gtElectron.v3.phi.V.to_int();
      gtObj.hwEta_ = gtElectron.v3.eta.V.to_int();
      gtObj.hwZ0_ = hwZ0;
      gtObj.hwIso_ = gtElectron.isolation.V.to_int();
      gtObj.hwQual_ = gtElectron.quality.V.to_int();
      gtObj.hwCharge_ = gtElectron.charge.V.to_int();
      gtObj.objectType_ = P2GTCandidate::CL2Electrons;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "CL2Electrons");
  }

  void L1GTProducer::produceCL2Taus(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const PFTauCollection &collection = event.get(cl2TauToken_);
    for (size_t i = 0; i < collection.size() && i < 12; i++) {
      l1gt::Tau gtTau = collection[i].getHWTauGT();
      P2GTCandidate gtObj(scales_.to_chg(gtTau.charge.V.to_int()),
                          reco::ParticleState::PolarLorentzVector(scales_.to_pT(gtTau.v3.pt.V.to_int()),
                                                                  scales_.to_eta(gtTau.v3.eta.V.to_int()),
                                                                  scales_.to_phi(gtTau.v3.phi.V.to_int()),
                                                                  0));
      gtObj.hwPT_ = gtTau.v3.pt.V.to_int();
      gtObj.hwPhi_ = gtTau.v3.phi.V.to_int();
      gtObj.hwEta_ = gtTau.v3.eta.V.to_int();
      gtObj.hwSeed_pT_ = gtTau.seed_pt.V.to_int();
      gtObj.hwSeed_z0_ = gtTau.seed_z0.V.to_int();
      gtObj.hwCharge_ = gtTau.charge.V.to_int();
      gtObj.hwType_ = gtTau.type.V.to_int();
      gtObj.hwIso_ = gtTau.isolation.V.to_int();
      gtObj.objectType_ = P2GTCandidate::CL2Taus;

      outputCollection->push_back(gtObj);
    }
    event.put(std::move(outputCollection), "CL2Taus");
  }

  void L1GTProducer::produceCL2EtSum(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const std::vector<EtSum> &collection = event.get(cl2EtSumToken_);
    const EtSum &met = collection[0];

    l1gt::Sum sum{true /* valid */, met.pt(), met.phi() / l1gt::Scales::ETAPHI_LSB, 0 /* scalar sum */};

    P2GTCandidate gtObj(0,
                        reco::ParticleState::PolarLorentzVector(
                            scales_.to_pT(sum.vector_pt.V.to_int()), 0, scales_.to_phi(sum.vector_phi.V.to_int()), 0));
    gtObj.hwPT_ = sum.vector_pt.V.to_int();
    gtObj.hwPhi_ = sum.vector_phi.V.to_int();
    gtObj.hwSca_sum_ = sum.scalar_pt.V.to_int();
    gtObj.objectType_ = P2GTCandidate::CL2EtSum;

    outputCollection->push_back(gtObj);
    event.put(std::move(outputCollection), "CL2EtSum");
  }

  void L1GTProducer::produceCl2HtSum(edm::Event &event) const {
    std::unique_ptr<P2GTCandidateCollection> outputCollection = std::make_unique<P2GTCandidateCollection>();
    const std::vector<EtSum> &collection = event.get(cl2HtSumToken_);
    const EtSum &ht = collection[0];
    const EtSum &mht = collection[1];

    P2GTCandidate gtObj(
        0, reco::ParticleState::PolarLorentzVector(scales_.to_pT(mht.hwPt()), 0, scales_.to_phi(mht.hwPhi()), 0));
    gtObj.hwPT_ = mht.hwPt();
    gtObj.hwPhi_ = mht.hwPhi();
    gtObj.hwSca_sum_ = ht.hwPt();
    gtObj.objectType_ = P2GTCandidate::CL2HtSum;

    outputCollection->push_back(gtObj);
    event.put(std::move(outputCollection), "CL2HtSum");
  }

  void L1GTProducer::produce(edm::StreamID, edm::Event &event, const edm::EventSetup &setup) const {
    produceGTTPromptJets(event);
    produceGTTDisplacedJets(event);
    produceGTTPrimaryVert(event);

    produceGMTSaPromptMuons(event);
    produceGMTSaDisplacedMuons(event);
    produceGMTTkMuons(event);

    produceCL2Jets(event);
    produceCL2Photons(event);
    produceCL2Electrons(event);
    produceCL2Taus(event);
    produceCL2EtSum(event);
    produceCl2HtSum(event);
  }
}  // namespace l1t

using namespace l1t;

DEFINE_FWK_MODULE(L1GTProducer);
