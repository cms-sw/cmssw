#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/PatUtils/interface/MiniIsolation.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

namespace pat {

  template <typename T>
  class LeptonUpdater : public edm::global::EDProducer<> {
  public:
    explicit LeptonUpdater(const edm::ParameterSet &iConfig)
        : src_(consumes<std::vector<T>>(iConfig.getParameter<edm::InputTag>("src"))),
          vertices_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertices"))),
          beamLineToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamspot"))),
          computeMiniIso_(iConfig.getParameter<bool>("computeMiniIso")),
          fixDxySign_(iConfig.getParameter<bool>("fixDxySign")) {
      //for mini-isolation calculation
      if (computeMiniIso_) {
        readMiniIsoParams(iConfig);
        pcToken_ = consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandsForMiniIso"));
      }
      recomputeMuonBasicSelectors_ = false;
      if (typeid(T) == typeid(pat::Muon))
        recomputeMuonBasicSelectors_ = iConfig.getParameter<bool>("recomputeMuonBasicSelectors");
      produces<std::vector<T>>();
    }

    ~LeptonUpdater() override {}

    void produce(edm::StreamID, edm::Event &, edm::EventSetup const &) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src")->setComment("Lepton collection");
      desc.add<edm::InputTag>("vertices")->setComment("Vertex collection");
      desc.add<edm::InputTag>("beamspot", edm::InputTag("offlineBeamSpot"))->setComment("Beam spot");
      desc.add<bool>("computeMiniIso", false)->setComment("Recompute miniIsolation");
      desc.add<bool>("fixDxySign", false)->setComment("Fix the IP sign");
      desc.addOptional<edm::InputTag>("pfCandsForMiniIso", edm::InputTag("packedPFCandidates"))
          ->setComment("PackedCandidate collection used for miniIso");
      if (typeid(T) == typeid(pat::Muon)) {
        desc.add<bool>("recomputeMuonBasicSelectors", false)
            ->setComment("Recompute basic cut-based muon selector flags");
        desc.addOptional<std::vector<double>>("miniIsoParams")
            ->setComment("Parameters used for miniIso (as in PATMuonProducer)");
        descriptions.add("muonsUpdated", desc);
      } else if (typeid(T) == typeid(pat::Electron)) {
        desc.addOptional<std::vector<double>>("miniIsoParamsB")
            ->setComment("Parameters used for miniIso in the barrel (as in PATElectronProducer)");
        desc.addOptional<std::vector<double>>("miniIsoParamsE")
            ->setComment("Parameters used for miniIso in the endcap (as in PATElectronProducer)");
        descriptions.add("electronsUpdated", desc);
      }
    }

    void setDZ(T &lep, const reco::Vertex &pv) const {}

    void readMiniIsoParams(const edm::ParameterSet &iConfig) {
      miniIsoParams_[0] = iConfig.getParameter<std::vector<double>>("miniIsoParams");
      if (miniIsoParams_[0].size() != 9)
        throw cms::Exception("ParameterError", "miniIsoParams must have exactly 9 elements.\n");
    }
    const std::vector<double> &miniIsoParams(const T &lep) const { return miniIsoParams_[0]; }

    void recomputeMuonBasicSelectors(T &, const reco::Vertex &, const bool) const;

  private:
    // configurables
    edm::EDGetTokenT<std::vector<T>> src_;
    edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_;
    edm::EDGetTokenT<reco::BeamSpot> beamLineToken_;
    bool computeMiniIso_;
    bool fixDxySign_;
    bool recomputeMuonBasicSelectors_;
    std::vector<double> miniIsoParams_[2];
    edm::EDGetTokenT<pat::PackedCandidateCollection> pcToken_;
  };

  // must do the specialization within the namespace otherwise gcc complains
  //
  template <>
  void LeptonUpdater<pat::Electron>::setDZ(pat::Electron &anElectron, const reco::Vertex &pv) const {
    auto track = anElectron.gsfTrack();
    anElectron.setDB(track->dz(pv.position()), std::hypot(track->dzError(), pv.zError()), pat::Electron::PVDZ);
  }

  template <>
  void LeptonUpdater<pat::Muon>::setDZ(pat::Muon &aMuon, const reco::Vertex &pv) const {
    auto track = aMuon.muonBestTrack();
    aMuon.setDB(track->dz(pv.position()), std::hypot(track->dzError(), pv.zError()), pat::Muon::PVDZ);
  }

  template <>
  void LeptonUpdater<pat::Electron>::readMiniIsoParams(const edm::ParameterSet &iConfig) {
    miniIsoParams_[0] = iConfig.getParameter<std::vector<double>>("miniIsoParamsB");
    miniIsoParams_[1] = iConfig.getParameter<std::vector<double>>("miniIsoParamsE");
    if (miniIsoParams_[0].size() != 9)
      throw cms::Exception("ParameterError", "miniIsoParamsB must have exactly 9 elements.\n");
    if (miniIsoParams_[1].size() != 9)
      throw cms::Exception("ParameterError", "miniIsoParamsE must have exactly 9 elements.\n");
  }
  template <>
  const std::vector<double> &LeptonUpdater<pat::Electron>::miniIsoParams(const pat::Electron &lep) const {
    return miniIsoParams_[lep.isEE()];
  }

  template <typename T>
  void LeptonUpdater<T>::recomputeMuonBasicSelectors(T &lep,
                                                     const reco::Vertex &pv,
                                                     const bool do_hip_mitigation_2016) const {}

  template <>
  void LeptonUpdater<pat::Muon>::recomputeMuonBasicSelectors(pat::Muon &lep,
                                                             const reco::Vertex &pv,
                                                             const bool do_hip_mitigation_2016) const {
    lep.setSelectors(muon::makeSelectorBitset(lep, &pv, do_hip_mitigation_2016));
  }

}  // namespace pat

template <typename T>
void pat::LeptonUpdater<T>::produce(edm::StreamID, edm::Event &iEvent, edm::EventSetup const &) const {
  edm::Handle<std::vector<T>> src;
  iEvent.getByToken(src_, src);

  edm::Handle<std::vector<reco::Vertex>> vertices;
  iEvent.getByToken(vertices_, vertices);
  const reco::Vertex &pv = vertices->front();

  edm::Handle<pat::PackedCandidateCollection> pc;
  if (computeMiniIso_)
    iEvent.getByToken(pcToken_, pc);

  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken(beamLineToken_, beamSpotHandle);
  reco::BeamSpot beamSpot;
  bool beamSpotIsValid = false;
  if (beamSpotHandle.isValid()) {
    beamSpot = *beamSpotHandle;
    beamSpotIsValid = true;
  } else {
    edm::LogError("DataNotAvailable") << "No beam spot available  \n";
  }

  std::unique_ptr<std::vector<T>> out(new std::vector<T>(*src));

  const bool do_hip_mitigation_2016 =
      recomputeMuonBasicSelectors_ && (272728 <= iEvent.run() && iEvent.run() <= 278808);

  for (unsigned int i = 0, n = src->size(); i < n; ++i) {
    T &lep = (*out)[i];
    setDZ(lep, pv);
    if (computeMiniIso_) {
      const auto &params = miniIsoParams(lep);
      pat::PFIsolation miniiso = pat::getMiniPFIsolation(pc.product(),
                                                         lep.polarP4(),
                                                         params[0],
                                                         params[1],
                                                         params[2],
                                                         params[3],
                                                         params[4],
                                                         params[5],
                                                         params[6],
                                                         params[7],
                                                         params[8]);
      lep.setMiniPFIsolation(miniiso);
    }
    if (recomputeMuonBasicSelectors_)
      recomputeMuonBasicSelectors(lep, pv, do_hip_mitigation_2016);
    //Fixing the sign of impact parameters
    if (fixDxySign_) {
      float signPV = 1.;
      float signBS = 1.;
      if (beamSpotIsValid) {
        if constexpr (std::is_same_v<T, pat::Electron>)
          signBS = copysign(1., lep.gsfTrack()->dxy(beamSpot));
        else
          signBS = copysign(1., lep.bestTrack()->dxy(beamSpot));
      }
      if constexpr (std::is_same_v<T, pat::Electron>)
        signPV = copysign(1., lep.gsfTrack()->dxy(pv.position()));
      else
        signPV = copysign(1., lep.bestTrack()->dxy(pv.position()));
      lep.setDB(abs(lep.dB(T::PV2D)) * signPV, lep.edB(T::PV2D), T::PV2D);
      lep.setDB(abs(lep.dB(T::BS2D)) * signBS, lep.edB(T::BS2D), T::BS2D);
    }
  }

  iEvent.put(std::move(out));
}

typedef pat::LeptonUpdater<pat::Electron> PATElectronUpdater;
typedef pat::LeptonUpdater<pat::Muon> PATMuonUpdater;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATElectronUpdater);
DEFINE_FWK_MODULE(PATMuonUpdater);
