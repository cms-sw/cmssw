/**
  \class    PATLeptonTimeLifeInfoProducer
  \brief    Produces lepton life-time information

  \author   Michal Bluj, NCBJ, Warsaw
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/TrackTimeLifeInfo.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <cstring>

template <typename T>
class PATLeptonTimeLifeInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit PATLeptonTimeLifeInfoProducer(const edm::ParameterSet&);
  ~PATLeptonTimeLifeInfoProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //--- private utility methods
  const reco::Track* getTrack(const T&);
  void produceAndFillIPInfo(const T&, const TransientTrackBuilder&, const reco::Vertex&, TrackTimeLifeInfo&);
  void produceAndFillSVInfo(const T&, const TransientTrackBuilder&, const reco::Vertex&, TrackTimeLifeInfo&);
  static bool fitVertex(const std::vector<reco::TransientTrack>& transTrk, TransientVertex& transVtx) {
    if (transTrk.size() < 2)
      return false;
    KalmanVertexFitter kvf(true);
    transVtx = kvf.vertex(transTrk);
    return transVtx.hasRefittedTracks() && transVtx.refittedTracks().size() == transTrk.size();
  }

  //--- configuration parameters
  edm::EDGetTokenT<std::vector<T>> leptonsToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transTrackBuilderToken_;
  const StringCutObjectSelector<T> selector_;
  int pvChoice_;

  enum PVChoice { useFront = 0, useClosestInDz };
  //--- value map for TrackTimeLifeInfo (to be stored into the event)
  using TrackTimeLifeInfoMap = edm::ValueMap<TrackTimeLifeInfo>;
};

template <typename T>
PATLeptonTimeLifeInfoProducer<T>::PATLeptonTimeLifeInfoProducer(const edm::ParameterSet& cfg)
    : leptonsToken_(consumes<std::vector<T>>(cfg.getParameter<edm::InputTag>("src"))),
      pvToken_(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("pvSource"))),
      transTrackBuilderToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      selector_(cfg.getParameter<std::string>("selection")),
      pvChoice_(cfg.getParameter<int>("pvChoice")) {
  produces<TrackTimeLifeInfoMap>();
}

template <typename T>
void PATLeptonTimeLifeInfoProducer<T>::produce(edm::Event& evt, const edm::EventSetup& es) {
  // Get leptons
  edm::Handle<std::vector<T>> leptons;
  evt.getByToken(leptonsToken_, leptons);

  // Get the vertices
  edm::Handle<reco::VertexCollection> vertices;
  evt.getByToken(pvToken_, vertices);

  // Get transient track builder
  const TransientTrackBuilder& transTrackBuilder = es.getData(transTrackBuilderToken_);

  std::vector<TrackTimeLifeInfo> infos;
  infos.reserve(leptons->size());

  for (const auto& lepton : *leptons) {
    TrackTimeLifeInfo info;

    // Do nothing for lepton not passing selection
    if (!selector_(lepton)) {
      infos.push_back(info);
      continue;
    }
    size_t pv_idx = 0;
    if (pvChoice_ == useClosestInDz && getTrack(lepton) != nullptr) {
      float dz_min = 999;
      size_t vtx_idx = 0;
      for (const auto& vtx : *vertices) {
        float dz_tmp = std::abs(getTrack(lepton)->dz(vtx.position()));
        if (dz_tmp < dz_min) {
          dz_min = dz_tmp;
          pv_idx = vtx_idx;
        }
        vtx_idx++;
      }
    }
    const reco::Vertex& pv = !vertices->empty() ? (*vertices)[pv_idx] : reco::Vertex();

    // Obtain IP vector and set related info into lepton
    produceAndFillIPInfo(lepton, transTrackBuilder, pv, info);

    // Fit SV and set related info for taus or do nothing for other lepton types
    produceAndFillSVInfo(lepton, transTrackBuilder, pv, info);
    infos.push_back(info);
  }  // end of lepton loop

  // Build the valuemap
  auto infoMap = std::make_unique<TrackTimeLifeInfoMap>();
  TrackTimeLifeInfoMap::Filler filler(*infoMap);
  filler.insert(leptons, infos.begin(), infos.end());
  filler.fill();

  // Store output into the event
  evt.put(std::move(infoMap));
}

template <>
const reco::Track* PATLeptonTimeLifeInfoProducer<pat::Electron>::getTrack(const pat::Electron& electron) {
  return electron.gsfTrack().isNonnull() ? electron.gsfTrack().get() : nullptr;
}

template <>
const reco::Track* PATLeptonTimeLifeInfoProducer<pat::Muon>::getTrack(const pat::Muon& muon) {
  return muon.innerTrack().isNonnull() ? muon.innerTrack().get() : nullptr;
}

template <>
const reco::Track* PATLeptonTimeLifeInfoProducer<pat::Tau>::getTrack(const pat::Tau& tau) {
  const reco::Track* track = nullptr;
  if (tau.leadChargedHadrCand().isNonnull())
    track = tau.leadChargedHadrCand()->bestTrack();
  return track;
}

template <typename T>
void PATLeptonTimeLifeInfoProducer<T>::produceAndFillIPInfo(const T& lepton,
                                                            const TransientTrackBuilder& transTrackBuilder,
                                                            const reco::Vertex& pv,
                                                            TrackTimeLifeInfo& info) {
  const reco::Track* track = getTrack(lepton);
  if (track != nullptr) {
    // Extrapolate track to the point closest to PV
    reco::TransientTrack transTrack = transTrackBuilder.build(track);
    AnalyticalImpactPointExtrapolator extrapolator(transTrack.field());
    TrajectoryStateOnSurface closestState =
        extrapolator.extrapolate(transTrack.impactPointState(), RecoVertex::convertPos(pv.position()));
    if (!closestState.isValid()) {
      edm::LogError("PATLeptonTimeLifeInfoProducer")
          << "closestState not valid! From:\n"
          << "transTrack.impactPointState():\n"
          << transTrack.impactPointState() << "RecoVertex::convertPos(pv.position()):\n"
          << RecoVertex::convertPos(pv.position());
      return;
    }
    GlobalPoint pca = closestState.globalPosition();
    GlobalError pca_cov = closestState.cartesianError().position();
    GlobalVector ip_vec = GlobalVector(pca.x() - pv.x(), pca.y() - pv.y(), pca.z() - pv.z());
    GlobalError ip_cov = pca_cov + GlobalError(pv.covariance());
    VertexDistance3D pca_dist;
    Measurement1D ip_mes = pca_dist.distance(pv, VertexState(pca, pca_cov));
    if (ip_vec.dot(GlobalVector(lepton.px(), lepton.py(), lepton.pz())) < 0)
      ip_mes = Measurement1D(-1. * ip_mes.value(), ip_mes.error());

    // Store Track and PCA info
    info.setTrack(track);
    info.setBField_z(transTrackBuilder.field()->inInverseGeV(GlobalPoint(track->vx(), track->vy(), track->vz())).z());
    info.setPCA(pca, pca_cov);
    info.setIP(ip_vec, ip_cov);
    info.setIPLength(ip_mes);
  }
}

template <typename T>
void PATLeptonTimeLifeInfoProducer<T>::produceAndFillSVInfo(const T& lepton,
                                                            const TransientTrackBuilder& transTrackBuilder,
                                                            const reco::Vertex& pv,
                                                            TrackTimeLifeInfo& info) {}

template <>
void PATLeptonTimeLifeInfoProducer<pat::Tau>::produceAndFillSVInfo(const pat::Tau& tau,
                                                                   const TransientTrackBuilder& transTrackBuilder,
                                                                   const reco::Vertex& pv,
                                                                   TrackTimeLifeInfo& info) {
  // Fit SV with tracks of charged tau decay products
  int fitOK = 0;
  if (tau.signalChargedHadrCands().size() + tau.signalLostTracks().size() > 1) {
    // Get tracks from tau signal charged candidates
    std::vector<reco::TransientTrack> transTrks;
    TransientVertex transVtx;
    for (const auto& cand : tau.signalChargedHadrCands()) {
      if (cand.isNull())
        continue;
      const reco::Track* track = cand->bestTrack();
      if (track != nullptr)
        transTrks.push_back(transTrackBuilder.build(track));
    }
    for (const auto& cand : tau.signalLostTracks()) {
      if (cand.isNull())
        continue;
      const reco::Track* track = cand->bestTrack();
      if (track != nullptr)
        transTrks.push_back(transTrackBuilder.build(track));
    }
    // Fit SV with KalmanVertexFitter
    fitOK = fitVertex(transTrks, transVtx) ? 1 : -1;
    if (fitOK > 0) {
      reco::Vertex sv = transVtx;
      // Get flight-length
      // Full PV->SV flight vector with its covariance
      GlobalVector flight_vec = GlobalVector(sv.x() - pv.x(), sv.y() - pv.y(), sv.z() - pv.z());
      GlobalError flight_cov = transVtx.positionError() + GlobalError(pv.covariance());
      //MB: can be taken from tau itself (but with different fit of PV and SV) as follows:
      //tau.flightLength().mag2());
      //tau.flightLengthSig();
      VertexDistance3D sv_dist;
      Measurement1D flightLength_mes = sv_dist.signedDistance(pv, sv, GlobalVector(tau.px(), tau.py(), tau.pz()));

      // Store SV info
      info.setSV(sv);
      info.setFlightVector(flight_vec, flight_cov);
      info.setFlightLength(flightLength_mes);
    }
  }
}

template <typename T>
void PATLeptonTimeLifeInfoProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pat{Electron,Muon,Tau}TimeLifeInfoProducer
  edm::ParameterSetDescription desc;

  std::string lepCollName;
  if (typeid(T) == typeid(pat::Electron))
    lepCollName = "slimmedElectrons";
  else if (typeid(T) == typeid(pat::Muon))
    lepCollName = "slimmedMuons";
  else if (typeid(T) == typeid(pat::Tau))
    lepCollName = "slimmedTaus";
  desc.add<edm::InputTag>("src", edm::InputTag(lepCollName));
  desc.add<edm::InputTag>("pvSource", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<std::string>("selection", "")->setComment("Selection required to produce and store time-life information");
  desc.add<int>("pvChoice", useFront)
      ->setComment(
          "Define PV to compute IP: 0: first PV, 1: PV with the smallest dz of the tau leading track (default: " +
          std::to_string(useFront) + ")");

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
typedef PATLeptonTimeLifeInfoProducer<pat::Electron> PATElectronTimeLifeInfoProducer;
DEFINE_FWK_MODULE(PATElectronTimeLifeInfoProducer);
typedef PATLeptonTimeLifeInfoProducer<pat::Muon> PATMuonTimeLifeInfoProducer;
DEFINE_FWK_MODULE(PATMuonTimeLifeInfoProducer);
typedef PATLeptonTimeLifeInfoProducer<pat::Tau> PATTauTimeLifeInfoProducer;
DEFINE_FWK_MODULE(PATTauTimeLifeInfoProducer);
