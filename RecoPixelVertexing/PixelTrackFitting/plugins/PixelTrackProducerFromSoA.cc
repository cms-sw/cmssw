#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/FitUtils.h"

#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"

#include "storeTracks.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"

/**
 * This class creates "leagcy"  reco::Track
 * objects from the output of SoA CA. 
 */
class PixelTrackProducerFromSoA : public edm::global::EDProducer<> {
public:
  using IndToEdm = std::vector<uint16_t>;

  explicit PixelTrackProducerFromSoA(const edm::ParameterSet &iConfig);
  ~PixelTrackProducerFromSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  //  using HitModuleStart = std::array<uint32_t, gpuClustering::maxNumModules + 1>;
  using HMSstorage = HostProduct<uint32_t[]>;

private:
  void produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

  // Event Data tokens
  const edm::EDGetTokenT<reco::BeamSpot> tBeamSpot_;
  const edm::EDGetTokenT<PixelTrackHeterogeneous> tokenTrack_;
  const edm::EDGetTokenT<SiPixelRecHitCollectionNew> cpuHits_;
  const edm::EDGetTokenT<HMSstorage> hmsToken_;
  // Event Setup tokens
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttTopoToken_;

  int32_t const minNumberOfHits_;
  pixelTrack::Quality const minQuality_;
};

PixelTrackProducerFromSoA::PixelTrackProducerFromSoA(const edm::ParameterSet &iConfig)
    : tBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      tokenTrack_(consumes<PixelTrackHeterogeneous>(iConfig.getParameter<edm::InputTag>("trackSrc"))),
      cpuHits_(consumes<SiPixelRecHitCollectionNew>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      hmsToken_(consumes<HMSstorage>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      idealMagneticFieldToken_(esConsumes()),
      ttTopoToken_(esConsumes()),
      minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
      minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))) {
  if (minQuality_ == pixelTrack::Quality::notQuality) {
    throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
  }
  if (minQuality_ < pixelTrack::Quality::dup) {
    throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " not supported";
  }
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<IndToEdm>();
}

void PixelTrackProducerFromSoA::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("trackSrc", edm::InputTag("pixelTracksSoA"));
  desc.add<edm::InputTag>("pixelRecHitLegacySrc", edm::InputTag("siPixelRecHitsPreSplittingLegacy"));
  desc.add<int>("minNumberOfHits", 0);
  desc.add<std::string>("minQuality", "loose");
  descriptions.addWithDefaultLabel(desc);
}

void PixelTrackProducerFromSoA::produce(edm::StreamID streamID,
                                        edm::Event &iEvent,
                                        const edm::EventSetup &iSetup) const {
  // enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity };
  reco::TrackBase::TrackQuality recoQuality[] = {reco::TrackBase::undefQuality,
                                                 reco::TrackBase::undefQuality,
                                                 reco::TrackBase::discarded,
                                                 reco::TrackBase::loose,
                                                 reco::TrackBase::tight,
                                                 reco::TrackBase::tight,
                                                 reco::TrackBase::highPurity};
  assert(reco::TrackBase::highPurity == recoQuality[int(pixelTrack::Quality::highPurity)]);

  // std::cout << "Converting gpu helix in reco tracks" << std::endl;

  auto indToEdmP = std::make_unique<IndToEdm>();
  auto &indToEdm = *indToEdmP;

  auto const &idealField = iSetup.getData(idealMagneticFieldToken_);

  pixeltrackfitting::TracksWithRecHits tracks;

  auto const &httopo = iSetup.getData(ttTopoToken_);

  const auto &bsh = iEvent.get(tBeamSpot_);
  GlobalPoint bs(bsh.x0(), bsh.y0(), bsh.z0());

  auto const &rechits = iEvent.get(cpuHits_);
  std::vector<TrackingRecHit const *> hitmap;
  auto const &rcs = rechits.data();
  auto nhits = rcs.size();
  hitmap.resize(nhits, nullptr);

  auto const *hitsModuleStart = iEvent.get(hmsToken_).get();
  auto fc = hitsModuleStart;

  for (auto const &h : rcs) {
    auto const &thit = static_cast<BaseTrackerRecHit const &>(h);
    auto detI = thit.det()->index();
    auto const &clus = thit.firstClusterRef();
    assert(clus.isPixel());
    auto i = fc[detI] + clus.pixelCluster().originalId();
    if (i >= hitmap.size())
      hitmap.resize(i + 256, nullptr);  // only in case of hit overflow in one module
    assert(nullptr == hitmap[i]);
    hitmap[i] = &h;
  }

  std::vector<const TrackingRecHit *> hits;
  hits.reserve(5);

  const auto &tsoa = *iEvent.get(tokenTrack_);

  auto const *quality = tsoa.qualityData();
  auto const &fit = tsoa.stateAtBS;
  auto const &hitIndices = tsoa.hitIndices;
  auto maxTracks = tsoa.stride();

  tracks.reserve(maxTracks);

  int32_t nt = 0;

  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    indToEdm.push_back(-1);
    auto q = quality[it];
    if (q < minQuality_)
      continue;
    if (nHits < minNumberOfHits_)
      continue;
    indToEdm.back() = nt;
    ++nt;

    hits.resize(nHits);
    auto b = hitIndices.begin(it);
    for (int iHit = 0; iHit < nHits; ++iHit)
      hits[iHit] = hitmap[*(b + iHit)];

    // mind: this values are respect the beamspot!

    float chi2 = tsoa.chi2(it);
    float phi = tsoa.phi(it);

    riemannFit::Vector5d ipar, opar;
    riemannFit::Matrix5d icov, ocov;
    fit.copyToDense(ipar, icov, it);
    riemannFit::transformToPerigeePlane(ipar, icov, opar, ocov);

    LocalTrajectoryParameters lpar(opar(0), opar(1), opar(2), opar(3), opar(4), 1.);
    AlgebraicSymMatrix55 m;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        m(i, j) = ocov(i, j);

    float sp = std::sin(phi);
    float cp = std::cos(phi);
    Surface::RotationType rot(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);

    Plane impPointPlane(bs, rot);
    GlobalTrajectoryParameters gp(
        impPointPlane.toGlobal(lpar.position()), impPointPlane.toGlobal(lpar.momentum()), lpar.charge(), &idealField);
    JacobianLocalToCurvilinear jl2c(impPointPlane, lpar, idealField);

    AlgebraicSymMatrix55 mo = ROOT::Math::Similarity(jl2c.jacobian(), m);

    int ndof = 2 * hits.size() - 5;
    chi2 = chi2 * ndof;
    GlobalPoint vv = gp.position();
    math::XYZPoint pos(vv.x(), vv.y(), vv.z());
    GlobalVector pp = gp.momentum();
    math::XYZVector mom(pp.x(), pp.y(), pp.z());

    auto track = std::make_unique<reco::Track>(chi2, ndof, pos, mom, gp.charge(), CurvilinearTrajectoryError(mo));

    // bad and edup not supported as fit not present or not reliable
    auto tkq = recoQuality[int(q)];
    track->setQuality(tkq);
    // loose,tight and HP are inclusive
    if (reco::TrackBase::highPurity == tkq) {
      track->setQuality(reco::TrackBase::tight);
      track->setQuality(reco::TrackBase::loose);
    } else if (reco::TrackBase::tight == tkq) {
      track->setQuality(reco::TrackBase::loose);
    }
    track->setQuality(tkq);
    // filter???
    tracks.emplace_back(track.release(), hits);
  }
  // std::cout << "processed " << nt << " good tuples " << tracks.size() << "out of " << indToEdm.size() << std::endl;

  // store tracks
  storeTracks(iEvent, tracks, httopo);
  iEvent.put(std::move(indToEdmP));
}

DEFINE_FWK_MODULE(PixelTrackProducerFromSoA);
