#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetTray.h"
#include "RecoMTD/DetLayers/interface/MTDRingForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetRing.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/ForwardDetId/interface/MTDChannelIdentifier.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

#include <sstream>
#include <random>

#include "Geometry/CommonTopologies/interface/Topology.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

using namespace std;
using namespace edm;
using namespace reco;

class MTDHitMatchingInfo {
public:
  MTDHitMatchingInfo() {
    hit = nullptr;
    estChi2 = std::numeric_limits<double>::max();
    timeChi2 = std::numeric_limits<double>::max();
  }

  MTDHitMatchingInfo(const MTDHitMatchingInfo& a) : hit(a.hit), estChi2(a.estChi2), timeChi2(a.timeChi2) {}
  //Operator used to sort the hits while performing the matching step at the MTD
  inline bool operator<(const MTDHitMatchingInfo& m2) const {
    //only for good matching in time use estChi2, otherwise use mostly time compatibility
    constexpr double chi2_cut = 10.;
    constexpr double low_weight = 3.;
    constexpr double high_weight = 8.;
    if (timeChi2 < chi2_cut && m2.timeChi2 < chi2_cut)
      return chi2(low_weight) < m2.chi2(low_weight);
    else
      return chi2(high_weight) < m2.chi2(high_weight);
  }

  inline double chi2(float timeWeight = 1.) const { return estChi2 + timeWeight * timeChi2; }

  const MTDTrackingRecHit* hit;
  double estChi2;
  double timeChi2;
};

template <class TrackCollection>
class TrackExtenderWithMTDT : public edm::stream::EDProducer<> {
public:
  typedef typename TrackCollection::value_type TrackType;
  typedef edm::View<TrackType> InputCollection;

  TrackExtenderWithMTDT(const ParameterSet& pset);

  template <class H, class T>
  void fillValueMap(edm::Event& iEvent, const H& handle, const std::vector<T>& vec, const std::string& name) const;

  void produce(edm::Event& ev, const edm::EventSetup& es) final;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  TransientTrackingRecHit::ConstRecHitContainer tryBTLLayers(const TrackType&,
                                                             const Trajectory& traj,
                                                             const MTDTrackingDetSetVector&,
                                                             const MTDDetLayerGeometry*,
                                                             const MagneticField* field,
                                                             const Propagator* prop,
                                                             const reco::BeamSpot& bs,
                                                             const double& vtxTime,
                                                             const bool matchVertex,
                                                             MTDHitMatchingInfo& bestHit) const;

  TransientTrackingRecHit::ConstRecHitContainer tryETLLayers(const TrackType&,
                                                             const Trajectory& traj,
                                                             const MTDTrackingDetSetVector&,
                                                             const MTDDetLayerGeometry*,
                                                             const MagneticField* field,
                                                             const Propagator* prop,
                                                             const reco::BeamSpot& bs,
                                                             const double& vtxTime,
                                                             const bool matchVertex,
                                                             MTDHitMatchingInfo& bestHit) const;

  RefitDirection::GeometricalDirection checkRecHitsOrdering(
      TransientTrackingRecHit::ConstRecHitContainer const& recHits) const {
    if (!recHits.empty()) {
      GlobalPoint first = gtg->idToDet(recHits.front()->geographicalId())->position();
      GlobalPoint last = gtg->idToDet(recHits.back()->geographicalId())->position();

      // maybe perp2?
      auto rFirst = first.mag2();
      auto rLast = last.mag2();
      if (rFirst < rLast)
        return RefitDirection::insideOut;
      if (rFirst > rLast)
        return RefitDirection::outsideIn;
    }
    LogDebug("TrackExtenderWithMTD") << "Impossible to determine the rechits order" << endl;
    return RefitDirection::undetermined;
  }

  reco::Track buildTrack(const reco::Track&,
                         const Trajectory&,
                         const Trajectory&,
                         const reco::BeamSpot&,
                         const MagneticField* field,
                         const Propagator* prop,
                         bool hasMTD,
                         float& pathLength,
                         float& tmtdOut,
                         float& sigmatmtdOut) const;
  reco::TrackExtra buildTrackExtra(const Trajectory& trajectory) const;

  string dumpLayer(const DetLayer* layer) const;

private:
  static constexpr char btlMatchChi2Name[] = "btlMatchChi2";
  static constexpr char etlMatchChi2Name[] = "etlMatchChi2";
  static constexpr char btlMatchTimeChi2Name[] = "btlMatchTimeChi2";
  static constexpr char etlMatchTimeChi2Name[] = "etlMatchTimeChi2";
  static constexpr char pathLengthName[] = "pathLength";
  static constexpr char tmtdName[] = "tmtd";
  static constexpr char sigmatmtdName[] = "sigmatmtd";
  static constexpr char pOrigTrkName[] = "generalTrackp";
  static constexpr char betaOrigTrkName[] = "generalTrackBeta";
  static constexpr char t0OrigTrkName[] = "generalTrackt0";
  static constexpr char sigmat0OrigTrkName[] = "generalTracksigmat0";
  static constexpr char pathLengthOrigTrkName[] = "generalTrackPathLength";
  static constexpr char tmtdOrigTrkName[] = "generalTracktmtd";
  static constexpr char sigmatmtdOrigTrkName[] = "generalTracksigmatmtd";
  static constexpr char assocOrigTrkName[] = "generalTrackassoc";

  edm::EDGetTokenT<InputCollection> tracksToken_;
  edm::EDGetTokenT<MTDTrackingDetSetVector> hitsToken_;
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  edm::EDGetTokenT<vector<SimVertex>> genVtxToken_;
  edm::EDGetTokenT<VertexCollection> vtxToken_;

  const bool updateTraj_, updateExtra_, updatePattern_;
  const std::string mtdRecHitBuilder_, propagator_, transientTrackBuilder_;
  std::unique_ptr<MeasurementEstimator> theEstimator;
  std::unique_ptr<TrackTransformer> theTransformer;
  edm::ESHandle<TransientTrackBuilder> builder;
  edm::ESHandle<TransientTrackingRecHitBuilder> hitbuilder;
  edm::ESHandle<GlobalTrackingGeometry> gtg;
  edm::ESHandle<Propagator> prop;

  float estMaxChi2_;
  float estMaxNSigma_;
  float btlChi2Cut_;
  float btlTimeChi2Cut_;
  float etlChi2Cut_;
  float etlTimeChi2Cut_;

  bool useVertex_;
  bool useSimVertex_;
  double dzCut_;

  std::default_random_engine randomGenerator_;
};

template <class TrackCollection>
TrackExtenderWithMTDT<TrackCollection>::TrackExtenderWithMTDT(const ParameterSet& iConfig)
    : tracksToken_(consumes<InputCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
      hitsToken_(consumes<MTDTrackingDetSetVector>(iConfig.getParameter<edm::InputTag>("hitsSrc"))),
      bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"))),
      updateTraj_(iConfig.getParameter<bool>("updateTrackTrajectory")),
      updateExtra_(iConfig.getParameter<bool>("updateTrackExtra")),
      updatePattern_(iConfig.getParameter<bool>("updateTrackHitPattern")),
      mtdRecHitBuilder_(iConfig.getParameter<std::string>("MTDRecHitBuilder")),
      propagator_(iConfig.getParameter<std::string>("Propagator")),
      transientTrackBuilder_(iConfig.getParameter<std::string>("TransientTrackBuilder")),
      estMaxChi2_(iConfig.getParameter<double>("EstimatorMaxChi2")),
      estMaxNSigma_(iConfig.getParameter<double>("EstimatorMaxNSigma")),
      btlChi2Cut_(iConfig.getParameter<double>("BTLChi2Cut")),
      btlTimeChi2Cut_(iConfig.getParameter<double>("BTLTimeChi2Cut")),
      etlChi2Cut_(iConfig.getParameter<double>("ETLChi2Cut")),
      etlTimeChi2Cut_(iConfig.getParameter<double>("ETLTimeChi2Cut")),
      useVertex_(iConfig.getParameter<bool>("UseVertex")),
      useSimVertex_(iConfig.getParameter<bool>("UseSimVertex")),
      dzCut_(iConfig.getParameter<double>("DZCut")) {
  if (useVertex_) {
    if (useSimVertex_)
      genVtxToken_ = consumes<vector<SimVertex>>(iConfig.getParameter<edm::InputTag>("genVtxSrc"));
    else
      vtxToken_ = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vtxSrc"));
  }

  theEstimator = std::make_unique<Chi2MeasurementEstimator>(estMaxChi2_, estMaxNSigma_);
  theTransformer = std::make_unique<TrackTransformer>(iConfig.getParameterSet("TrackTransformer"));

  produces<edm::ValueMap<float>>(btlMatchChi2Name);
  produces<edm::ValueMap<float>>(etlMatchChi2Name);
  produces<edm::ValueMap<float>>(btlMatchTimeChi2Name);
  produces<edm::ValueMap<float>>(etlMatchTimeChi2Name);
  produces<edm::ValueMap<float>>(pathLengthName);
  produces<edm::ValueMap<float>>(tmtdName);
  produces<edm::ValueMap<float>>(sigmatmtdName);
  produces<edm::ValueMap<float>>(pOrigTrkName);
  produces<edm::ValueMap<float>>(betaOrigTrkName);
  produces<edm::ValueMap<float>>(t0OrigTrkName);
  produces<edm::ValueMap<float>>(sigmat0OrigTrkName);
  produces<edm::ValueMap<float>>(pathLengthOrigTrkName);
  produces<edm::ValueMap<float>>(tmtdOrigTrkName);
  produces<edm::ValueMap<float>>(sigmatmtdOrigTrkName);
  produces<edm::ValueMap<int>>(assocOrigTrkName);

  produces<edm::OwnVector<TrackingRecHit>>();
  produces<reco::TrackExtraCollection>();
  produces<TrackCollection>();
}

template <class TrackCollection>
void TrackExtenderWithMTDT<TrackCollection>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc, transDesc;
  desc.add<edm::InputTag>("tracksSrc", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("hitsSrc", edm::InputTag("mtdTrackingRecHits"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("genVtxSrc", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("vtxSrc", edm::InputTag("offlinePrimaryVertices4D"));
  desc.add<bool>("updateTrackTrajectory", true);
  desc.add<bool>("updateTrackExtra", true);
  desc.add<bool>("updateTrackHitPattern", true);
  desc.add<std::string>("TransientTrackBuilder", "TransientTrackBuilder");
  desc.add<std::string>("MTDRecHitBuilder", "MTDRecHitBuilder");
  desc.add<std::string>("Propagator", "PropagatorWithMaterialForMTD");
  TrackTransformer::fillPSetDescription(transDesc,
                                        false,
                                        "KFFitterForRefitInsideOut",
                                        "KFSmootherForRefitInsideOut",
                                        "PropagatorWithMaterialForMTD",
                                        "alongMomentum",
                                        true,
                                        "WithTrackAngle",
                                        "MuonRecHitBuilder",
                                        "MTDRecHitBuilder");
  desc.add<edm::ParameterSetDescription>("TrackTransformer", transDesc);
  desc.add<double>("EstimatorMaxChi2", 500.);
  desc.add<double>("EstimatorMaxNSigma", 10.);
  desc.add<double>("BTLChi2Cut", 50.);
  desc.add<double>("BTLTimeChi2Cut", 5.);
  desc.add<double>("ETLChi2Cut", 50.);
  desc.add<double>("ETLTimeChi2Cut", 5.);
  desc.add<bool>("UseVertex", false);
  desc.add<bool>("UseSimVertex", false);
  desc.add<double>("DZCut", 0.1);
  descriptions.add("trackExtenderWithMTDBase", desc);
}

template <class TrackCollection>
template <class H, class T>
void TrackExtenderWithMTDT<TrackCollection>::fillValueMap(edm::Event& iEvent,
                                                          const H& handle,
                                                          const std::vector<T>& vec,
                                                          const std::string& name) const {
  auto out = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(handle, vec.begin(), vec.end());
  filler.fill();
  iEvent.put(std::move(out), name);
}

template <class TrackCollection>
void TrackExtenderWithMTDT<TrackCollection>::produce(edm::Event& ev, const edm::EventSetup& es) {
  //this produces pieces of the track extra
  Traj2TrackHits t2t;

  theTransformer->setServices(es);

  TrackingRecHitRefProd hitsRefProd = ev.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd extrasRefProd = ev.getRefBeforePut<reco::TrackExtraCollection>();

  es.get<GlobalTrackingGeometryRecord>().get(gtg);

  edm::ESHandle<MTDDetLayerGeometry> geo;
  es.get<MTDRecoGeometryRecord>().get(geo);

  edm::ESHandle<MagneticField> magfield;
  es.get<IdealMagneticFieldRecord>().get(magfield);

  es.get<TransientTrackRecord>().get(transientTrackBuilder_, builder);
  es.get<TransientRecHitRecord>().get(mtdRecHitBuilder_, hitbuilder);

  edm::ESHandle<Propagator> prop;
  es.get<TrackingComponentsRecord>().get(propagator_, prop);

  edm::ESHandle<TrackerTopology> httopo;
  es.get<TrackerTopologyRcd>().get(httopo);
  const TrackerTopology& ttopo = *httopo;

  auto output = std::make_unique<TrackCollection>();
  auto extras = std::make_unique<reco::TrackExtraCollection>();
  auto outhits = std::make_unique<edm::OwnVector<TrackingRecHit>>();

  std::vector<float> btlMatchChi2;
  std::vector<float> etlMatchChi2;
  std::vector<float> btlMatchTimeChi2;
  std::vector<float> etlMatchTimeChi2;
  std::vector<float> pathLengthsRaw;
  std::vector<float> tmtdRaw;
  std::vector<float> sigmatmtdRaw;
  std::vector<float> pOrigTrkRaw;
  std::vector<float> betaOrigTrkRaw;
  std::vector<float> t0OrigTrkRaw;
  std::vector<float> sigmat0OrigTrkRaw;
  std::vector<float> pathLengthsOrigTrkRaw;
  std::vector<float> tmtdOrigTrkRaw;
  std::vector<float> sigmatmtdOrigTrkRaw;
  std::vector<int> assocOrigTrkRaw;

  edm::Handle<InputCollection> tracksH;
  ev.getByToken(tracksToken_, tracksH);
  const auto& tracks = *tracksH;

  edm::Handle<MTDTrackingDetSetVector> hitsH;
  ev.getByToken(hitsToken_, hitsH);
  const auto& hits = *hitsH;

  edm::Handle<reco::BeamSpot> bsH;
  ev.getByToken(bsToken_, bsH);
  const auto& bs = *bsH;

  const Vertex* PV = nullptr;
  if (useVertex_ && !useSimVertex_) {
    edm::Handle<VertexCollection> vtxH;
    ev.getByToken(vtxToken_, vtxH);
    if (vtxH.isValid())
      PV = &(vtxH.product()->at(0));
  }

  const SimVertex* genPV = nullptr;
  if (useVertex_ && useSimVertex_) {
    edm::Handle<vector<SimVertex>> genVtxH;
    ev.getByToken(genVtxToken_, genVtxH);
    if (genVtxH.isValid())
      genPV = &(genVtxH.product()->at(0));
  }

  double vtxTime = 0.;
  constexpr double simVtxSmear = 0.008;  //assign 8ps resolution from reco studies
  if (useVertex_) {
    if (useSimVertex_ && genPV) {
      vtxTime = genPV->position().t() * CLHEP::second;  //convert to ns
      std::normal_distribution<double> vtx_smearer(vtxTime, simVtxSmear);
      vtxTime = vtx_smearer(randomGenerator_);
    } else if (PV)
      vtxTime = PV->t();  //already in ns
  }

  std::vector<unsigned> track_indices;
  unsigned itrack = 0;

  for (const auto& track : tracks) {
    double trackVtxTime = 0.;
    if (useVertex_) {
      double dz;
      if (useSimVertex_)
        dz = std::abs(track.dz(math::XYZPoint(genPV->position().x(), genPV->position().y(), genPV->position().z())));
      else
        dz = std::abs(track.dz(math::XYZPoint(PV->position().x(), PV->position().y(), PV->position().z())));

      if (dz < dzCut_)
        trackVtxTime = vtxTime;
    }

    reco::TransientTrack ttrack(track, magfield.product(), gtg);
    const auto& trajs = theTransformer->transform(track);
    auto thits = theTransformer->getTransientRecHits(ttrack);
    TransientTrackingRecHit::ConstRecHitContainer mtdthits;
    MTDHitMatchingInfo mBTL, mETL;
    if (!trajs.empty()) {
      const auto& btlhits = tryBTLLayers(track,
                                         trajs.front(),
                                         hits,
                                         geo.product(),
                                         magfield.product(),
                                         prop.product(),
                                         bs,
                                         trackVtxTime,
                                         trackVtxTime != 0.,
                                         mBTL);
      mtdthits.insert(mtdthits.end(), btlhits.begin(), btlhits.end());

      // in the future this should include an intermediate refit before propagating to the ETL
      // for now it is ok
      const auto& etlhits = tryETLLayers(track,
                                         trajs.front(),
                                         hits,
                                         geo.product(),
                                         magfield.product(),
                                         prop.product(),
                                         bs,
                                         trackVtxTime,
                                         trackVtxTime != 0.,
                                         mETL);
      mtdthits.insert(mtdthits.end(), etlhits.begin(), etlhits.end());
    }

    auto ordering = checkRecHitsOrdering(thits);
    if (ordering == RefitDirection::insideOut) {
      thits.insert(thits.end(), mtdthits.begin(), mtdthits.end());
    } else {
      std::reverse(mtdthits.begin(), mtdthits.end());
      mtdthits.insert(mtdthits.end(), thits.begin(), thits.end());
      thits.swap(mtdthits);
    }

    const auto& trajwithmtd = theTransformer->transform(ttrack, thits);
    float pMap = 0.f, betaMap = 0.f, t0Map = 0.f, sigmat0Map = -1.f, pathLengthMap = -1.f, tmtdMap = 0.f,
          sigmatmtdMap = -1.f;
    int iMap = -1;

    for (const auto& trj : trajwithmtd) {
      const auto& thetrj = (updateTraj_ ? trj : trajs.front());
      float pathLength = 0.f, tmtd = 0.f, sigmatmtd = -1.f;
      reco::Track result = buildTrack(
          track, thetrj, trj, bs, magfield.product(), prop.product(), !mtdthits.empty(), pathLength, tmtd, sigmatmtd);
      if (result.ndof() >= 0) {
        /// setup the track extras
        reco::TrackExtra::TrajParams trajParams;
        reco::TrackExtra::Chi2sFive chi2s;
        size_t hitsstart = outhits->size();
        if (updatePattern_) {
          t2t(trj, *outhits, trajParams, chi2s);  // this fills the output hit collection
        } else {
          t2t(thetrj, *outhits, trajParams, chi2s);
        }
        size_t hitsend = outhits->size();
        extras->push_back(buildTrackExtra(trj));  // always push back the fully built extra, update by setting in track
        extras->back().setHits(hitsRefProd, hitsstart, hitsend - hitsstart);
        extras->back().setTrajParams(trajParams, chi2s);
        //create the track
        output->push_back(result);
        btlMatchChi2.push_back(mBTL.hit ? mBTL.estChi2 : -1);
        etlMatchChi2.push_back(mETL.hit ? mETL.estChi2 : -1);
        btlMatchTimeChi2.push_back(mBTL.hit ? mBTL.timeChi2 : -1);
        etlMatchTimeChi2.push_back(mETL.hit ? mETL.timeChi2 : -1);
        pathLengthsRaw.push_back(pathLength);
        tmtdRaw.push_back(tmtd);
        sigmatmtdRaw.push_back(sigmatmtd);
        pathLengthMap = pathLength;
        tmtdMap = tmtd;
        sigmatmtdMap = sigmatmtd;
        auto& backtrack = output->back();
        iMap = output->size() - 1;
        pMap = backtrack.p();
        betaMap = backtrack.beta();
        t0Map = backtrack.t0();
        sigmat0Map = std::copysign(std::sqrt(std::abs(backtrack.covt0t0())), backtrack.covt0t0());
        reco::TrackExtraRef extraRef(extrasRefProd, extras->size() - 1);
        backtrack.setExtra((updateExtra_ ? extraRef : track.extra()));
        for (unsigned ihit = hitsstart; ihit < hitsend; ++ihit) {
          backtrack.appendHitPattern((*outhits)[ihit], ttopo);
        }
      }
    }
    pOrigTrkRaw.push_back(pMap);
    betaOrigTrkRaw.push_back(betaMap);
    t0OrigTrkRaw.push_back(t0Map);
    sigmat0OrigTrkRaw.push_back(sigmat0Map);
    pathLengthsOrigTrkRaw.push_back(pathLengthMap);
    tmtdOrigTrkRaw.push_back(tmtdMap);
    sigmatmtdOrigTrkRaw.push_back(sigmatmtdMap);
    assocOrigTrkRaw.push_back(iMap);
    ++itrack;
  }

  auto outTrksHandle = ev.put(std::move(output));
  ev.put(std::move(extras));
  ev.put(std::move(outhits));

  fillValueMap(ev, outTrksHandle, btlMatchChi2, btlMatchChi2Name);
  fillValueMap(ev, outTrksHandle, etlMatchChi2, etlMatchChi2Name);
  fillValueMap(ev, outTrksHandle, btlMatchTimeChi2, btlMatchTimeChi2Name);
  fillValueMap(ev, outTrksHandle, etlMatchTimeChi2, etlMatchTimeChi2Name);
  fillValueMap(ev, outTrksHandle, pathLengthsRaw, pathLengthName);
  fillValueMap(ev, outTrksHandle, tmtdRaw, tmtdName);
  fillValueMap(ev, outTrksHandle, sigmatmtdRaw, sigmatmtdName);
  fillValueMap(ev, tracksH, pOrigTrkRaw, pOrigTrkName);
  fillValueMap(ev, tracksH, betaOrigTrkRaw, betaOrigTrkName);
  fillValueMap(ev, tracksH, t0OrigTrkRaw, t0OrigTrkName);
  fillValueMap(ev, tracksH, sigmat0OrigTrkRaw, sigmat0OrigTrkName);
  fillValueMap(ev, tracksH, pathLengthsOrigTrkRaw, pathLengthOrigTrkName);
  fillValueMap(ev, tracksH, tmtdOrigTrkRaw, tmtdOrigTrkName);
  fillValueMap(ev, tracksH, sigmatmtdOrigTrkRaw, sigmatmtdOrigTrkName);
  fillValueMap(ev, tracksH, assocOrigTrkRaw, assocOrigTrkName);
}

namespace {
  bool cmp_for_detset(const unsigned one, const unsigned two) { return one < two; };

  bool getTrajectoryStateClosestToBeamLine(const Trajectory& traj,
                                           const reco::BeamSpot& bs,
                                           const Propagator* thePropagator,
                                           TrajectoryStateClosestToBeamLine& tscbl) {
    // get the state closest to the beamline
    TrajectoryStateOnSurface stateForProjectionToBeamLineOnSurface =
        traj.closestMeasurement(GlobalPoint(bs.x0(), bs.y0(), bs.z0())).updatedState();

    if
      UNLIKELY(!stateForProjectionToBeamLineOnSurface.isValid()) {
        edm::LogError("CannotPropagateToBeamLine")
            << "the state on the closest measurement isnot valid. skipping track.";
        return false;
      }

    const FreeTrajectoryState& stateForProjectionToBeamLine = *stateForProjectionToBeamLineOnSurface.freeState();

    TSCBLBuilderWithPropagator tscblBuilder(*thePropagator);
    tscbl = tscblBuilder(stateForProjectionToBeamLine, bs);

    if
      UNLIKELY(!tscbl.isValid()) { return false; }

    return true;
  }

  bool trackPathLength(const Trajectory& traj,
                       const reco::BeamSpot& bs,
                       const Propagator* thePropagator,
                       double& pathlength) {
    pathlength = 0.;

    TrajectoryStateClosestToBeamLine tscbl;
    bool tscbl_status = getTrajectoryStateClosestToBeamLine(traj, bs, thePropagator, tscbl);

    if (!tscbl_status)
      return false;

    bool validpropagation = true;
    double pathlength1 = 0.;
    double pathlength2 = 0.;

    //add pathlength layer by layer
    for (auto it = traj.measurements().begin(); it != traj.measurements().end() - 1; ++it) {
      const auto& propresult = thePropagator->propagateWithPath(it->updatedState(), (it + 1)->updatedState().surface());
      double layerpathlength = std::abs(propresult.second);
      if (layerpathlength == 0.) {
        validpropagation = false;
      }
      pathlength1 += layerpathlength;
    }

    //add distance from bs to first measurement
    if (traj.direction() == alongMomentum) {
      const auto& propresult2 =
          thePropagator->propagateWithPath(tscbl.trackStateAtPCA(), traj.firstMeasurement().updatedState().surface());
      pathlength2 = propresult2.second;
      if (pathlength2 == 0.) {
        validpropagation = false;
      }
      pathlength = pathlength1 + pathlength2;
    } else {
      const auto& propresult2 =
          thePropagator->propagateWithPath(tscbl.trackStateAtPCA(), traj.lastMeasurement().updatedState().surface());
      pathlength2 = propresult2.second;
      if (pathlength2 == 0.) {
        validpropagation = false;
      }
      pathlength = pathlength1 + pathlength2;
    }

    return validpropagation;
  }

  struct TrackTofPidInfo {
    TrackTofPidInfo(const double& magp2,
                    const double& length,
                    const double& t_mtd,
                    const double& t_mtderr,
                    const double& t_vtx,
                    const double& t_vtx_err,
                    bool addPIDError = true) {
      tmtd = t_mtd;
      tmtderror = t_mtderr;

      pathlength = length;

      gammasq_pi = 1. + magp2 * m_pi_inv2;
      beta_pi = std::sqrt(1. - 1. / gammasq_pi);
      dt_pi = pathlength / beta_pi * c_inv;

      gammasq_k = 1. + magp2 * m_k_inv2;
      beta_k = std::sqrt(1. - 1. / gammasq_k);
      dt_k = pathlength / beta_k * c_inv;

      gammasq_p = 1. + magp2 * m_p_inv2;
      beta_p = std::sqrt(1. - 1. / gammasq_p);
      dt_p = pathlength / beta_p * c_inv;

      dt = tmtd - dt_pi - t_vtx;  //assume by default the pi hypothesis
      dterror = sqrt(tmtderror * tmtderror + t_vtx_err * t_vtx_err);
      betaerror = 0;
      if (addPIDError) {
        dterror = sqrt(dterror * dterror + (dt_p - dt_pi) * (dt_p - dt_pi));
        betaerror = beta_p - beta_pi;
      }

      dtchi2 = (dt * dt) / (dterror * dterror);

      dt_best = dt;
      dterror_best = dterror;
      dtchi2_best = dtchi2;

      prob_pi = -1.;
      prob_k = -1.;
      prob_p = -1.;

      if (!addPIDError) {
        //*TODO* deal with heavier nucleons and/or BSM case here?
        double chi2_pi = dtchi2;
        double chi2_k = (tmtd - dt_k - t_vtx) * (tmtd - dt_k - t_vtx) / (dterror * dterror);
        double chi2_p = (tmtd - dt_p - t_vtx) * (tmtd - dt_p - t_vtx) / (dterror * dterror);

        double rawprob_pi = exp(-0.5 * chi2_pi);
        double rawprob_k = exp(-0.5 * chi2_k);
        double rawprob_p = exp(-0.5 * chi2_p);
        double normprob = 1. / (rawprob_pi + rawprob_k + rawprob_p);

        prob_pi = rawprob_pi * normprob;
        prob_k = rawprob_k * normprob;
        prob_p = rawprob_p * normprob;

        double prob_heavy = 1. - prob_pi;
        constexpr double heavy_threshold = 0.75;

        if (prob_heavy > heavy_threshold) {
          if (chi2_k < chi2_p) {
            dt_best = (tmtd - dt_k - t_vtx);
            dtchi2_best = chi2_k;
          } else {
            dt_best = (tmtd - dt_p - t_vtx);
            dtchi2_best = chi2_p;
          }
        }
      }
    }

    static constexpr double m_pi = 0.13957018;
    static constexpr double m_pi_inv2 = 1.0 / m_pi / m_pi;
    static constexpr double m_k = 0.493677;
    static constexpr double m_k_inv2 = 1.0 / m_k / m_k;
    static constexpr double m_p = 0.9382720813;
    static constexpr double m_p_inv2 = 1.0 / m_p / m_p;
    static constexpr double c_cm_ns = CLHEP::c_light * CLHEP::ns / CLHEP::cm;  //[cm/ns]
    static constexpr double c_inv = 1.0 / c_cm_ns;
    //this should come from BeamSpot in principle
    //static constexpr double bserror = 0.18;

    double tmtd;
    double tmtderror;

    double pathlength;

    double dt;
    double dterror;
    double dtchi2;

    double dt_best;
    double dterror_best;
    double dtchi2_best;

    double gammasq_pi;
    double beta_pi;
    double dt_pi;

    double gammasq_k;
    double beta_k;
    double dt_k;

    double gammasq_p;
    double beta_p;
    double dt_p;

    double prob_pi;
    double prob_k;
    double prob_p;

    double betaerror;
  };

  void find_hits_in_dets(const MTDTrackingDetSetVector& hits,
                         const Trajectory& traj,
                         const DetLayer* layer,
                         const TrajectoryStateOnSurface& tsos,
                         const double& vtxTime,
                         const reco::BeamSpot& bs,
                         const Propagator* prop,
                         const MeasurementEstimator& theEstimator,
                         const TransientTrackingRecHitBuilder& hitbuilder,
                         bool useVtxConstraint,
                         std::set<MTDHitMatchingInfo>& out) {
    TrajectoryStateClosestToBeamLine tscbl;
    bool tscbl_status = getTrajectoryStateClosestToBeamLine(traj, bs, prop, tscbl);

    if (!tscbl_status)
      return;

    GlobalVector p = tscbl.trackStateAtPCA().momentum();

    double pathlength;
    trackPathLength(traj, bs, prop, pathlength);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, *prop, theEstimator);
    if (comp.first) {
      vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, *prop, theEstimator);
      if (!compDets.empty()) {
        for (const auto& detWithState : compDets) {
          auto range = hits.equal_range(detWithState.first->geographicalId(), cmp_for_detset);
          for (auto detitr = range.first; detitr != range.second; ++detitr) {
            for (auto itr = detitr->begin(); itr != detitr->end(); ++itr) {
              auto est = theEstimator.estimate(detWithState.second, *itr);
              auto pl = prop->propagateWithPath(tsos, detWithState.second.surface());

              if (!est.first || std::abs(pl.second) == 0.)
                continue;

              double tot_pl = pathlength + std::abs(pl.second);  //
              double t_vtx = useVtxConstraint ? vtxTime : 0.;
              double t_vtx_err = useVtxConstraint ? 0.008 : 0.18;  //should use beam spot in the future
              constexpr double t_res_manual = 0.035;
              TrackTofPidInfo tof(p.mag2(),
                                  tot_pl,
                                  itr->time(),
                                  t_res_manual,
                                  t_vtx,
                                  t_vtx_err,
                                  false);  //put hit error by hand for the moment
              MTDHitMatchingInfo mi;
              mi.hit = &(*itr);
              mi.estChi2 = est.second;
              mi.timeChi2 = tof.dtchi2_best;  //use the chi2 for the best matching hypothesis

              out.insert(mi);
            }
          }
        }
      }
    }
  }
}  // namespace

template <class TrackCollection>
TransientTrackingRecHit::ConstRecHitContainer TrackExtenderWithMTDT<TrackCollection>::tryBTLLayers(
    const TrackType& track,
    const Trajectory& traj,
    const MTDTrackingDetSetVector& hits,
    const MTDDetLayerGeometry* geo,
    const MagneticField* field,
    const Propagator* prop,
    const reco::BeamSpot& bs,
    const double& vtxTime,
    const bool matchVertex,
    MTDHitMatchingInfo& bestHit) const {
  TransientTrackingRecHit::ConstRecHitContainer output;
  const vector<const DetLayer*>& layers = geo->allBTLLayers();
  auto tTrack = builder->build(track);
  // get the outermost trajectory point on the track
  TrajectoryStateOnSurface tsos = tTrack.outermostMeasurementState();

  bestHit = MTDHitMatchingInfo();

  for (const DetLayer* ilay : layers) {
    std::set<MTDHitMatchingInfo> hitsInLayer;
    bool hitMatched = false;

    if (useVertex_ && matchVertex)
      find_hits_in_dets(hits, traj, ilay, tsos, vtxTime, bs, prop, *theEstimator, *hitbuilder, true, hitsInLayer);
    else
      find_hits_in_dets(hits, traj, ilay, tsos, 0., bs, prop, *theEstimator, *hitbuilder, false, hitsInLayer);

    //just take the first hit because the hits are sorted on their matching quality
    if (!hitsInLayer.empty()) {
      //check hits to pass minimum quality matching requirements
      if (hitsInLayer.begin()->estChi2 < btlChi2Cut_ && hitsInLayer.begin()->timeChi2 < btlTimeChi2Cut_) {
        hitMatched = true;
        output.push_back(hitbuilder->build(hitsInLayer.begin()->hit));
        if (*(hitsInLayer.begin()) < bestHit)
          bestHit = *(hitsInLayer.begin());
      }
    }

    if (useVertex_ && matchVertex && !hitMatched) {
      //try a second search with beamspot hypothesis
      hitsInLayer.clear();
      find_hits_in_dets(hits, traj, ilay, tsos, 0., bs, prop, *theEstimator, *hitbuilder, false, hitsInLayer);
      if (!hitsInLayer.empty()) {
        if (hitsInLayer.begin()->timeChi2 < btlTimeChi2Cut_) {
          if (hitsInLayer.begin()->estChi2 < btlChi2Cut_) {
            hitMatched = true;
            output.push_back(hitbuilder->build(hitsInLayer.begin()->hit));
            if ((*hitsInLayer.begin()) < bestHit)
              bestHit = *(hitsInLayer.begin());
          }
        }
      }
    }
  }
  return output;
}

template <class TrackCollection>
TransientTrackingRecHit::ConstRecHitContainer TrackExtenderWithMTDT<TrackCollection>::tryETLLayers(
    const TrackType& track,
    const Trajectory& traj,
    const MTDTrackingDetSetVector& hits,
    const MTDDetLayerGeometry* geo,
    const MagneticField* field,
    const Propagator* prop,
    const reco::BeamSpot& bs,
    const double& vtxTime,
    const bool matchVertex,
    MTDHitMatchingInfo& bestHit) const {
  TransientTrackingRecHit::ConstRecHitContainer output;
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  bestHit = MTDHitMatchingInfo();
  auto tTrack = builder->build(track);
  // get the outermost trajectory point on the track
  TrajectoryStateOnSurface tsos = tTrack.outermostMeasurementState();

  for (const DetLayer* ilay : layers) {
    const BoundDisk& disk = static_cast<const MTDRingForwardDoubleLayer*>(ilay)->specificSurface();
    const double diskZ = disk.position().z();

    if (tsos.globalPosition().z() * diskZ < 0)
      continue;  // only propagate to the disk that's on the same side

    std::set<MTDHitMatchingInfo> hitsInLayer;
    bool hitMatched = false;

    if (useVertex_ && matchVertex)
      find_hits_in_dets(hits, traj, ilay, tsos, vtxTime, bs, prop, *theEstimator, *hitbuilder, true, hitsInLayer);
    else
      find_hits_in_dets(hits, traj, ilay, tsos, 0., bs, prop, *theEstimator, *hitbuilder, false, hitsInLayer);

    //just take the first hit because the hits are sorted on their matching quality
    if (!hitsInLayer.empty()) {
      //check hits to pass minimum quality matching requirements
      if (hitsInLayer.begin()->estChi2 < btlChi2Cut_ && hitsInLayer.begin()->timeChi2 < btlTimeChi2Cut_) {
        hitMatched = true;
        output.push_back(hitbuilder->build(hitsInLayer.begin()->hit));
        if (*(hitsInLayer.begin()) < bestHit)
          bestHit = *(hitsInLayer.begin());
      }
    }

    if (useVertex_ && matchVertex && !hitMatched) {
      //try a second search with beamspot hypothesis
      hitsInLayer.clear();
      find_hits_in_dets(hits, traj, ilay, tsos, 0., bs, prop, *theEstimator, *hitbuilder, false, hitsInLayer);
      if (!hitsInLayer.empty()) {
        if (hitsInLayer.begin()->timeChi2 < btlTimeChi2Cut_) {
          if (hitsInLayer.begin()->estChi2 < btlChi2Cut_) {
            hitMatched = true;
            output.push_back(hitbuilder->build(hitsInLayer.begin()->hit));
            if ((*hitsInLayer.begin()) < bestHit)
              bestHit = *(hitsInLayer.begin());
          }
        }
      }
    }
  }
  return output;
}

//below is unfortunately ripped from other places but
//since track producer doesn't know about MTD we have to do this
template <class TrackCollection>
reco::Track TrackExtenderWithMTDT<TrackCollection>::buildTrack(const reco::Track& orig,
                                                               const Trajectory& traj,
                                                               const Trajectory& trajWithMtd,
                                                               const reco::BeamSpot& bs,
                                                               const MagneticField* field,
                                                               const Propagator* thePropagator,
                                                               bool hasMTD,
                                                               float& pathLengthOut,
                                                               float& tmtdOut,
                                                               float& sigmatmtdOut) const {
  TrajectoryStateClosestToBeamLine tscbl;
  bool tsbcl_status = getTrajectoryStateClosestToBeamLine(traj, bs, thePropagator, tscbl);

  if (!tsbcl_status)
    return reco::Track();

  GlobalPoint v = tscbl.trackStateAtPCA().position();
  math::XYZPoint pos(v.x(), v.y(), v.z());
  GlobalVector p = tscbl.trackStateAtPCA().momentum();
  math::XYZVector mom(p.x(), p.y(), p.z());

  int ndof = traj.ndof();

  double t0 = 0.;
  double covt0t0 = -1.;
  pathLengthOut = -1.f;  // if there is no MTD flag the pathlength with -1
  tmtdOut = 0.f;
  sigmatmtdOut = -1.f;
  double betaOut = 0.;
  double covbetabeta = -1.;

  //compute path length for time backpropagation, using first MTD hit for the momentum
  if (hasMTD) {
    double pathlength;
    bool validpropagation = trackPathLength(trajWithMtd, bs, thePropagator, pathlength);
    double thit = 0.;
    double thiterror = -1.;
    bool validmtd = false;

    //need to better handle the cases with >1 hit in MTD
    for (auto it = trajWithMtd.measurements().begin(); it != trajWithMtd.measurements().end(); ++it) {
      bool ismtd = it->recHit()->geographicalId().det() == DetId::Forward &&
                   ForwardSubdetector(it->recHit()->geographicalId().subdetId()) == FastTime;
      if (ismtd) {
        const MTDTrackingRecHit* mtdhit = static_cast<const MTDTrackingRecHit*>(it->recHit()->hit());
        thit = mtdhit->time();
        thiterror = mtdhit->timeError();
        validmtd = true;
        break;
      }
    }

    if (validmtd && validpropagation) {
      TrackTofPidInfo tofInfo(
          p.mag2(),
          pathlength,
          thit,
          thiterror,
          0.,
          0.,
          true);                   //here add the PID uncertainty for later use in the 1st step of 4D vtx reconstruction
      pathLengthOut = pathlength;  // set path length if we've got a timing hit
      tmtdOut = thit;
      sigmatmtdOut = thiterror;
      t0 = tofInfo.dt;
      covt0t0 = tofInfo.dterror * tofInfo.dterror;
      betaOut = tofInfo.beta_pi;
      covbetabeta = tofInfo.betaerror * tofInfo.betaerror;
    }
  }

  return reco::Track(traj.chiSquared(),
                     int(ndof),
                     pos,
                     mom,
                     tscbl.trackStateAtPCA().charge(),
                     tscbl.trackStateAtPCA().curvilinearError(),
                     orig.algo(),
                     reco::TrackBase::undefQuality,
                     t0,
                     betaOut,
                     covt0t0,
                     covbetabeta);
}

template <class TrackCollection>
reco::TrackExtra TrackExtenderWithMTDT<TrackCollection>::buildTrackExtra(const Trajectory& trajectory) const {
  static const string metname = "TrackExtenderWithMTD";

  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();

  // put the collection of TrackingRecHit in the event

  // sets the outermost and innermost TSOSs
  // ToDo: validation for track states with MTD
  TrajectoryStateOnSurface outerTSOS;
  TrajectoryStateOnSurface innerTSOS;
  unsigned int innerId = 0, outerId = 0;
  TrajectoryMeasurement::ConstRecHitPointer outerRecHit;
  DetId outerDetId;

  if (trajectory.direction() == alongMomentum) {
    LogTrace(metname) << "alongMomentum";
    outerTSOS = trajectory.lastMeasurement().updatedState();
    innerTSOS = trajectory.firstMeasurement().updatedState();
    outerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    outerRecHit = trajectory.lastMeasurement().recHit();
    outerDetId = trajectory.lastMeasurement().recHit()->geographicalId();
  } else if (trajectory.direction() == oppositeToMomentum) {
    LogTrace(metname) << "oppositeToMomentum";
    outerTSOS = trajectory.firstMeasurement().updatedState();
    innerTSOS = trajectory.lastMeasurement().updatedState();
    outerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    outerRecHit = trajectory.firstMeasurement().recHit();
    outerDetId = trajectory.firstMeasurement().recHit()->geographicalId();
  } else
    LogError(metname) << "Wrong propagation direction!";

  const GeomDet* outerDet = gtg->idToDet(outerDetId);
  GlobalPoint outerTSOSPos = outerTSOS.globalParameters().position();
  bool inside = outerDet->surface().bounds().inside(outerDet->toLocal(outerTSOSPos));

  GlobalPoint hitPos =
      (outerRecHit->isValid()) ? outerRecHit->globalPosition() : outerTSOS.globalParameters().position();

  if (!inside) {
    LogTrace(metname) << "The Global Muon outerMostMeasurementState is not compatible with the recHit detector!"
                      << " Setting outerMost postition to recHit position if recHit isValid: "
                      << outerRecHit->isValid();
    LogTrace(metname) << "From " << outerTSOSPos << " to " << hitPos;
  }

  //build the TrackExtra
  GlobalPoint v = (inside) ? outerTSOSPos : hitPos;
  GlobalVector p = outerTSOS.globalParameters().momentum();
  math::XYZPoint outpos(v.x(), v.y(), v.z());
  math::XYZVector outmom(p.x(), p.y(), p.z());

  v = innerTSOS.globalParameters().position();
  p = innerTSOS.globalParameters().momentum();
  math::XYZPoint inpos(v.x(), v.y(), v.z());
  math::XYZVector inmom(p.x(), p.y(), p.z());

  reco::TrackExtra trackExtra(outpos,
                              outmom,
                              true,
                              inpos,
                              inmom,
                              true,
                              outerTSOS.curvilinearError(),
                              outerId,
                              innerTSOS.curvilinearError(),
                              innerId,
                              trajectory.direction(),
                              trajectory.seedRef());

  return trackExtra;
}

template <class TrackCollection>
string TrackExtenderWithMTDT<TrackCollection>::dumpLayer(const DetLayer* layer) const {
  stringstream output;

  const BoundSurface* sur = nullptr;
  const BoundCylinder* bc = nullptr;
  const BoundDisk* bd = nullptr;

  sur = &(layer->surface());
  if ((bc = dynamic_cast<const BoundCylinder*>(sur))) {
    output << "  Cylinder of radius: " << bc->radius() << endl;
  } else if ((bd = dynamic_cast<const BoundDisk*>(sur))) {
    output << "  Disk at: " << bd->position().z() << endl;
  }
  return output.str();
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
typedef TrackExtenderWithMTDT<reco::TrackCollection> TrackExtenderWithMTD;

DEFINE_FWK_MODULE(TrackExtenderWithMTD);
