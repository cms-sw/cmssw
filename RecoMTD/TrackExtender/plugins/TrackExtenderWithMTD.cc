#include <sstream>
#include <format>

#include "RecoMTD/TrackExtender/plugins/TrackExtenderWithMTD.h"

#include <CLHEP/Units/GlobalPhysicalConstants.h>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/ForwardDetId/interface/MTDChannelIdentifier.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Rounding.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"
#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackRefitter/interface/RefitDirection.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace mtdtof;

template <class TrackCollection>
TrackExtenderWithMTDT<TrackCollection>::TrackExtenderWithMTDT(const ParameterSet& iConfig)
    : tracksToken_(consumes<TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
      trajTrackAToken_(consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trjtrkAssSrc"))),
      hitsToken_(consumes<MTDTrackingDetSetVector>(iConfig.getParameter<edm::InputTag>("hitsSrc"))),
      bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"))),
      updateTraj_(iConfig.getParameter<bool>("updateTrackTrajectory")),
      updateExtra_(iConfig.getParameter<bool>("updateTrackExtra")),
      updatePattern_(iConfig.getParameter<bool>("updateTrackHitPattern")),
      mtdRecHitBuilder_(iConfig.getParameter<std::string>("MTDRecHitBuilder")),
      propagator_(iConfig.getParameter<std::string>("Propagator")),
      transientTrackBuilder_(iConfig.getParameter<std::string>("TransientTrackBuilder")),
      useVertex_(iConfig.getParameter<bool>("useVertex")) {
  baseMTDExtender_ = std::make_unique<BaseExtenderWithMTD>(iConfig);

  if (useVertex_) {
    vtxToken_ = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vtxSrc"));
  }

  theTransformer = std::make_unique<TrackTransformer>(iConfig.getParameterSet("TrackTransformer"), consumesCollector());

  btlMatchChi2Token_ = produces<edm::ValueMap<float>>("btlMatchChi2");
  etlMatchChi2Token_ = produces<edm::ValueMap<float>>("etlMatchChi2");
  btlMatchTimeChi2Token_ = produces<edm::ValueMap<float>>("btlMatchTimeChi2");
  etlMatchTimeChi2Token_ = produces<edm::ValueMap<float>>("etlMatchTimeChi2");
  npixBarrelToken_ = produces<edm::ValueMap<int>>("npixBarrel");
  npixEndcapToken_ = produces<edm::ValueMap<int>>("npixEndcap");
  outermostHitPositionToken_ = produces<edm::ValueMap<float>>("generalTrackOutermostHitPosition");
  pOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackp");
  betaOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackBeta");
  t0OrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackt0");
  sigmat0OrigTrkToken_ = produces<edm::ValueMap<float>>("generalTracksigmat0");
  pathLengthOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackPathLength");
  tmtdOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTracktmtd");
  sigmatmtdOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTracksigmatmtd");
  tmtdPosOrigTrkToken_ = produces<edm::ValueMap<GlobalPoint>>("generalTrackmtdpos");
  tofpiOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackTofPi");
  tofkOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackTofK");
  tofpOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackTofP");
  sigmatofpiOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackSigmaTofPi");
  sigmatofkOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackSigmaTofK");
  sigmatofpOrigTrkToken_ = produces<edm::ValueMap<float>>("generalTrackSigmaTofP");
  assocOrigTrkToken_ = produces<edm::ValueMap<int>>("generalTrackassoc");

  builderToken_ = esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", transientTrackBuilder_));
  hitbuilderToken_ =
      esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(edm::ESInputTag("", mtdRecHitBuilder_));
  gtgToken_ = esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>();
  dlgeoToken_ = esConsumes<MTDDetLayerGeometry, MTDRecoGeometryRecord>();
  magfldToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  propToken_ = esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", propagator_));
  ttopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();

  produces<edm::OwnVector<TrackingRecHit>>();
  produces<reco::TrackExtraCollection>();
  produces<TrackCollection>();
}

template <class TrackCollection>
template <class H, class T>
void TrackExtenderWithMTDT<TrackCollection>::fillValueMap(edm::Event& iEvent,
                                                          const H& handle,
                                                          const std::vector<T>& vec,
                                                          const edm::EDPutToken& token) const {
  auto out = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(handle, vec.begin(), vec.end());
  filler.fill();
  iEvent.put(token, std::move(out));
}

template <class TrackCollection>
void TrackExtenderWithMTDT<TrackCollection>::produce(edm::Event& ev, const edm::EventSetup& es) {
  //this produces pieces of the track extra
  Traj2TrackHits t2t;

  theTransformer->setServices(es);
  TrackingRecHitRefProd hitsRefProd = ev.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd extrasRefProd = ev.getRefBeforePut<reco::TrackExtraCollection>();

  gtg_ = es.getHandle(gtgToken_);

  auto geo = es.getTransientHandle(dlgeoToken_);

  auto magfield = es.getTransientHandle(magfldToken_);

  builder_ = es.getHandle(builderToken_);
  hitbuilder_ = es.getHandle(hitbuilderToken_);

  baseMTDExtender_->setParameters(&*hitbuilder_, &*gtg_);

  auto propH = es.getTransientHandle(propToken_);
  const Propagator* prop = propH.product();

  auto httopo = es.getTransientHandle(ttopoToken_);
  const TrackerTopology& ttopo = *httopo;

  auto output = std::make_unique<TrackCollection>();
  auto extras = std::make_unique<reco::TrackExtraCollection>();
  auto outhits = std::make_unique<edm::OwnVector<TrackingRecHit>>();

  std::vector<float> btlMatchChi2;
  std::vector<float> etlMatchChi2;
  std::vector<float> btlMatchTimeChi2;
  std::vector<float> etlMatchTimeChi2;
  std::vector<int> npixBarrel;
  std::vector<int> npixEndcap;
  std::vector<float> outermostHitPosition;
  std::vector<float> pOrigTrkRaw;
  std::vector<float> betaOrigTrkRaw;
  std::vector<float> t0OrigTrkRaw;
  std::vector<float> sigmat0OrigTrkRaw;
  std::vector<float> pathLengthsOrigTrkRaw;
  std::vector<float> tmtdOrigTrkRaw;
  std::vector<float> sigmatmtdOrigTrkRaw;
  std::vector<GlobalPoint> tmtdPosOrigTrkRaw;
  std::vector<float> tofpiOrigTrkRaw;
  std::vector<float> tofkOrigTrkRaw;
  std::vector<float> tofpOrigTrkRaw;
  std::vector<float> sigmatofpiOrigTrkRaw;
  std::vector<float> sigmatofkOrigTrkRaw;
  std::vector<float> sigmatofpOrigTrkRaw;
  std::vector<int> assocOrigTrkRaw;

  auto const tracksH = ev.getHandle(tracksToken_);

  const auto& trjtrks = ev.get(trajTrackAToken_);

  //MTD hits DetSet
  const auto& hits = ev.get(hitsToken_);

  //beam spot
  const auto& bs = ev.get(bsToken_);

  bool vtxConstraint(false);

  VertexCollection vtxs;
  if (useVertex_) {
    vtxs = ev.get(vtxToken_);
    if (!vtxs.empty()) {
      vtxConstraint = true;
    }
  }

  std::vector<unsigned> track_indices;
  unsigned itrack = 0;

  for (const auto& trjtrk : trjtrks) {
    const Trajectory& trajs = *trjtrk.key;
    const reco::TrackRef& track = trjtrk.val;

    LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: extrapolating track " << itrack
                                     << " p/pT = " << track->p() << " " << track->pt() << " eta = " << track->eta();
    LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: sigma_p = "
                                     << sqrt(track->covariance()(0, 0)) * track->p2()
                                     << " sigma_p/p = " << sqrt(track->covariance()(0, 0)) * track->p() * 100 << " %";

    float trackVtxTime = 0.f;
    float trackVtxTimeError = 0.f;
    if (vtxConstraint) {
      for (const auto& vtx : vtxs) {
        for (size_t itrk = 0; itrk < vtx.tracksSize(); itrk++) {
          if (track == vtx.trackRefAt(itrk).castTo<TrackRef>()) {
            trackVtxTime = vtx.t();
            trackVtxTimeError = vtx.tError();
            break;
          }
        }
      }
    }

    reco::TransientTrack ttrack(track, magfield.product(), gtg_);
    auto thits = theTransformer->getTransientRecHits(ttrack);
    TransientTrackingRecHit::ConstRecHitContainer mtdthits;
    MTDHitMatchingInfo mBTL, mETL;

    if (trajs.isValid()) {
      // get the outermost trajectory point on the track
      TrajectoryStateOnSurface tsos = builder_->build(track).outermostMeasurementState();
      TrajectoryStateClosestToBeamLine tscbl;
      bool tscbl_status = getTrajectoryStateClosestToBeamLine(trajs, bs, prop, tscbl);

      if (tscbl_status) {
        float pmag2 = tscbl.trackStateAtPCA().momentum().mag2();
        float pathlength0;
        TrackSegments trs0;
        trackPathLength(trajs, tscbl, prop, pathlength0, trs0);

        const auto& btlhits = baseMTDExtender_->tryBTLLayers(tsos,
                                                             trajs,
                                                             pmag2,
                                                             pathlength0,
                                                             trs0,
                                                             hits,
                                                             geo.product(),
                                                             magfield.product(),
                                                             prop,
                                                             bs,
                                                             trackVtxTime,
                                                             trackVtxTimeError,
                                                             mBTL);
        mtdthits.insert(mtdthits.end(), btlhits.begin(), btlhits.end());

        // in the future this should include an intermediate refit before propagating to the ETL
        // for now it is ok
        const auto& etlhits = baseMTDExtender_->tryETLLayers(tsos,
                                                             trajs,
                                                             pmag2,
                                                             pathlength0,
                                                             trs0,
                                                             hits,
                                                             geo.product(),
                                                             magfield.product(),
                                                             prop,
                                                             bs,
                                                             trackVtxTime,
                                                             trackVtxTimeError,
                                                             mETL);
        mtdthits.insert(mtdthits.end(), etlhits.begin(), etlhits.end());
      }
#ifdef EDM_ML_DEBUG
      else {
        LogTrace("TrackExtenderWithMTD") << "Failing getTrajectoryStateClosestToBeamLine, no search for hits in MTD!";
      }
#endif
    }

    auto ordering = baseMTDExtender_->checkRecHitsOrdering(thits);
    if (ordering == RefitDirection::insideOut) {
      thits.insert(thits.end(), mtdthits.begin(), mtdthits.end());
    } else {
      std::reverse(mtdthits.begin(), mtdthits.end());
      mtdthits.insert(mtdthits.end(), thits.begin(), thits.end());
      thits.swap(mtdthits);
    }

    const auto& trajwithmtd =
        mtdthits.empty() ? std::vector<Trajectory>(1, trajs) : theTransformer->transform(ttrack, thits);
    float pMap = 0.f, betaMap = 0.f, t0Map = 0.f, sigmat0Map = -1.f, pathLengthMap = -1.f, tmtdMap = 0.f,
          sigmatmtdMap = -1.f, tofpiMap = 0.f, tofkMap = 0.f, tofpMap = 0.f, sigmatofpiMap = -1.f, sigmatofkMap = -1.f,
          sigmatofpMap = -1.f;
    GlobalPoint tmtdPosMap{0., 0., 0.};
    int iMap = -1;

    for (const auto& trj : trajwithmtd) {
      const auto& thetrj = (updateTraj_ ? trj : trajs);
      float pathLength = 0.f, tmtd = 0.f, sigmatmtd = -1.f, tofpi = 0.f, tofk = 0.f, tofp = 0.f, sigmatofpi = -1.f,
            sigmatofk = -1.f, sigmatofp = -1.f;
      GlobalPoint tmtdPos{0., 0., 0.};
      LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: refit track " << itrack << " p/pT = " << track->p()
                                       << " " << track->pt() << " eta = " << track->eta();
      reco::Track result = baseMTDExtender_->buildTrack(track,
                                                        thetrj,
                                                        trj,
                                                        bs,
                                                        magfield.product(),
                                                        prop,
                                                        !trajwithmtd.empty() && !mtdthits.empty(),
                                                        pathLength,
                                                        tmtd,
                                                        sigmatmtd,
                                                        tmtdPos,
                                                        tofpi,
                                                        tofk,
                                                        tofp,
                                                        sigmatofpi,
                                                        sigmatofk,
                                                        sigmatofp);
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
        extras->push_back(baseMTDExtender_->buildTrackExtra(
            trj));  // always push back the fully built extra, update by setting in track
        extras->back().setHits(hitsRefProd, hitsstart, hitsend - hitsstart);
        extras->back().setTrajParams(trajParams, chi2s);
        //create the track
        output->push_back(result);
        btlMatchChi2.push_back(mBTL.hit ? mBTL.estChi2 : -1.f);
        etlMatchChi2.push_back(mETL.hit ? mETL.estChi2 : -1.f);
        btlMatchTimeChi2.push_back(mBTL.hit ? mBTL.timeChi2 : -1.f);
        etlMatchTimeChi2.push_back(mETL.hit ? mETL.timeChi2 : -1.f);
        pathLengthMap = pathLength;
        tmtdMap = tmtd;
        sigmatmtdMap = sigmatmtd;
        tmtdPosMap = tmtdPos;
        auto& backtrack = output->back();
        iMap = output->size() - 1;
        pMap = backtrack.p();
        betaMap = backtrack.beta();
        t0Map = backtrack.t0();
        sigmat0Map = std::copysign(std::sqrt(std::abs(backtrack.covt0t0())), backtrack.covt0t0());
        tofpiMap = tofpi;
        tofkMap = tofk;
        tofpMap = tofp;
        sigmatofpiMap = sigmatofpi;
        sigmatofkMap = sigmatofk;
        sigmatofpMap = sigmatofp;
        reco::TrackExtraRef extraRef(extrasRefProd, extras->size() - 1);
        backtrack.setExtra((updateExtra_ ? extraRef : track->extra()));
        for (unsigned ihit = hitsstart; ihit < hitsend; ++ihit) {
          backtrack.appendHitPattern((*outhits)[ihit], ttopo);
        }
#ifdef EDM_ML_DEBUG
        LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: hit pattern of refitted track";
        for (int i = 0; i < backtrack.hitPattern().numberOfAllHits(reco::HitPattern::TRACK_HITS); i++) {
          backtrack.hitPattern().printHitPattern(reco::HitPattern::TRACK_HITS, i, std::cout);
        }
        LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: missing hit pattern of refitted track";
        for (int i = 0; i < backtrack.hitPattern().numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS); i++) {
          backtrack.hitPattern().printHitPattern(reco::HitPattern::MISSING_INNER_HITS, i, std::cout);
        }
#endif
        npixBarrel.push_back(backtrack.hitPattern().numberOfValidPixelBarrelHits());
        npixEndcap.push_back(backtrack.hitPattern().numberOfValidPixelEndcapHits());

        if (mBTL.hit || mETL.hit) {
          outermostHitPosition.push_back(
              mBTL.hit ? (float)(*track).outerRadius()
                       : (float)(*track).outerZ());  // save R of the outermost hit for BTL, z for ETL.
        } else {
          outermostHitPosition.push_back(std::abs(track->eta()) < trackMaxBtlEta_ ? (float)(*track).outerRadius()
                                                                                  : (float)(*track).outerZ());
        }

        LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: tmtd " << tmtdMap << " +/- " << sigmatmtdMap
                                         << " t0 " << t0Map << " +/- " << sigmat0Map << " tof pi/K/p " << tofpiMap
                                         << "+/-" << std::format("{:0.2g}", sigmatofpiMap) << " ("
                                         << std::format("{:0.2g}", sigmatofpiMap / tofpiMap * 100) << "%) " << tofkMap
                                         << "+/-" << std::format("{:0.2g}", sigmatofkMap) << " ("
                                         << std::format("{:0.2g}", sigmatofkMap / tofkMap * 100) << "%) " << tofpMap
                                         << "+/-" << std::format("{:0.2g}", sigmatofpMap) << " ("
                                         << std::format("{:0.2g}", sigmatofpMap / tofpMap * 100) << "%) ";
      } else {
        LogTrace("TrackExtenderWithMTD") << "Error in the MTD track refitting. This should not happen";
      }
    }

    pOrigTrkRaw.push_back(pMap);
    betaOrigTrkRaw.push_back(betaMap);
    t0OrigTrkRaw.push_back(t0Map);
    sigmat0OrigTrkRaw.push_back(sigmat0Map);
    pathLengthsOrigTrkRaw.push_back(pathLengthMap);
    tmtdOrigTrkRaw.push_back(tmtdMap);
    sigmatmtdOrigTrkRaw.push_back(sigmatmtdMap);
    tmtdPosOrigTrkRaw.push_back(tmtdPosMap);
    tofpiOrigTrkRaw.push_back(tofpiMap);
    tofkOrigTrkRaw.push_back(tofkMap);
    tofpOrigTrkRaw.push_back(tofpMap);
    sigmatofpiOrigTrkRaw.push_back(sigmatofpiMap);
    sigmatofkOrigTrkRaw.push_back(sigmatofkMap);
    sigmatofpOrigTrkRaw.push_back(sigmatofpMap);
    assocOrigTrkRaw.push_back(iMap);

    if (iMap == -1) {
      btlMatchChi2.push_back(-1.f);
      etlMatchChi2.push_back(-1.f);
      btlMatchTimeChi2.push_back(-1.f);
      etlMatchTimeChi2.push_back(-1.f);
      npixBarrel.push_back(-1.f);
      npixEndcap.push_back(-1.f);
      outermostHitPosition.push_back(0.);
    }

    ++itrack;
  }

  ev.put(std::move(output));
  ev.put(std::move(extras));
  ev.put(std::move(outhits));

  fillValueMap(ev, tracksH, btlMatchChi2, btlMatchChi2Token_);
  fillValueMap(ev, tracksH, etlMatchChi2, etlMatchChi2Token_);
  fillValueMap(ev, tracksH, btlMatchTimeChi2, btlMatchTimeChi2Token_);
  fillValueMap(ev, tracksH, etlMatchTimeChi2, etlMatchTimeChi2Token_);
  fillValueMap(ev, tracksH, npixBarrel, npixBarrelToken_);
  fillValueMap(ev, tracksH, npixEndcap, npixEndcapToken_);
  fillValueMap(ev, tracksH, outermostHitPosition, outermostHitPositionToken_);
  fillValueMap(ev, tracksH, pOrigTrkRaw, pOrigTrkToken_);
  fillValueMap(ev, tracksH, betaOrigTrkRaw, betaOrigTrkToken_);
  fillValueMap(ev, tracksH, t0OrigTrkRaw, t0OrigTrkToken_);
  fillValueMap(ev, tracksH, sigmat0OrigTrkRaw, sigmat0OrigTrkToken_);
  fillValueMap(ev, tracksH, pathLengthsOrigTrkRaw, pathLengthOrigTrkToken_);
  fillValueMap(ev, tracksH, tmtdOrigTrkRaw, tmtdOrigTrkToken_);
  fillValueMap(ev, tracksH, sigmatmtdOrigTrkRaw, sigmatmtdOrigTrkToken_);
  fillValueMap(ev, tracksH, tmtdPosOrigTrkRaw, tmtdPosOrigTrkToken_);
  fillValueMap(ev, tracksH, tofpiOrigTrkRaw, tofpiOrigTrkToken_);
  fillValueMap(ev, tracksH, tofkOrigTrkRaw, tofkOrigTrkToken_);
  fillValueMap(ev, tracksH, tofpOrigTrkRaw, tofpOrigTrkToken_);
  fillValueMap(ev, tracksH, sigmatofpiOrigTrkRaw, sigmatofpiOrigTrkToken_);
  fillValueMap(ev, tracksH, sigmatofkOrigTrkRaw, sigmatofkOrigTrkToken_);
  fillValueMap(ev, tracksH, sigmatofpOrigTrkRaw, sigmatofpOrigTrkToken_);
  fillValueMap(ev, tracksH, assocOrigTrkRaw, assocOrigTrkToken_);
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
typedef TrackExtenderWithMTDT<reco::TrackCollection> TrackExtenderWithMTD;

DEFINE_FWK_MODULE(TrackExtenderWithMTD);
