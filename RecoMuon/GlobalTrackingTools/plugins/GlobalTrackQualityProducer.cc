// -*- C++ -*-
//
// Package:    GlobalTrackingTools
// Class:      GlobalTrackQualityProducer
//
//
// Original Author:  Adam Everett
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "RecoMuon/GlobalTrackingTools/plugins/GlobalTrackQualityProducer.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

GlobalTrackQualityProducer::GlobalTrackQualityProducer(const edm::ParameterSet& iConfig)
    : inputCollection_(iConfig.getParameter<edm::InputTag>("InputCollection")),
      inputLinksCollection_(iConfig.getParameter<edm::InputTag>("InputLinksCollection")),
      tTopoToken_(esConsumes()),
      theService(nullptr),
      theGlbRefitter(nullptr),
      theGlbMatcher(nullptr) {
  // service parameters
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters, consumesCollector());

  // TrackRefitter parameters
  edm::ConsumesCollector iC = consumesCollector();
  edm::ParameterSet refitterParameters = iConfig.getParameter<edm::ParameterSet>("RefitterParameters");
  theGlbRefitter = new GlobalMuonRefitter(refitterParameters, theService, iC);

  edm::ParameterSet trackMatcherPSet = iConfig.getParameter<edm::ParameterSet>("GlobalMuonTrackMatcher");
  theGlbMatcher = new GlobalMuonTrackMatcher(trackMatcherPSet, theService);

  double maxChi2 = iConfig.getParameter<double>("MaxChi2");
  double nSigma = iConfig.getParameter<double>("nSigma");
  theEstimator = new Chi2MeasurementEstimator(maxChi2, nSigma);

  glbMuonsToken = consumes<reco::TrackCollection>(inputCollection_);
  linkCollectionToken = consumes<reco::MuonTrackLinksCollection>(inputLinksCollection_);

  produces<edm::ValueMap<reco::MuonQuality>>();
}

GlobalTrackQualityProducer::~GlobalTrackQualityProducer() {
  if (theService)
    delete theService;
  if (theGlbRefitter)
    delete theGlbRefitter;
  if (theGlbMatcher)
    delete theGlbMatcher;
  if (theEstimator)
    delete theEstimator;
}

void GlobalTrackQualityProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string theCategory = "Muon|RecoMuon|GlobalTrackQualityProducer";

  theService->update(iSetup);

  theGlbRefitter->setEvent(iEvent);

  theGlbRefitter->setServices(theService->eventSetup());

  // Take the GLB muon container(s)
  edm::Handle<reco::TrackCollection> glbMuons;
  iEvent.getByToken(glbMuonsToken, glbMuons);

  edm::Handle<reco::MuonTrackLinksCollection> linkCollectionHandle;
  iEvent.getByToken(linkCollectionToken, linkCollectionHandle);

  //Retrieve tracker topology from geometry
  const TrackerTopology* tTopo = &iSetup.getData(tTopoToken_);

  // reserve some space
  std::vector<reco::MuonQuality> valuesQual;
  valuesQual.reserve(glbMuons->size());

  int trackIndex = 0;
  for (reco::TrackCollection::const_iterator track = glbMuons->begin(); track != glbMuons->end();
       ++track, ++trackIndex) {
    reco::TrackRef glbRef(glbMuons, trackIndex);
    reco::TrackRef staTrack = reco::TrackRef();

    std::vector<Trajectory> refitted = theGlbRefitter->refit(*track, 1, tTopo);

    LogTrace(theCategory) << "GLBQual N refitted " << refitted.size();

    std::pair<double, double> thisKink;
    double relative_muon_chi2 = 0.0;
    double relative_tracker_chi2 = 0.0;
    double glbTrackProbability = 0.0;
    if (!refitted.empty()) {
      thisKink = kink(refitted.front());
      std::pair<double, double> chi = newChi2(refitted.front());
      relative_muon_chi2 = chi.second;    //normalized inside to /sum(muHits.dimension)
      relative_tracker_chi2 = chi.first;  //normalized inside to /sum(tkHits.dimension)
      glbTrackProbability = trackProbability(refitted.front());
    }

    LogTrace(theCategory) << "GLBQual: Kink " << thisKink.first << " " << thisKink.second;
    LogTrace(theCategory) << "GLBQual: Rel Chi2 " << relative_tracker_chi2 << " " << relative_muon_chi2;
    LogTrace(theCategory) << "GLBQual: trackProbability " << glbTrackProbability;

    // Fill the STA-TK match information
    float chi2, d, dist, Rpos;
    chi2 = d = dist = Rpos = -1.0;
    bool passTight = false;
    typedef MuonTrajectoryBuilder::TrackCand TrackCand;
    if (linkCollectionHandle.isValid()) {
      for (reco::MuonTrackLinksCollection::const_iterator links = linkCollectionHandle->begin();
           links != linkCollectionHandle->end();
           ++links) {
        if (links->trackerTrack().isNull() || links->standAloneTrack().isNull() || links->globalTrack().isNull()) {
          edm::LogWarning(theCategory) << "Global muon links to constituent tracks are invalid. There should be no "
                                          "such object. Muon is skipped.";
          continue;
        }
        if (links->globalTrack() == glbRef) {
          staTrack = !links->standAloneTrack().isNull() ? links->standAloneTrack() : reco::TrackRef();
          TrackCand staCand = TrackCand((Trajectory*)nullptr, links->standAloneTrack());
          TrackCand tkCand = TrackCand((Trajectory*)nullptr, links->trackerTrack());
          chi2 = theGlbMatcher->match(staCand, tkCand, 0, 0);
          d = theGlbMatcher->match(staCand, tkCand, 1, 0);
          Rpos = theGlbMatcher->match(staCand, tkCand, 2, 0);
          dist = theGlbMatcher->match(staCand, tkCand, 3, 0);
          passTight = theGlbMatcher->matchTight(staCand, tkCand);
        }
      }
    }

    if (!staTrack.isNull())
      LogTrace(theCategory) << "GLBQual: Used UpdatedAtVtx : "
                            << (iEvent.getStableProvenance(staTrack.id()).productInstanceName() ==
                                std::string("UpdatedAtVtx"));

    float maxFloat01 =
        std::numeric_limits<float>::max() * 0.1;  // a better solution would be to use float above .. m/be not
    reco::MuonQuality muQual;
    if (!staTrack.isNull())
      muQual.updatedSta =
          iEvent.getStableProvenance(staTrack.id()).productInstanceName() == std::string("UpdatedAtVtx");
    muQual.trkKink = thisKink.first > maxFloat01 ? maxFloat01 : thisKink.first;
    muQual.glbKink = thisKink.second > maxFloat01 ? maxFloat01 : thisKink.second;
    muQual.trkRelChi2 = relative_tracker_chi2 > maxFloat01 ? maxFloat01 : relative_tracker_chi2;
    muQual.staRelChi2 = relative_muon_chi2 > maxFloat01 ? maxFloat01 : relative_muon_chi2;
    muQual.tightMatch = passTight;
    muQual.chi2LocalPosition = dist;
    muQual.chi2LocalMomentum = chi2;
    muQual.localDistance = d;
    muQual.globalDeltaEtaPhi = Rpos;
    muQual.glbTrackProbability = glbTrackProbability;
    valuesQual.push_back(muQual);
  }

  /*
  for(int i = 0; i < valuesTkRelChi2.size(); i++) {
    LogTrace(theCategory)<<"value " << valuesTkRelChi2[i] ;
  }
  */

  // create and fill value maps
  auto outQual = std::make_unique<edm::ValueMap<reco::MuonQuality>>();
  edm::ValueMap<reco::MuonQuality>::Filler fillerQual(*outQual);
  fillerQual.insert(glbMuons, valuesQual.begin(), valuesQual.end());
  fillerQual.fill();

  // put value map into event
  iEvent.put(std::move(outQual));
}

std::pair<double, double> GlobalTrackQualityProducer::kink(Trajectory& muon) const {
  const std::string theCategory = "Muon|RecoMuon|GlobalTrackQualityProducer";

  using namespace std;
  using namespace edm;
  using namespace reco;

  double result = 0.0;
  double resultGlb = 0.0;

  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef ConstRecHitPointer RecHit;
  typedef std::vector<TrajectoryMeasurement>::const_iterator TMI;

  vector<TrajectoryMeasurement> meas = muon.measurements();

  for (TMI m = meas.begin(); m != meas.end(); m++) {
    TransientTrackingRecHit::ConstRecHitPointer hit = m->recHit();

    //not used    double estimate = 0.0;

    RecHit rhit = (*m).recHit();
    bool ok = false;
    if (rhit->isValid()) {
      if (DetId::Tracker == rhit->geographicalId().det())
        ok = true;
    }

    //if ( !ok ) continue;

    const TrajectoryStateOnSurface& tsos = (*m).predictedState();

    if (tsos.isValid() && rhit->isValid() && rhit->hit()->isValid() &&
        !edm::isNotFinite(rhit->localPositionError().xx())     //this is paranoia induced by reported case
        && !edm::isNotFinite(rhit->localPositionError().xy())  //it's better to track down the origin of bad numbers
        && !edm::isNotFinite(rhit->localPositionError().yy())) {
      double phi1 = tsos.globalPosition().phi();
      if (phi1 < 0)
        phi1 = 2 * M_PI + phi1;

      double phi2 = rhit->globalPosition().phi();
      if (phi2 < 0)
        phi2 = 2 * M_PI + phi2;

      double diff = fabs(phi1 - phi2);
      if (diff > M_PI)
        diff = 2 * M_PI - diff;

      GlobalPoint hitPos = rhit->globalPosition();

      GlobalError hitErr = rhit->globalPositionError();
      //LogDebug(theCategory)<<"hitPos " << hitPos;
      double error = hitErr.phierr(hitPos);  // error squared

      double s = (error > 0.0) ? (diff * diff) / error : (diff * diff);

      if (ok)
        result += s;
      resultGlb += s;
    }
  }

  return std::pair<double, double>(result, resultGlb);
}

std::pair<double, double> GlobalTrackQualityProducer::newChi2(Trajectory& muon) const {
  const std::string theCategory = "Muon|RecoMuon|GlobalTrackQualityProducer";

  using namespace std;
  using namespace edm;
  using namespace reco;

  double muChi2 = 0.0;
  double tkChi2 = 0.0;
  unsigned int muNdof = 0;
  unsigned int tkNdof = 0;

  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef ConstRecHitPointer RecHit;
  typedef vector<TrajectoryMeasurement>::const_iterator TMI;

  vector<TrajectoryMeasurement> meas = muon.measurements();

  for (TMI m = meas.begin(); m != meas.end(); m++) {
    TransientTrackingRecHit::ConstRecHitPointer hit = m->recHit();
    const TrajectoryStateOnSurface& uptsos = (*m).updatedState();
    // FIXME FIXME CLONE!!!
    // TrackingRecHit::RecHitPointer preciseHit = hit->clone(uptsos);
    const auto& preciseHit = hit;
    double estimate = 0.0;
    if (preciseHit->isValid() && uptsos.isValid()) {
      estimate = theEstimator->estimate(uptsos, *preciseHit).second;
    }

    //LogTrace(theCategory) << "estimate " << estimate << " TM.est " << m->estimate();
    //UNUSED:    double tkDiff = 0.0;
    //UNUSED:    double staDiff = 0.0;
    if (hit->isValid() && (hit->geographicalId().det()) == DetId::Tracker) {
      tkChi2 += estimate;
      //UNUSED:      tkDiff = estimate - m->estimate();
      tkNdof += hit->dimension();
    }
    if (hit->isValid() && (hit->geographicalId().det()) == DetId::Muon) {
      muChi2 += estimate;
      //UNUSED      staDiff = estimate - m->estimate();
      muNdof += hit->dimension();
    }
  }

  //For tkNdof < 6, should a large number or something else
  // be used instead of just tkChi2 directly?
  if (tkNdof > 5) {
    tkChi2 /= (tkNdof - 5.);
  }

  //For muNdof < 6, should a large number or something else
  // be used instead of just muChi2 directly?
  if (muNdof > 5) {
    muChi2 /= (muNdof - 5.);
  }

  return std::pair<double, double>(tkChi2, muChi2);
}

//
// calculate the tail probability (-ln(P)) of a fit
//
double GlobalTrackQualityProducer::trackProbability(Trajectory& track) const {
  if (track.ndof() > 0 && track.chiSquared() > 0) {
    return -LnChiSquaredProbability(track.chiSquared(), track.ndof());
  } else {
    return 0.0;
  }
}

void GlobalTrackQualityProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd1;
    psd1.setAllowAnything();
    desc.add<edm::ParameterSetDescription>("ServiceParameters", psd1);
  }
  {
    edm::ParameterSetDescription psd1;
    psd1.setAllowAnything();
    desc.add<edm::ParameterSetDescription>("GlobalMuonTrackMatcher", psd1);
  }
  desc.add<edm::InputTag>("InputCollection", edm::InputTag("globalMuons"));
  desc.add<edm::InputTag>("InputLinksCollection", edm::InputTag("globalMuons"));
  desc.add<std::string>("BaseLabel", "GLB");
  {
    edm::ParameterSetDescription descGlbMuonRefitter;
    descGlbMuonRefitter.setAllowAnything();
    descGlbMuonRefitter.add<edm::InputTag>("DTRecSegmentLabel", edm::InputTag("dt1DRecHits"));
    descGlbMuonRefitter.add<edm::InputTag>("CSCRecSegmentLabel", edm::InputTag("csc2DRecHits"));
    descGlbMuonRefitter.add<edm::InputTag>("GEMRecHitLabel", edm::InputTag("gemRecHits"));
    descGlbMuonRefitter.add<edm::InputTag>("ME0RecHitLabel", edm::InputTag("me0Segments"));
    descGlbMuonRefitter.add<edm::InputTag>("RPCRecSegmentLabel", edm::InputTag("rpcRecHits"));

    descGlbMuonRefitter.add<std::string>("Fitter", "KFFitterForRefitInsideOut");
    descGlbMuonRefitter.add<std::string>("Smoother", "KFSmootherForRefitInsideOut");
    descGlbMuonRefitter.add<std::string>("Propagator", "SmartPropagatorAnyRK");
    descGlbMuonRefitter.add<std::string>("TrackerRecHitBuilder", "WithAngleAndTemplate");
    descGlbMuonRefitter.add<std::string>("MuonRecHitBuilder", "MuonRecHitBuilder");
    descGlbMuonRefitter.add<bool>("DoPredictionsOnly", false);
    descGlbMuonRefitter.add<std::string>("RefitDirection", "insideOut");
    descGlbMuonRefitter.add<bool>("PropDirForCosmics", false);
    descGlbMuonRefitter.add<bool>("RefitRPCHits", true);

    descGlbMuonRefitter.add<std::vector<int>>("DYTthrs", {10, 10});
    descGlbMuonRefitter.add<int>("DYTselector", 1);
    descGlbMuonRefitter.add<bool>("DYTupdator", false);
    descGlbMuonRefitter.add<bool>("DYTuseAPE", false);
    descGlbMuonRefitter.add<bool>("DYTuseThrsParametrization", true);
    {
      edm::ParameterSetDescription descDYTthrs;
      descDYTthrs.add<std::vector<double>>("eta0p8", {1, -0.919853, 0.990742});
      descDYTthrs.add<std::vector<double>>("eta1p2", {1, -0.897354, 0.987738});
      descDYTthrs.add<std::vector<double>>("eta2p0", {4, -0.986855, 0.998516});
      descDYTthrs.add<std::vector<double>>("eta2p2", {1, -0.940342, 0.992955});
      descDYTthrs.add<std::vector<double>>("eta2p4", {1, -0.947633, 0.993762});
      descGlbMuonRefitter.add<edm::ParameterSetDescription>("DYTthrsParameters", descDYTthrs);
    }

    descGlbMuonRefitter.add<int>("SkipStation", -1);
    descGlbMuonRefitter.add<int>("TrackerSkipSystem", -1);
    descGlbMuonRefitter.add<int>("TrackerSkipSection", -1);
    descGlbMuonRefitter.add<bool>("RefitFlag", true);

    desc.add<edm::ParameterSetDescription>("RefitterParameters", descGlbMuonRefitter);
  }
  desc.add<double>("nSigma", 3.0);
  desc.add<double>("MaxChi2", 100000.0);

  descriptions.add("globalTrackQualityProducer", desc);
}
//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_FWK_MODULE(GlobalTrackQualityProducer);
