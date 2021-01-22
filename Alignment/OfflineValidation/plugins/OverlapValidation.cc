// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      OverlapValidation
//
/**\ Alignment/OfflineValidation/src/OverlapValidation.cc

 Description: <one line class summary>
	
*/
//
// Original Authors:  Wolfgang Adam, Keith Ulmer
//         Created:  Thu Oct 11 14:53:32 CEST 2007
// $Id: OverlapValidation.cc,v 1.16 2009/11/04 19:40:27 kaulmer Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <utility>

using namespace std;
//
// class decleration
//

class OverlapValidation : public edm::one::EDAnalyzer<> {
public:
  explicit OverlapValidation(const edm::ParameterSet&);
  ~OverlapValidation() override;

private:
  typedef vector<Trajectory> TrajectoryCollection;

  edm::EDGetTokenT<TrajectoryCollection> trajectoryToken_;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  virtual void analyze(const Trajectory&, const Propagator&, TrackerHitAssociator&, const TrackerTopology* const tTopo);
  int layerFromId(const DetId&, const TrackerTopology* const tTopo) const;

  // ----------member data ---------------------------

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  edm::ParameterSet config_;
  edm::InputTag trajectoryTag_;
  SiStripDetInfoFileReader* reader;
  bool doSimHit_;
  const TrackerGeometry* trackerGeometry_;
  const MagneticField* magField_;

  TrajectoryStateCombiner combiner_;
  int overlapCounts_[3];

  TTree* rootTree_;
  edm::FileInPath FileInPath_;
  const int compressionSettings_;
  const bool addExtraBranches_;
  const int minHitsCut_;
  const float chi2ProbCut_;

  float overlapPath_;
  uint layer_;
  unsigned short hitCounts_[2];
  float chi2_[2];
  unsigned int overlapIds_[2];
  float predictedPositions_[3][2];
  float predictedLocalParameters_[5][2];
  float predictedLocalErrors_[5][2];
  float predictedDeltaXError_;
  float predictedDeltaYError_;
  char relativeXSign_;
  char relativeYSign_;
  float hitPositions_[2];
  float hitErrors_[2];
  float hitPositionsY_[2];
  float hitErrorsY_[2];
  float simHitPositions_[2];
  float simHitPositionsY_[2];
  float clusterWidthX_[2];
  float clusterWidthY_[2];
  float clusterSize_[2];
  uint clusterCharge_[2];
  int edge_[2];

  vector<bool> acceptLayer;
  float momentum_;
  uint run_, event_;
  bool barrelOnly_;

  //added by Heshy and Jared
  float moduleX_[2];
  float moduleY_[2];
  float moduleZ_[2];
  int subdetID;
  const Point2DBase<float, LocalTag> zerozero = Point2DBase<float, LocalTag>(0, 0);
  const Point2DBase<float, LocalTag> onezero = Point2DBase<float, LocalTag>(1, 0);
  const Point2DBase<float, LocalTag> zeroone = Point2DBase<float, LocalTag>(0, 1);
  //added by Jason
  float localxdotglobalphi_[2];
  float localxdotglobalr_[2];
  float localxdotglobalz_[2];
  float localxdotglobalx_[2];
  float localxdotglobaly_[2];
  float localydotglobalphi_[2];
  float localydotglobalr_[2];
  float localydotglobalz_[2];
  float localydotglobalx_[2];
  float localydotglobaly_[2];
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

using std::vector;
//
// constructors and destructor
//
OverlapValidation::OverlapValidation(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      magFieldToken_(esConsumes()),
      topoToken_(esConsumes()),
      config_(iConfig),
      rootTree_(nullptr),
      FileInPath_("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"),
      compressionSettings_(iConfig.getUntrackedParameter<int>("compressionSettings", -1)),
      addExtraBranches_(false),
      minHitsCut_(6),
      chi2ProbCut_(0.001) {
  edm::ConsumesCollector&& iC = consumesCollector();
  //now do what ever initialization is needed
  trajectoryTag_ = iConfig.getParameter<edm::InputTag>("trajectories");
  trajectoryToken_ = iC.consumes<TrajectoryCollection>(trajectoryTag_);
  doSimHit_ = iConfig.getParameter<bool>("associateStrip");
  reader = new SiStripDetInfoFileReader(FileInPath_.fullPath());

  overlapCounts_[0] = 0;  // #trajectories
  overlapCounts_[1] = 0;  // #hits
  overlapCounts_[2] = 0;  // #overlap hits
  acceptLayer.resize(7, false);
  acceptLayer[PixelSubdetector::PixelBarrel] = iConfig.getParameter<bool>("usePXB");
  acceptLayer[PixelSubdetector::PixelEndcap] = iConfig.getParameter<bool>("usePXF");
  acceptLayer[StripSubdetector::TIB] = iConfig.getParameter<bool>("useTIB");
  acceptLayer[StripSubdetector::TOB] = iConfig.getParameter<bool>("useTOB");
  acceptLayer[StripSubdetector::TID] = iConfig.getParameter<bool>("useTID");
  acceptLayer[StripSubdetector::TEC] = iConfig.getParameter<bool>("useTEC");
  barrelOnly_ = iConfig.getParameter<bool>("barrelOnly");

  edm::Service<TFileService> fs;
  //
  // root output
  //
  if (compressionSettings_ > 0) {
    fs->file().SetCompressionSettings(compressionSettings_);
  }

  rootTree_ = fs->make<TTree>("Overlaps", "Overlaps");
  if (addExtraBranches_) {
    rootTree_->Branch("hitCounts", hitCounts_, "found/s:lost/s");
    rootTree_->Branch("chi2", chi2_, "chi2/F:ndf/F");
    rootTree_->Branch("path", &overlapPath_, "path/F");
  }
  rootTree_->Branch("layer", &layer_, "layer/i");
  rootTree_->Branch("detids", overlapIds_, "id[2]/i");
  rootTree_->Branch("predPos", predictedPositions_, "gX[2]/F:gY[2]/F:gZ[2]/F");
  rootTree_->Branch("predPar", predictedLocalParameters_, "predQP[2]/F:predDX[2]/F:predDY[2]/F:predX[2]/F:predY[2]/F");
  rootTree_->Branch("predErr", predictedLocalErrors_, "predEQP[2]/F:predEDX[2]/F:predEDY[2]/F:predEX[2]/F:predEY[2]/F");
  rootTree_->Branch("predEDeltaX", &predictedDeltaXError_, "sigDeltaX/F");
  rootTree_->Branch("predEDeltaY", &predictedDeltaYError_, "sigDeltaY/F");
  rootTree_->Branch("relSignX", &relativeXSign_, "relSignX/B");
  rootTree_->Branch("relSignY", &relativeYSign_, "relSignY/B");
  rootTree_->Branch("hitX", hitPositions_, "hitX[2]/F");
  rootTree_->Branch("hitEX", hitErrors_, "hitEX[2]/F");
  rootTree_->Branch("hitY", hitPositionsY_, "hitY[2]/F");
  rootTree_->Branch("hitEY", hitErrorsY_, "hitEY[2]/F");
  if (addExtraBranches_) {
    rootTree_->Branch("simX", simHitPositions_, "simX[2]/F");
    rootTree_->Branch("simY", simHitPositionsY_, "simY[2]/F");
    rootTree_->Branch("clusterSize", clusterSize_, "clusterSize[2]/F");
    rootTree_->Branch("clusterWidthX", clusterWidthX_, "clusterWidthX[2]/F");
    rootTree_->Branch("clusterWidthY", clusterWidthY_, "clusterWidthY[2]/F");
    rootTree_->Branch("clusterCharge", clusterCharge_, "clusterCharge[2]/i");
    rootTree_->Branch("edge", edge_, "edge[2]/I");
  }
  rootTree_->Branch("momentum", &momentum_, "momentum/F");
  rootTree_->Branch("run", &run_, "run/i");
  rootTree_->Branch("event", &event_, "event/i");
  rootTree_->Branch("subdetID", &subdetID, "subdetID/I");
  rootTree_->Branch("moduleX", moduleX_, "moduleX[2]/F");
  rootTree_->Branch("moduleY", moduleY_, "moduleY[2]/F");
  rootTree_->Branch("moduleZ", moduleZ_, "moduleZ[2]/F");
  rootTree_->Branch("localxdotglobalphi", localxdotglobalphi_, "localxdotglobalphi[2]/F");
  rootTree_->Branch("localxdotglobalr", localxdotglobalr_, "localxdotglobalr[2]/F");
  rootTree_->Branch("localxdotglobalz", localxdotglobalz_, "localxdotglobalz[2]/F");
  rootTree_->Branch("localxdotglobalx", localxdotglobalx_, "localxdotglobalx[2]/F");
  rootTree_->Branch("localxdotglobaly", localxdotglobaly_, "localxdotglobaly[2]/F");
  rootTree_->Branch("localydotglobalphi", localydotglobalphi_, "localydotglobalphi[2]/F");
  rootTree_->Branch("localydotglobalr", localydotglobalr_, "localydotglobalr[2]/F");
  rootTree_->Branch("localydotglobalz", localydotglobalz_, "localydotglobalz[2]/F");
  rootTree_->Branch("localydotglobalx", localydotglobalx_, "localydotglobalx[2]/F");
  rootTree_->Branch("localydotglobaly", localydotglobaly_, "localydotglobaly[2]/F");
}

OverlapValidation::~OverlapValidation() {
  edm::LogWarning w("Overlaps");
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  w << "Counters =";
  w << " Number of tracks: " << overlapCounts_[0];
  w << " Number of valid hits: " << overlapCounts_[1];
  w << " Number of overlaps: " << overlapCounts_[2];
}

//
// member functions
//

// ------------ method called to for each event  ------------
void OverlapValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  //
  // mag field & search tracker
  //
  const MagneticField* magField_ = &iSetup.getData(magFieldToken_);
  //
  // propagator
  //
  AnalyticalPropagator propagator(magField_, anyDirection);
  //
  // geometry
  //
  trackerGeometry_ = &iSetup.getData(geomToken_);
  //
  // make associator for SimHits
  //
  TrackerHitAssociator* associator;
  if (doSimHit_) {
    TrackerHitAssociator::Config hitassociatorconfig(config_, consumesCollector());
    associator = new TrackerHitAssociator(iEvent, hitassociatorconfig);
  } else {
    associator = nullptr;
  }

  //if(doSimHit_) associator = new TrackerHitAssociator(iEvent, config_); else associator = 0;

  //
  // trajectories (from refit)
  //
  //typedef vector<Trajectory> TrajectoryCollection;
  edm::Handle<TrajectoryCollection> trajectoryCollectionHandle;
  iEvent.getByToken(trajectoryToken_, trajectoryCollectionHandle);
  const TrajectoryCollection* const trajectoryCollection = trajectoryCollectionHandle.product();

  //
  // loop over trajectories from refit
  const TrackerTopology* const tTopo = &iSetup.getData(topoToken_);
  for (const auto& trajectory : *trajectoryCollection)
    analyze(trajectory, propagator, *associator, tTopo);

  run_ = iEvent.id().run();
  event_ = iEvent.id().event();
}

void OverlapValidation::analyze(const Trajectory& trajectory,
                                const Propagator& propagator,
                                TrackerHitAssociator& associator,
                                const TrackerTopology* const tTopo) {
  typedef std::pair<const TrajectoryMeasurement*, const TrajectoryMeasurement*> Overlap;
  typedef vector<Overlap> OverlapContainer;
  ++overlapCounts_[0];

  OverlapContainer overlapHits;

  // quality cuts on trajectory
  // min. # hits / matched hits

  if (trajectory.foundHits() < minHitsCut_)
    return;
  if (ChiSquaredProbability((double)(trajectory.chiSquared()), (double)(trajectory.ndof(false))) < chi2ProbCut_)
    return;
  //
  // loop over measurements in the trajectory and calculate residuals
  //

  vector<TrajectoryMeasurement> measurements(trajectory.measurements());
  for (vector<TrajectoryMeasurement>::const_iterator itm = measurements.begin(); itm != measurements.end(); ++itm) {
    //
    // skip "invalid" (i.e. missing) hits
    //
    ConstRecHitPointer hit = itm->recHit();
    DetId id = hit->geographicalId();
    int layer(layerFromId(id, tTopo));
    int subDet = id.subdetId();

    if (!hit->isValid()) {
      edm::LogVerbatim("OverlapValidation") << "Invalid";
      continue;
    }
    if (barrelOnly_ && (subDet == StripSubdetector::TID || subDet == StripSubdetector::TEC))
      return;

    //edm::LogVerbatim("OverlapValidation")  << "Check " <<subDet << ", layer = " << layer<<" stereo: "<< ((subDet > 2)?(SiStripDetId(id).stereo()):2);
    //cout << "Check SubID " <<subDet << ", layer = " << layer<<" stereo: "<< ((subDet > 2)?(SiStripDetId(id).stereo()):2) << endl;

    //
    // check for overlap: same subdet-id && layer number for
    // two consecutive hits
    //
    ++overlapCounts_[1];
    if ((layer != -1) && (acceptLayer[subDet])) {
      for (vector<TrajectoryMeasurement>::const_iterator itmCompare = itm - 1;
           itmCompare >= measurements.begin() && itmCompare > itm - 4;
           --itmCompare) {
        DetId compareId = itmCompare->recHit()->geographicalId();

        if (subDet != compareId.subdetId() || layer != layerFromId(compareId, tTopo))
          break;
        if (!itmCompare->recHit()->isValid())
          continue;
        if ((subDet == PixelSubdetector::PixelBarrel || subDet == PixelSubdetector::PixelEndcap) ||
            (SiStripDetId(id).stereo() == SiStripDetId(compareId).stereo())) {
          overlapHits.push_back(std::make_pair(&(*itmCompare), &(*itm)));
          //edm::LogVerbatim("OverlapValidation") << "adding pair "<< ((subDet > 2)?(SiStripDetId(id).stereo()) : 2)
          //			     << " from layer = " << layer;
          //cout << "adding pair "<< ((subDet > 2)?(SiStripDetId(id).stereo()) : 2) << " from subDet = " << subDet << " and layer = " << layer;
          //cout << " \t"<<run_<< "\t"<<event_<<"\t";
          //cout << min(id.rawId(),compareId.rawId())<<"\t"<<max(id.rawId(),compareId.rawId())<<endl;
          if (SiStripDetId(id).glued() == id.rawId())
            edm::LogInfo("Overlaps") << "BAD GLUED: Have glued layer with id = " << id.rawId()
                                     << " and glued id = " << SiStripDetId(id).glued()
                                     << "  and stereo = " << SiStripDetId(id).stereo() << endl;
          if (SiStripDetId(compareId).glued() == compareId.rawId())
            edm::LogInfo("Overlaps") << "BAD GLUED: Have glued layer with id = " << compareId.rawId()
                                     << " and glued id = " << SiStripDetId(compareId).glued()
                                     << "  and stereo = " << SiStripDetId(compareId).stereo() << endl;
          break;
        }
      }
    }
  }

  //
  // Loop over all overlap pairs.
  //
  hitCounts_[0] = trajectory.foundHits();
  hitCounts_[1] = trajectory.lostHits();
  chi2_[0] = trajectory.chiSquared();
  chi2_[1] = trajectory.ndof(false);

  for (const auto& overlapHit : overlapHits) {
    //
    // create reference state @ module 1 (no info from overlap hits)
    //
    ++overlapCounts_[2];
    // backward predicted state at module 1
    TrajectoryStateOnSurface bwdPred1 = overlapHit.first->backwardPredictedState();
    if (!bwdPred1.isValid())
      continue;
    //cout << "momentum from backward predicted state = " << bwdPred1.globalMomentum().mag() << endl;
    // forward predicted state at module 2
    TrajectoryStateOnSurface fwdPred2 = overlapHit.second->forwardPredictedState();
    //cout << "momentum from forward predicted state = " << fwdPred2.globalMomentum().mag() << endl;
    if (!fwdPred2.isValid())
      continue;
    // extrapolate fwdPred2 to module 1
    TrajectoryStateOnSurface fwdPred2At1 = propagator.propagate(fwdPred2, bwdPred1.surface());
    if (!fwdPred2At1.isValid())
      continue;
    // combine fwdPred2At1 with bwdPred1 (ref. state, best estimate without hits 1 and 2)
    TrajectoryStateOnSurface comb1 = combiner_.combine(bwdPred1, fwdPred2At1);
    if (!comb1.isValid())
      continue;
    //
    // propagation of reference parameters to module 2
    //
    std::pair<TrajectoryStateOnSurface, double> tsosWithS = propagator.propagateWithPath(comb1, fwdPred2.surface());
    TrajectoryStateOnSurface comb1At2 = tsosWithS.first;
    if (!comb1At2.isValid())
      continue;
    //distance of propagation from one surface to the next==could cut here
    overlapPath_ = tsosWithS.second;
    if (abs(overlapPath_) > 15)
      continue;  //cut to remove hit pairs > 15 cm apart
    // global position on module 1
    GlobalPoint position = comb1.globalPosition();
    predictedPositions_[0][0] = position.x();
    predictedPositions_[1][0] = position.y();
    predictedPositions_[2][0] = position.z();
    momentum_ = comb1.globalMomentum().mag();
    //cout << "momentum from combination = " << momentum_ << endl;
    //cout << "magnetic field from TSOS = " << comb1.magneticField()->inTesla(position).mag() << endl;
    // local parameters and errors on module 1
    AlgebraicVector5 pars = comb1.localParameters().vector();
    AlgebraicSymMatrix55 errs = comb1.localError().matrix();
    for (int i = 0; i < 5; ++i) {
      predictedLocalParameters_[i][0] = pars[i];
      predictedLocalErrors_[i][0] = sqrt(errs(i, i));
    }
    // global position on module 2
    position = comb1At2.globalPosition();
    predictedPositions_[0][1] = position.x();
    predictedPositions_[1][1] = position.y();
    predictedPositions_[2][1] = position.z();
    // local parameters and errors on module 2
    pars = comb1At2.localParameters().vector();
    errs = comb1At2.localError().matrix();
    for (int i = 0; i < 5; ++i) {
      predictedLocalParameters_[i][1] = pars[i];
      predictedLocalErrors_[i][1] = sqrt(errs(i, i));
    }

    //print out local errors in X to check
    //cout << "Predicted local error in X at 1 = " << predictedLocalErrors_[3][0] << "   and predicted local error in X at 2 is = " <<  predictedLocalErrors_[3][1] << endl;
    //cout << "Predicted local error in Y at 1 = " << predictedLocalErrors_[4][0] << "   and predicted local error in Y at 2 is = " <<  predictedLocalErrors_[4][1] << endl;

    //
    // jacobians (local-to-global@1,global 1-2,global-to-local@2)
    //
    JacobianLocalToCurvilinear jacLocToCurv(comb1.surface(), comb1.localParameters(), *magField_);
    AnalyticalCurvilinearJacobian jacCurvToCurv(
        comb1.globalParameters(), comb1At2.globalPosition(), comb1At2.globalMomentum(), tsosWithS.second);
    JacobianCurvilinearToLocal jacCurvToLoc(comb1At2.surface(), comb1At2.localParameters(), *magField_);
    // combined jacobian local-1-to-local-2
    AlgebraicMatrix55 jacobian = jacLocToCurv.jacobian() * jacCurvToCurv.jacobian() * jacCurvToLoc.jacobian();
    // covariance on module 1
    AlgebraicSymMatrix55 covComb1 = comb1.localError().matrix();
    // variance and correlations for predicted local_x on modules 1 and 2
    double c00 = covComb1(3, 3);
    double c10(0.);
    double c11(0.);
    for (int i = 1; i < 5; ++i) {
      c10 += jacobian(3, i) * covComb1(i, 3);
      for (int j = 1; j < 5; ++j)
        c11 += jacobian(3, i) * covComb1(i, j) * jacobian(3, j);
    }
    // choose relative sign in order to minimize error on difference
    double diff = c00 - 2 * fabs(c10) + c11;
    diff = diff > 0 ? sqrt(diff) : -sqrt(-diff);
    predictedDeltaXError_ = diff;
    relativeXSign_ = c10 > 0 ? -1 : 1;
    //
    // now find variance and correlations for predicted local_y
    double c00Y = covComb1(4, 4);
    double c10Y(0.);
    double c11Y(0.);
    for (int i = 1; i < 5; ++i) {
      c10Y += jacobian(4, i) * covComb1(i, 4);
      for (int j = 1; j < 5; ++j)
        c11Y += jacobian(4, i) * covComb1(i, j) * jacobian(4, j);
    }
    double diffY = c00Y - 2 * fabs(c10Y) + c11Y;
    diffY = diffY > 0 ? sqrt(diffY) : -sqrt(-diffY);
    predictedDeltaYError_ = diffY;
    relativeYSign_ = c10Y > 0 ? -1 : 1;

    // information on modules and hits
    overlapIds_[0] = overlapHit.first->recHit()->geographicalId().rawId();
    overlapIds_[1] = overlapHit.second->recHit()->geographicalId().rawId();

    //added by Heshy and Jared
    moduleX_[0] = overlapHit.first->recHit()->det()->surface().position().x();
    moduleX_[1] = overlapHit.second->recHit()->det()->surface().position().x();
    moduleY_[0] = overlapHit.first->recHit()->det()->surface().position().y();
    moduleY_[1] = overlapHit.second->recHit()->det()->surface().position().y();
    moduleZ_[0] = overlapHit.first->recHit()->det()->surface().position().z();
    moduleZ_[1] = overlapHit.second->recHit()->det()->surface().position().z();
    subdetID = overlapHit.first->recHit()->geographicalId().subdetId();
    localxdotglobalphi_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(onezero).phi() -
                             overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).phi();
    localxdotglobalphi_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(onezero).phi() -
                             overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).phi();
    //added by Jason
    localxdotglobalr_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(onezero).perp() -
                           overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).perp();
    localxdotglobalr_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(onezero).perp() -
                           overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).perp();
    localxdotglobalz_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(onezero).z() -
                           overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).z();
    localxdotglobalz_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(onezero).z() -
                           overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).z();
    localxdotglobalx_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(onezero).x() -
                           overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).x();
    localxdotglobalx_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(onezero).x() -
                           overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).x();
    localxdotglobaly_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(onezero).y() -
                           overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).y();
    localxdotglobaly_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(onezero).y() -
                           overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).y();
    localydotglobalr_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(zeroone).perp() -
                           overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).perp();
    localydotglobalr_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(zeroone).perp() -
                           overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).perp();
    localydotglobalz_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(zeroone).z() -
                           overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).z();
    localydotglobalz_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(zeroone).z() -
                           overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).z();
    localydotglobalx_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(zeroone).x() -
                           overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).x();
    localydotglobalx_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(zeroone).x() -
                           overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).x();
    localydotglobaly_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(zeroone).y() -
                           overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).y();
    localydotglobaly_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(zeroone).y() -
                           overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).y();
    localydotglobalphi_[0] = overlapHit.first->recHit()->det()->surface().toGlobal(zeroone).phi() -
                             overlapHit.first->recHit()->det()->surface().toGlobal(zerozero).phi();
    localydotglobalphi_[1] = overlapHit.second->recHit()->det()->surface().toGlobal(zeroone).phi() -
                             overlapHit.second->recHit()->det()->surface().toGlobal(zerozero).phi();

    if (overlapHit.first->recHit()->geographicalId().subdetId() == StripSubdetector::TIB)
      layer_ = layerFromId(overlapHit.first->recHit()->geographicalId().rawId(), tTopo);
    else if (overlapHit.first->recHit()->geographicalId().subdetId() == StripSubdetector::TOB)
      layer_ = layerFromId(overlapHit.first->recHit()->geographicalId().rawId(), tTopo) + 4;
    else if (overlapHit.first->recHit()->geographicalId().subdetId() == StripSubdetector::TID)
      layer_ = layerFromId(overlapHit.first->recHit()->geographicalId().rawId(), tTopo) + 10;
    else if (overlapHit.first->recHit()->geographicalId().subdetId() == StripSubdetector::TEC)
      layer_ = layerFromId(overlapHit.first->recHit()->geographicalId().rawId(), tTopo) + 13;
    else if (overlapHit.first->recHit()->geographicalId().subdetId() == PixelSubdetector::PixelBarrel)
      layer_ = layerFromId(overlapHit.first->recHit()->geographicalId().rawId(), tTopo) + 20;
    else if (overlapHit.first->recHit()->geographicalId().subdetId() == PixelSubdetector::PixelEndcap)
      layer_ = layerFromId(overlapHit.first->recHit()->geographicalId().rawId(), tTopo) + 30;
    else
      layer_ = 99;

    if (overlapIds_[0] == SiStripDetId(overlapHit.first->recHit()->geographicalId()).glued())
      edm::LogWarning("Overlaps") << "BAD GLUED: First Id = " << overlapIds_[0] << " has glued = "
                                  << SiStripDetId(overlapHit.first->recHit()->geographicalId()).glued()
                                  << "  and stereo = "
                                  << SiStripDetId(overlapHit.first->recHit()->geographicalId()).stereo() << endl;
    if (overlapIds_[1] == SiStripDetId(overlapHit.second->recHit()->geographicalId()).glued())
      edm::LogWarning("Overlaps") << "BAD GLUED: Second Id = " << overlapIds_[1] << " has glued = "
                                  << SiStripDetId(overlapHit.second->recHit()->geographicalId()).glued()
                                  << "  and stereo = "
                                  << SiStripDetId(overlapHit.second->recHit()->geographicalId()).stereo() << endl;

    const TransientTrackingRecHit::ConstRecHitPointer firstRecHit = overlapHit.first->recHit();
    const TransientTrackingRecHit::ConstRecHitPointer secondRecHit = overlapHit.second->recHit();
    hitPositions_[0] = firstRecHit->localPosition().x();
    hitErrors_[0] = sqrt(firstRecHit->localPositionError().xx());
    hitPositions_[1] = secondRecHit->localPosition().x();
    hitErrors_[1] = sqrt(secondRecHit->localPositionError().xx());

    hitPositionsY_[0] = firstRecHit->localPosition().y();
    hitErrorsY_[0] = sqrt(firstRecHit->localPositionError().yy());
    hitPositionsY_[1] = secondRecHit->localPosition().y();
    hitErrorsY_[1] = sqrt(secondRecHit->localPositionError().yy());

    //cout << "printing local X hit position and error for the overlap hits. Hit 1 = " << hitPositions_[0] << "+-" << hitErrors_[0] << "  and hit 2 is "  << hitPositions_[1] << "+-" << hitErrors_[1] << endl;

    DetId id1 = overlapHit.first->recHit()->geographicalId();
    DetId id2 = overlapHit.second->recHit()->geographicalId();
    int layer1 = layerFromId(id1, tTopo);
    int subDet1 = id1.subdetId();
    int layer2 = layerFromId(id2, tTopo);
    int subDet2 = id2.subdetId();
    if (abs(hitPositions_[0]) > 5)
      edm::LogInfo("Overlaps") << "BAD: Bad hit position: Id = " << id1.rawId()
                               << " stereo = " << SiStripDetId(id1).stereo()
                               << "  glued = " << SiStripDetId(id1).glued() << " from subdet = " << subDet1
                               << " and layer = " << layer1 << endl;
    if (abs(hitPositions_[1]) > 5)
      edm::LogInfo("Overlaps") << "BAD: Bad hit position: Id = " << id2.rawId()
                               << " stereo = " << SiStripDetId(id2).stereo()
                               << "  glued = " << SiStripDetId(id2).glued() << " from subdet = " << subDet2
                               << " and layer = " << layer2 << endl;

    // get track momentum
    momentum_ = comb1.globalMomentum().mag();

    // get cluster size
    if (!(subDet1 == PixelSubdetector::PixelBarrel || subDet1 == PixelSubdetector::PixelEndcap)) {  //strip
      const TransientTrackingRecHit::ConstRecHitPointer thit1 = overlapHit.first->recHit();
      const SiStripRecHit1D* hit1 = dynamic_cast<const SiStripRecHit1D*>((*thit1).hit());
      if (hit1) {
        //check cluster width
        const SiStripRecHit1D::ClusterRef& cluster1 = hit1->cluster();
        clusterSize_[0] = (cluster1->amplitudes()).size();
        clusterWidthX_[0] = (cluster1->amplitudes()).size();
        clusterWidthY_[0] = -1;

        //check if cluster at edge of sensor
        uint16_t firstStrip = cluster1->firstStrip();
        uint16_t lastStrip = firstStrip + (cluster1->amplitudes()).size() - 1;
        unsigned short Nstrips;
        Nstrips = reader->getNumberOfApvsAndStripLength(id1).first * 128;
        bool atEdge = false;
        if (firstStrip == 0 || lastStrip == (Nstrips - 1))
          atEdge = true;
        if (atEdge)
          edge_[0] = 1;
        else
          edge_[0] = -1;

        // get cluster total charge
        const auto& stripCharges = cluster1->amplitudes();
        uint16_t charge = 0;
        for (uint i = 0; i < stripCharges.size(); i++) {
          charge += stripCharges[i];
        }
        clusterCharge_[0] = charge;
      }

      const TransientTrackingRecHit::ConstRecHitPointer thit2 = overlapHit.second->recHit();
      const SiStripRecHit1D* hit2 = dynamic_cast<const SiStripRecHit1D*>((*thit2).hit());
      if (hit2) {
        const SiStripRecHit1D::ClusterRef& cluster2 = hit2->cluster();
        clusterSize_[1] = (cluster2->amplitudes()).size();
        clusterWidthX_[1] = (cluster2->amplitudes()).size();
        clusterWidthY_[1] = -1;

        uint16_t firstStrip = cluster2->firstStrip();
        uint16_t lastStrip = firstStrip + (cluster2->amplitudes()).size() - 1;
        unsigned short Nstrips;
        Nstrips = reader->getNumberOfApvsAndStripLength(id2).first * 128;
        bool atEdge = false;
        if (firstStrip == 0 || lastStrip == (Nstrips - 1))
          atEdge = true;
        if (atEdge)
          edge_[1] = 1;
        else
          edge_[1] = -1;

        // get cluster total charge
        const auto& stripCharges = cluster2->amplitudes();
        uint16_t charge = 0;
        for (uint i = 0; i < stripCharges.size(); i++) {
          charge += stripCharges[i];
        }
        clusterCharge_[1] = charge;
      }
      //cout << "strip cluster size2 = " << clusterWidthX_[0] << "  and size 2 = " << clusterWidthX_[1] << endl;
    }

    if (subDet2 == PixelSubdetector::PixelBarrel || subDet2 == PixelSubdetector::PixelEndcap) {  //pixel

      const TransientTrackingRecHit::ConstRecHitPointer thit1 = overlapHit.first->recHit();
      const SiPixelRecHit* recHitPix1 = dynamic_cast<const SiPixelRecHit*>((*thit1).hit());
      if (recHitPix1) {
        // check for cluster size and width
        SiPixelRecHit::ClusterRef const& cluster1 = recHitPix1->cluster();

        clusterSize_[0] = cluster1->size();
        clusterWidthX_[0] = cluster1->sizeX();
        clusterWidthY_[0] = cluster1->sizeY();

        // check for cluster at edge
        const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>((*trackerGeometry_).idToDet(id1));
        const PixelTopology* topol = (&(theGeomDet->specificTopology()));

        int minPixelRow = cluster1->minPixelRow();  //x
        int maxPixelRow = cluster1->maxPixelRow();
        int minPixelCol = cluster1->minPixelCol();  //y
        int maxPixelCol = cluster1->maxPixelCol();

        bool edgeHitX = (topol->isItEdgePixelInX(minPixelRow)) || (topol->isItEdgePixelInX(maxPixelRow));
        bool edgeHitY = (topol->isItEdgePixelInY(minPixelCol)) || (topol->isItEdgePixelInY(maxPixelCol));
        if (edgeHitX || edgeHitY)
          edge_[0] = 1;
        else
          edge_[0] = -1;

        clusterCharge_[0] = (uint)cluster1->charge();

      } else {
        edm::LogWarning("Overlaps") << "didn't find pixel cluster" << endl;
        continue;
      }

      const TransientTrackingRecHit::ConstRecHitPointer thit2 = overlapHit.second->recHit();
      const SiPixelRecHit* recHitPix2 = dynamic_cast<const SiPixelRecHit*>((*thit2).hit());
      if (recHitPix2) {
        SiPixelRecHit::ClusterRef const& cluster2 = recHitPix2->cluster();

        clusterSize_[1] = cluster2->size();
        clusterWidthX_[1] = cluster2->sizeX();
        clusterWidthY_[1] = cluster2->sizeY();
        //cout << "second pixel cluster is " << clusterSize_[1] << " pixels with x width = " << clusterWidthX_[1] << " and y width = " << clusterWidthY_[1] << endl;

        const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>((*trackerGeometry_).idToDet(id2));
        const PixelTopology* topol = (&(theGeomDet->specificTopology()));

        int minPixelRow = cluster2->minPixelRow();  //x
        int maxPixelRow = cluster2->maxPixelRow();
        int minPixelCol = cluster2->minPixelCol();  //y
        int maxPixelCol = cluster2->maxPixelCol();

        bool edgeHitX = (topol->isItEdgePixelInX(minPixelRow)) || (topol->isItEdgePixelInX(maxPixelRow));
        bool edgeHitY = (topol->isItEdgePixelInY(minPixelCol)) || (topol->isItEdgePixelInY(maxPixelCol));
        if (edgeHitX || edgeHitY)
          edge_[1] = 1;
        else
          edge_[1] = -1;

        clusterCharge_[1] = (uint)cluster2->charge();

      } else {
        edm::LogWarning("Overlaps") << "didn't find pixel cluster" << endl;
        continue;
      }
    }

    //also check for edge pixels

    //try writing out the SimHit info (for MC only)
    if (doSimHit_) {
      std::vector<PSimHit> psimHits1;
      std::vector<PSimHit> psimHits2;
      //calculate layer
      DetId id = overlapHit.first->recHit()->geographicalId();
      int layer(-1);
      layer = layerFromId(id, tTopo);
      int subDet = id.subdetId();
      edm::LogVerbatim("OverlapValidation") << "Subdet = " << subDet << " ; layer = " << layer;

      psimHits1 = associator.associateHit(*(firstRecHit->hit()));
      edm::LogVerbatim("OverlapValidation") << "single hit ";
      edm::LogVerbatim("OverlapValidation") << "length of psimHits1: " << psimHits1.size();
      if (!psimHits1.empty()) {
        float closest_dist = 99999.9;
        std::vector<PSimHit>::const_iterator closest_simhit = psimHits1.begin();
        for (std::vector<PSimHit>::const_iterator m = psimHits1.begin(); m < psimHits1.end(); m++) {
          //find closest simHit to the recHit
          float simX = (*m).localPosition().x();
          float dist = fabs(simX - (overlapHit.first->recHit()->localPosition().x()));
          edm::LogVerbatim("OverlapValidation")
              << "simHit1 simX = " << simX << "   hitX = " << overlapHit.first->recHit()->localPosition().x()
              << "   distX = " << dist << "   layer = " << layer;
          if (dist < closest_dist) {
            //cout << "found newest closest dist for simhit1" << endl;
            closest_dist = dist;
            closest_simhit = m;
          }
        }
        //if glued layer, convert sim hit position to matchedhit surface
        //layer index from 1-4 for TIB, 1-6 for TOB
        // Are the sim hits on the glued layers or are they split???
        if (subDet > 2 && !SiStripDetId(id).glued()) {
          const GluedGeomDet* gluedDet =
              (const GluedGeomDet*)(*trackerGeometry_).idToDet((*firstRecHit).hit()->geographicalId());
          const StripGeomDetUnit* stripDet = (StripGeomDetUnit*)gluedDet->monoDet();
          GlobalPoint gp = stripDet->surface().toGlobal((*closest_simhit).localPosition());
          LocalPoint lp = gluedDet->surface().toLocal(gp);
          LocalVector localdirection = (*closest_simhit).localDirection();
          GlobalVector globaldirection = stripDet->surface().toGlobal(localdirection);
          LocalVector direction = gluedDet->surface().toLocal(globaldirection);
          float scale = -lp.z() / direction.z();
          LocalPoint projectedPos = lp + scale * direction;
          simHitPositions_[0] = projectedPos.x();
          edm::LogVerbatim("OverlapValidation") << "simhit position from matched layer = " << simHitPositions_[0];
          simHitPositionsY_[0] = projectedPos.y();
        } else {
          simHitPositions_[0] = (*closest_simhit).localPosition().x();
          simHitPositionsY_[0] = (*closest_simhit).localPosition().y();
          edm::LogVerbatim("OverlapValidation") << "simhit position from non-matched layer = " << simHitPositions_[0];
        }
        edm::LogVerbatim("OverlapValidation") << "hit position = " << hitPositions_[0];
      } else {
        simHitPositions_[0] = -99.;
        simHitPositionsY_[0] = -99.;
        //cout << " filling simHitX: " << -99 << endl;
      }

      psimHits2 = associator.associateHit(*(secondRecHit->hit()));
      if (!psimHits2.empty()) {
        float closest_dist = 99999.9;
        std::vector<PSimHit>::const_iterator closest_simhit = psimHits2.begin();
        for (std::vector<PSimHit>::const_iterator m = psimHits2.begin(); m < psimHits2.end(); m++) {
          float simX = (*m).localPosition().x();
          float dist = fabs(simX - (overlapHit.second->recHit()->localPosition().x()));
          if (dist < closest_dist) {
            closest_dist = dist;
            closest_simhit = m;
          }
        }
        //if glued layer, convert sim hit position to matchedhit surface
        // if no sim hits on matched layers then this section can be removed
        if (subDet > 2 && !SiStripDetId(id).glued()) {
          const GluedGeomDet* gluedDet =
              (const GluedGeomDet*)(*trackerGeometry_).idToDet((*secondRecHit).hit()->geographicalId());
          const StripGeomDetUnit* stripDet = (StripGeomDetUnit*)gluedDet->monoDet();
          GlobalPoint gp = stripDet->surface().toGlobal((*closest_simhit).localPosition());
          LocalPoint lp = gluedDet->surface().toLocal(gp);
          LocalVector localdirection = (*closest_simhit).localDirection();
          GlobalVector globaldirection = stripDet->surface().toGlobal(localdirection);
          LocalVector direction = gluedDet->surface().toLocal(globaldirection);
          float scale = -lp.z() / direction.z();
          LocalPoint projectedPos = lp + scale * direction;
          simHitPositions_[1] = projectedPos.x();
          simHitPositionsY_[1] = projectedPos.y();
        } else {
          simHitPositions_[1] = (*closest_simhit).localPosition().x();
          simHitPositionsY_[1] = (*closest_simhit).localPosition().y();
        }
      } else {
        simHitPositions_[1] = -99.;
        simHitPositionsY_[1] = -99.;
      }
    }
    rootTree_->Fill();
  }
}

int OverlapValidation::layerFromId(const DetId& id, const TrackerTopology* const tTopo) const {
  TrackerAlignableId aliid;
  std::pair<int, int> subdetandlayer = aliid.typeAndLayerFromDetId(id, tTopo);
  int layer = subdetandlayer.second;

  return layer;
}

void OverlapValidation::endJob() {
  if (rootTree_) {
    rootTree_->GetDirectory()->cd();
    rootTree_->Write();
    delete rootTree_;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(OverlapValidation);
