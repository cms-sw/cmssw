#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "CondFormats/Alignment/interface/Definitions.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"

#include "TMath.h"

#include <string>

TrackerValidationVariables::TrackerValidationVariables(const edm::ParameterSet& config, edm::ConsumesCollector&& iC)
    : magneticFieldToken_{iC.esConsumes<MagneticField, IdealMagneticFieldRecord>()} {
  trajCollectionToken_ =
      iC.consumes<std::vector<Trajectory>>(edm::InputTag(config.getParameter<std::string>("trajectoryInput")));
  tracksToken_ = iC.consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("Tracks"));
}

TrackerValidationVariables::~TrackerValidationVariables() {}

void TrackerValidationVariables::fillHitQuantities(reco::Track const& track, std::vector<AVHitStruct>& v_avhitout) {
  auto const& trajParams = track.extra()->trajParams();
  auto const& residuals = track.extra()->residuals();

  assert(trajParams.size() == track.recHitsSize());
  auto hb = track.recHitsBegin();
  for (unsigned int h = 0; h < track.recHitsSize(); h++) {
    auto hit = *(hb + h);
    if (!hit->isValid())
      continue;

    AVHitStruct hitStruct;
    const DetId& hit_detId = hit->geographicalId();
    auto IntRawDetID = hit_detId.rawId();
    auto IntSubDetID = hit_detId.subdetId();

    if (IntSubDetID == 0)
      continue;

    if (IntSubDetID == PixelSubdetector::PixelBarrel || IntSubDetID == PixelSubdetector::PixelEndcap) {
      const SiPixelRecHit* prechit = dynamic_cast<const SiPixelRecHit*>(
          hit);  //to be used to get the associated cluster and the cluster probability
      if (prechit->isOnEdge())
        hitStruct.isOnEdgePixel = true;
      if (prechit->hasBadPixels())
        hitStruct.isOtherBadPixel = true;
    }

    auto lPTrk = trajParams[h].position();  // update state
    auto lVTrk = trajParams[h].direction();

    auto gtrkdirup = hit->surface()->toGlobal(lVTrk);

    hitStruct.rawDetId = IntRawDetID;
    hitStruct.phi = gtrkdirup.phi();  // direction, not position
    hitStruct.eta = gtrkdirup.eta();  // same

    hitStruct.localAlpha = std::atan2(lVTrk.x(), lVTrk.z());  // wrt. normal tg(alpha)=x/z
    hitStruct.localBeta = std::atan2(lVTrk.y(), lVTrk.z());   // wrt. normal tg(beta)= y/z

    hitStruct.resX = residuals.residualX(h);
    hitStruct.resY = residuals.residualY(h);
    hitStruct.resErrX = hitStruct.resX / residuals.pullX(h);  // for backward compatibility....
    hitStruct.resErrY = hitStruct.resY / residuals.pullY(h);

    // hitStruct.localX = lPhit.x();
    // hitStruct.localY = lPhit.y();
    // EM: use predictions for local coordinates
    hitStruct.localX = lPTrk.x();
    hitStruct.localY = lPTrk.y();

    // now calculate residuals taking global orientation of modules and radial topology in TID/TEC into account
    float resXprime(999.F), resYprime(999.F);
    float resXatTrkY(999.F);
    float resXprimeErr(999.F), resYprimeErr(999.F);

    if (hit->detUnit()) {  // is it a single physical module?
      float uOrientation(-999.F), vOrientation(-999.F);
      float resXTopol(999.F), resYTopol(999.F);
      float resXatTrkYTopol(999.F);

      const Surface& surface = hit->detUnit()->surface();
      const BoundPlane& boundplane = hit->detUnit()->surface();
      const Bounds& bound = boundplane.bounds();

      float length = 0;
      float width = 0;

      LocalPoint lPModule(0., 0., 0.), lUDirection(1., 0., 0.), lVDirection(0., 1., 0.);
      GlobalPoint gPModule = surface.toGlobal(lPModule), gUDirection = surface.toGlobal(lUDirection),
                  gVDirection = surface.toGlobal(lVDirection);

      if (IntSubDetID == PixelSubdetector::PixelBarrel || IntSubDetID == PixelSubdetector::PixelEndcap ||
          IntSubDetID == StripSubdetector::TIB || IntSubDetID == StripSubdetector::TOB) {
        if (IntSubDetID == PixelSubdetector::PixelEndcap) {
          uOrientation = gUDirection.perp() - gPModule.perp() >= 0 ? +1.F : -1.F;
          vOrientation = deltaPhi(gVDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
        } else {
          uOrientation = deltaPhi(gUDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
          vOrientation = gVDirection.z() - gPModule.z() >= 0 ? +1.F : -1.F;
        }

        resXTopol = hitStruct.resX;
        resXatTrkYTopol = hitStruct.resX;
        resYTopol = hitStruct.resY;
        resXprimeErr = hitStruct.resErrX;
        resYprimeErr = hitStruct.resErrY;

        const RectangularPlaneBounds* rectangularBound = dynamic_cast<const RectangularPlaneBounds*>(&bound);
        if (rectangularBound != nullptr) {
          hitStruct.inside = rectangularBound->inside(lPTrk);
          length = rectangularBound->length();
          width = rectangularBound->width();
          hitStruct.localXnorm = 2 * hitStruct.localX / width;
          hitStruct.localYnorm = 2 * hitStruct.localY / length;
        } else {
          throw cms::Exception("Geometry Error")
              << "[TrackerValidationVariables] Cannot cast bounds to RectangularPlaneBounds as expected for TPE";
        }

      } else if (IntSubDetID == StripSubdetector::TID || IntSubDetID == StripSubdetector::TEC) {
        // not possible to compute precisely as with Trajectory
      } else {
        edm::LogWarning("TrackerValidationVariables") << "@SUB=TrackerValidationVariables::fillHitQuantities"
                                                      << "No valid tracker subdetector " << IntSubDetID;
        continue;
      }

      resXprime = resXTopol * uOrientation;
      resXatTrkY = resXatTrkYTopol;
      resYprime = resYTopol * vOrientation;

    } else {  // not a detUnit, so must be a virtual 2D-Module
      // FIXME: at present only for det units residuals are calculated and filled in the hitStruct
      // But in principle this method should also be useable for the gluedDets (2D modules in TIB, TID, TOB, TEC)
      // In this case, only orientation should be taken into account for primeResiduals, but not the radial topology
      // At present, default values (999.F) are given out
    }

    hitStruct.resXprime = resXprime;
    hitStruct.resXatTrkY = resXatTrkY;
    hitStruct.resYprime = resYprime;
    hitStruct.resXprimeErr = resXprimeErr;
    hitStruct.resYprimeErr = resYprimeErr;

    v_avhitout.push_back(hitStruct);
  }
}

void TrackerValidationVariables::fillHitQuantities(const Trajectory* trajectory, std::vector<AVHitStruct>& v_avhitout) {
  TrajectoryStateCombiner tsoscomb;

  const std::vector<TrajectoryMeasurement>& tmColl = trajectory->measurements();
  for (std::vector<TrajectoryMeasurement>::const_iterator itTraj = tmColl.begin(); itTraj != tmColl.end(); ++itTraj) {
    if (!itTraj->updatedState().isValid())
      continue;

    TrajectoryStateOnSurface tsos = tsoscomb(itTraj->forwardPredictedState(), itTraj->backwardPredictedState());
    if (!tsos.isValid())
      continue;
    TransientTrackingRecHit::ConstRecHitPointer hit = itTraj->recHit();

    if (!hit->isValid() || hit->geographicalId().det() != DetId::Tracker)
      continue;

    AVHitStruct hitStruct;
    const DetId& hit_detId = hit->geographicalId();
    unsigned int IntRawDetID = (hit_detId.rawId());
    unsigned int IntSubDetID = (hit_detId.subdetId());

    if (IntSubDetID == 0)
      continue;

    if (IntSubDetID == PixelSubdetector::PixelBarrel || IntSubDetID == PixelSubdetector::PixelEndcap) {
      const SiPixelRecHit* prechit = dynamic_cast<const SiPixelRecHit*>(
          hit.get());  //to be used to get the associated cluster and the cluster probability
      if (prechit->isOnEdge())
        hitStruct.isOnEdgePixel = true;
      if (prechit->hasBadPixels())
        hitStruct.isOtherBadPixel = true;
    }

    //first calculate residuals in cartesian coordinates in the local module coordinate system

    LocalPoint lPHit = hit->localPosition();
    LocalPoint lPTrk = tsos.localPosition();
    LocalVector lVTrk = tsos.localDirection();

    hitStruct.localAlpha = atan2(lVTrk.x(), lVTrk.z());  // wrt. normal tg(alpha)=x/z
    hitStruct.localBeta = atan2(lVTrk.y(), lVTrk.z());   // wrt. normal tg(beta)= y/z

    LocalError errHit = hit->localPositionError();
    // no need to add  APE to hitError anymore
    // AlgebraicROOTObject<2>::SymMatrix mat = asSMatrix<2>(hit->parametersError());
    // LocalError errHit = LocalError( mat(0,0),mat(0,1),mat(1,1) );
    LocalError errTrk = tsos.localError().positionError();

    //check for negative error values: track error can have negative value, if matrix inversion fails (very rare case)
    //hit error should always give positive values
    if (errHit.xx() < 0. || errHit.yy() < 0. || errTrk.xx() < 0. || errTrk.yy() < 0.) {
      edm::LogError("TrackerValidationVariables")
          << "@SUB=TrackerValidationVariables::fillHitQuantities"
          << "One of the squared error methods gives negative result"
          << "\n\terrHit.xx()\terrHit.yy()\terrTrk.xx()\terrTrk.yy()"
          << "\n\t" << errHit.xx() << "\t" << errHit.yy() << "\t" << errTrk.xx() << "\t" << errTrk.yy();
      continue;
    }

    align::LocalVector res = lPTrk - lPHit;

    float resXErr = std::sqrt(errHit.xx() + errTrk.xx());
    float resYErr = std::sqrt(errHit.yy() + errTrk.yy());

    hitStruct.resX = res.x();
    hitStruct.resY = res.y();
    hitStruct.resErrX = resXErr;
    hitStruct.resErrY = resYErr;

    // hitStruct.localX = lPhit.x();
    // hitStruct.localY = lPhit.y();
    // EM: use predictions for local coordinates
    hitStruct.localX = lPTrk.x();
    hitStruct.localY = lPTrk.y();

    // now calculate residuals taking global orientation of modules and radial topology in TID/TEC into account
    float resXprime(999.F), resYprime(999.F);
    float resXatTrkY(999.F);
    float resXprimeErr(999.F), resYprimeErr(999.F);

    if (hit->detUnit()) {  // is it a single physical module?
      const GeomDetUnit& detUnit = *(hit->detUnit());
      float uOrientation(-999.F), vOrientation(-999.F);
      float resXTopol(999.F), resYTopol(999.F);
      float resXatTrkYTopol(999.F);

      const Surface& surface = hit->detUnit()->surface();
      const BoundPlane& boundplane = hit->detUnit()->surface();
      const Bounds& bound = boundplane.bounds();

      float length = 0;
      float width = 0;

      LocalPoint lPModule(0., 0., 0.), lUDirection(1., 0., 0.), lVDirection(0., 1., 0.);
      GlobalPoint gPModule = surface.toGlobal(lPModule), gUDirection = surface.toGlobal(lUDirection),
                  gVDirection = surface.toGlobal(lVDirection);

      if (IntSubDetID == PixelSubdetector::PixelBarrel || IntSubDetID == StripSubdetector::TIB ||
          IntSubDetID == StripSubdetector::TOB) {
        uOrientation = deltaPhi(gUDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
        vOrientation = gVDirection.z() - gPModule.z() >= 0 ? +1.F : -1.F;
        resXTopol = res.x();
        resXatTrkYTopol = res.x();
        resYTopol = res.y();
        resXprimeErr = resXErr;
        resYprimeErr = resYErr;

        const RectangularPlaneBounds* rectangularBound = dynamic_cast<const RectangularPlaneBounds*>(&bound);
        if (rectangularBound != nullptr) {
          hitStruct.inside = rectangularBound->inside(lPTrk);
          length = rectangularBound->length();
          width = rectangularBound->width();
          hitStruct.localXnorm = 2 * hitStruct.localX / width;
          hitStruct.localYnorm = 2 * hitStruct.localY / length;
        } else {
          throw cms::Exception("Geometry Error") << "[TrackerValidationVariables] Cannot cast bounds to "
                                                    "RectangularPlaneBounds as expected for TPB, TIB and TOB";
        }

      } else if (IntSubDetID == PixelSubdetector::PixelEndcap) {
        uOrientation = gUDirection.perp() - gPModule.perp() >= 0 ? +1.F : -1.F;
        vOrientation = deltaPhi(gVDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
        resXTopol = res.x();
        resXatTrkYTopol = res.x();
        resYTopol = res.y();
        resXprimeErr = resXErr;
        resYprimeErr = resYErr;

        const RectangularPlaneBounds* rectangularBound = dynamic_cast<const RectangularPlaneBounds*>(&bound);
        if (rectangularBound != nullptr) {
          hitStruct.inside = rectangularBound->inside(lPTrk);
          length = rectangularBound->length();
          width = rectangularBound->width();
          hitStruct.localXnorm = 2 * hitStruct.localX / width;
          hitStruct.localYnorm = 2 * hitStruct.localY / length;
        } else {
          throw cms::Exception("Geometry Error")
              << "[TrackerValidationVariables] Cannot cast bounds to RectangularPlaneBounds as expected for TPE";
        }

      } else if (IntSubDetID == StripSubdetector::TID || IntSubDetID == StripSubdetector::TEC) {
        uOrientation = deltaPhi(gUDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
        vOrientation = gVDirection.perp() - gPModule.perp() >= 0. ? +1.F : -1.F;

        if (!dynamic_cast<const RadialStripTopology*>(&detUnit.type().topology()))
          continue;
        const RadialStripTopology& topol = dynamic_cast<const RadialStripTopology&>(detUnit.type().topology());

        MeasurementPoint measHitPos = topol.measurementPosition(lPHit);
        MeasurementPoint measTrkPos = topol.measurementPosition(lPTrk);

        MeasurementError measHitErr = topol.measurementError(lPHit, errHit);
        MeasurementError measTrkErr = topol.measurementError(lPTrk, errTrk);

        if (measHitErr.uu() < 0. || measHitErr.vv() < 0. || measTrkErr.uu() < 0. || measTrkErr.vv() < 0.) {
          edm::LogError("TrackerValidationVariables")
              << "@SUB=TrackerValidationVariables::fillHitQuantities"
              << "One of the squared error methods gives negative result"
              << "\n\tmeasHitErr.uu()\tmeasHitErr.vv()\tmeasTrkErr.uu()\tmeasTrkErr.vv()"
              << "\n\t" << measHitErr.uu() << "\t" << measHitErr.vv() << "\t" << measTrkErr.uu() << "\t"
              << measTrkErr.vv();
          continue;
        }

        float localStripLengthHit = topol.localStripLength(lPHit);
        float localStripLengthTrk = topol.localStripLength(lPTrk);
        float phiHit = topol.stripAngle(measHitPos.x());
        float phiTrk = topol.stripAngle(measTrkPos.x());
        float r_0 = topol.originToIntersection();

        resXTopol = (phiTrk - phiHit) * r_0;
        //      resXTopol = (tan(phiTrk)-tan(phiHit))*r_0;

        LocalPoint LocalHitPosCor = topol.localPosition(MeasurementPoint(measHitPos.x(), measTrkPos.y()));
        resXatTrkYTopol = lPTrk.x() - LocalHitPosCor.x();

        //resYTopol = measTrkPos.y()*localStripLengthTrk - measHitPos.y()*localStripLengthHit;
        float cosPhiHit(cos(phiHit)), cosPhiTrk(cos(phiTrk)), sinPhiHit(sin(phiHit)), sinPhiTrk(sin(phiTrk));
        float l_0 = r_0 - topol.detHeight() / 2;
        resYTopol = measTrkPos.y() * localStripLengthTrk - measHitPos.y() * localStripLengthHit +
                    l_0 * (1 / cosPhiTrk - 1 / cosPhiHit);

        resXprimeErr = std::sqrt(measHitErr.uu() + measTrkErr.uu()) * topol.angularWidth() * r_0;
        //resYprimeErr = std::sqrt(measHitErr.vv()*localStripLengthHit*localStripLengthHit + measTrkErr.vv()*localStripLengthTrk*localStripLengthTrk);
        float helpSummand = l_0 * l_0 * topol.angularWidth() * topol.angularWidth() *
                            (sinPhiHit * sinPhiHit / pow(cosPhiHit, 4) * measHitErr.uu() +
                             sinPhiTrk * sinPhiTrk / pow(cosPhiTrk, 4) * measTrkErr.uu());
        resYprimeErr = std::sqrt(measHitErr.vv() * localStripLengthHit * localStripLengthHit +
                                 measTrkErr.vv() * localStripLengthTrk * localStripLengthTrk + helpSummand);

        const TrapezoidalPlaneBounds* trapezoidalBound = dynamic_cast<const TrapezoidalPlaneBounds*>(&bound);
        if (trapezoidalBound != nullptr) {
          hitStruct.inside = trapezoidalBound->inside(lPTrk);
          length = trapezoidalBound->length();
          width = trapezoidalBound->width();
          //float widthAtHalfLength = trapezoidalBound->widthAtHalfLength();

          //	int yAxisOrientation=trapezoidalBound->yAxisOrientation();
          // for trapezoidal shape modules, scale with as function of local y coordinate
          //	float widthAtlocalY=width-(1-yAxisOrientation*2*lPTrk.y()/length)*(width-widthAtHalfLength);
          //	hitStruct.localXnorm = 2*hitStruct.localX/widthAtlocalY;
          hitStruct.localXnorm = 2 * hitStruct.localX / width;
          hitStruct.localYnorm = 2 * hitStruct.localY / length;
        } else {
          throw cms::Exception("Geometry Error") << "[TrackerValidationVariables] Cannot cast bounds to "
                                                    "TrapezoidalPlaneBounds as expected for TID and TEC";
        }

      } else {
        edm::LogWarning("TrackerValidationVariables") << "@SUB=TrackerValidationVariables::fillHitQuantities"
                                                      << "No valid tracker subdetector " << IntSubDetID;
        continue;
      }

      resXprime = resXTopol * uOrientation;
      resXatTrkY = resXatTrkYTopol;
      resYprime = resYTopol * vOrientation;

    } else {  // not a detUnit, so must be a virtual 2D-Module
      // FIXME: at present only for det units residuals are calculated and filled in the hitStruct
      // But in principle this method should also be useable for the gluedDets (2D modules in TIB, TID, TOB, TEC)
      // In this case, only orientation should be taken into account for primeResiduals, but not the radial topology
      // At present, default values (999.F) are given out
    }

    hitStruct.resXprime = resXprime;
    hitStruct.resXatTrkY = resXatTrkY;
    hitStruct.resYprime = resYprime;
    hitStruct.resXprimeErr = resXprimeErr;
    hitStruct.resYprimeErr = resYprimeErr;

    hitStruct.rawDetId = IntRawDetID;
    hitStruct.phi = tsos.globalDirection().phi();
    hitStruct.eta = tsos.globalDirection().eta();

    v_avhitout.push_back(hitStruct);
  }
}

void TrackerValidationVariables::fillTrackQuantities(const edm::Event& event,
                                                     const edm::EventSetup& eventSetup,
                                                     std::vector<AVTrackStruct>& v_avtrackout) {
  fillTrackQuantities(
      event, eventSetup, [](const reco::Track&) -> bool { return true; }, v_avtrackout);
}

void TrackerValidationVariables::fillTrackQuantities(const edm::Event& event,
                                                     const edm::EventSetup& eventSetup,
                                                     std::function<bool(const reco::Track&)> trackFilter,
                                                     std::vector<AVTrackStruct>& v_avtrackout) {
  const MagneticField& magneticField = eventSetup.getData(magneticFieldToken_);

  edm::Handle<reco::TrackCollection> tracksH;
  event.getByToken(tracksToken_, tracksH);
  if (!tracksH.isValid())
    return;
  auto const& tracks = *tracksH;
  auto ntrk = tracks.size();
  LogDebug("TrackerValidationVariables") << "Track collection size " << ntrk;

  edm::Handle<std::vector<Trajectory>> trajsH;
  event.getByToken(trajCollectionToken_, trajsH);
  bool yesTraj = trajsH.isValid();
  std::vector<Trajectory> const* trajs = nullptr;
  if (yesTraj)
    trajs = &(*trajsH);
  if (yesTraj)
    assert(trajs->size() == tracks.size());

  Trajectory const* trajectory = nullptr;
  for (unsigned int i = 0; i < ntrk; ++i) {
    auto const& track = tracks[i];
    if (yesTraj)
      trajectory = &(*trajs)[i];

    if (!trackFilter(track))
      continue;

    AVTrackStruct trackStruct;

    trackStruct.p = track.p();
    trackStruct.pt = track.pt();
    trackStruct.ptError = track.ptError();
    trackStruct.px = track.px();
    trackStruct.py = track.py();
    trackStruct.pz = track.pz();
    trackStruct.eta = track.eta();
    trackStruct.phi = track.phi();
    trackStruct.chi2 = track.chi2();
    trackStruct.chi2Prob = TMath::Prob(track.chi2(), track.ndof());
    trackStruct.normchi2 = track.normalizedChi2();
    GlobalPoint gPoint(track.vx(), track.vy(), track.vz());
    double theLocalMagFieldInInverseGeV = magneticField.inInverseGeV(gPoint).z();
    trackStruct.kappa = -track.charge() * theLocalMagFieldInInverseGeV / track.pt();
    trackStruct.charge = track.charge();
    trackStruct.d0 = track.d0();
    trackStruct.dz = track.dz();
    trackStruct.numberOfValidHits = track.numberOfValidHits();
    trackStruct.numberOfLostHits = track.numberOfLostHits();
    if (trajectory)
      fillHitQuantities(trajectory, trackStruct.hits);
    else
      fillHitQuantities(track, trackStruct.hits);

    v_avtrackout.push_back(trackStruct);
  }
}
