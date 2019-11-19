//
// -*- C++ -*-
// Package:    PFTracking
// Class:      PFTrackTransformer
//
// Original Author:  Michele Pioppi
// Other Author: Daniele Benedetti

#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// Add by Daniele
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "RecoParticleFlow/PFTracking/interface/PFGsfHelper.h"

using namespace std;
using namespace reco;
using namespace edm;

PFTrackTransformer::PFTrackTransformer(const math::XYZVector& B) : B_(B) {
  LogInfo("PFTrackTransformer") << "PFTrackTransformer built";

  onlyprop_ = false;
}

PFTrackTransformer::~PFTrackTransformer() {}

bool PFTrackTransformer::addPoints(reco::PFRecTrack& pftrack,
                                   const reco::Track& track,
                                   const Trajectory& traj,
                                   bool msgwarning) const {
  LogDebug("PFTrackTransformer") << "Trajectory propagation started";
  using namespace reco;
  using namespace std;

  float PT = track.pt();
  float pfmass = (pftrack.algoType() == reco::PFRecTrack::KF_ELCAND) ? 0.0005 : 0.139;
  float pfenergy = sqrt((pfmass * pfmass) + (track.p() * track.p()));
  // closest approach
  BaseParticlePropagator theParticle = BaseParticlePropagator(
      RawParticle(XYZTLorentzVector(track.px(), track.py(), track.pz(), pfenergy),
                  XYZTLorentzVector(track.vertex().x(), track.vertex().y(), track.vertex().z(), 0.),
                  track.charge()),
      0.,
      0.,
      B_.z());

  float pfoutenergy = sqrt((pfmass * pfmass) + track.outerMomentum().Mag2());
  BaseParticlePropagator theOutParticle = BaseParticlePropagator(
      RawParticle(
          XYZTLorentzVector(
              track.outerMomentum().x(), track.outerMomentum().y(), track.outerMomentum().z(), pfoutenergy),
          XYZTLorentzVector(track.outerPosition().x(), track.outerPosition().y(), track.outerPosition().z(), 0.),
          track.charge()),
      0.,
      0.,
      B_.z());

  math::XYZTLorentzVector momClosest = math::XYZTLorentzVector(track.px(), track.py(), track.pz(), track.p());
  const math::XYZPoint& posClosest = track.vertex();

  pftrack.addPoint(PFTrajectoryPoint(-1, PFTrajectoryPoint::ClosestApproach, posClosest, momClosest));

  //BEAMPIPE
  theParticle.setPropagationConditions(
      pfGeometry_.outerRadius(PFGeometry::BeamPipe), pfGeometry_.outerZ(PFGeometry::BeamPipe), false);
  theParticle.propagate();
  if (theParticle.getSuccess() != 0)
    pftrack.addPoint(PFTrajectoryPoint(-1,
                                       PFTrajectoryPoint::BeamPipeOrEndVertex,
                                       math::XYZPoint(theParticle.particle().vertex()),
                                       math::XYZTLorentzVector(theParticle.particle().momentum())));
  else {
    PFTrajectoryPoint dummyMaxSh;
    pftrack.addPoint(dummyMaxSh);
  }

  //trajectory points

  if (!onlyprop_) {
    bool direction = (traj.direction() == alongMomentum);
    vector<TrajectoryMeasurement> measurements = traj.measurements();
    int iTrajFirst = (direction) ? 0 : measurements.size() - 1;
    int increment = (direction) ? +1 : -1;
    int iTrajLast = (direction) ? int(measurements.size()) : -1;

    for (int iTraj = iTrajFirst; iTraj != iTrajLast; iTraj += increment) {
      GlobalPoint v = measurements[iTraj].updatedState().globalPosition();
      GlobalVector p = measurements[iTraj].updatedState().globalMomentum();
      unsigned int iid = measurements[iTraj].recHit()->det()->geographicalId().rawId();
      pftrack.addPoint(PFTrajectoryPoint(
          iid, -1, math::XYZPoint(v.x(), v.y(), v.z()), math::XYZTLorentzVector(p.x(), p.y(), p.z(), p.mag())));
    }
  }

  // ES1
  bool isBelowPS = false;
  theOutParticle.propagateToPreshowerLayer1(false);
  if (theOutParticle.getSuccess() != 0)
    pftrack.addPoint(PFTrajectoryPoint(-1,
                                       PFTrajectoryPoint::PS1,
                                       math::XYZPoint(theOutParticle.particle().vertex()),
                                       math::XYZTLorentzVector(theOutParticle.particle().momentum())));
  else {
    PFTrajectoryPoint dummyPS1;
    pftrack.addPoint(dummyPS1);
  }

  // ES1
  theOutParticle.propagateToPreshowerLayer2(false);
  if (theOutParticle.getSuccess() != 0) {
    pftrack.addPoint(PFTrajectoryPoint(-1,
                                       PFTrajectoryPoint::PS2,
                                       math::XYZPoint(theOutParticle.particle().vertex()),
                                       math::XYZTLorentzVector(theOutParticle.particle().momentum())));
    isBelowPS = true;
  } else {
    PFTrajectoryPoint dummyPS2;
    pftrack.addPoint(dummyPS2);
  }

  //ECAL entrance
  theOutParticle.propagateToEcalEntrance(false);
  if (theOutParticle.getSuccess() != 0) {
    pftrack.addPoint(PFTrajectoryPoint(-1,
                                       PFTrajectoryPoint::ECALEntrance,
                                       math::XYZPoint(theOutParticle.particle().vertex()),
                                       math::XYZTLorentzVector(theOutParticle.particle().momentum())));
    double ecalShowerDepth = PFCluster::getDepthCorrection(theOutParticle.particle().momentum().E(), isBelowPS, false);

    math::XYZPoint meanShower =
        math::XYZPoint(theOutParticle.particle().vertex()) +
        math::XYZTLorentzVector(theOutParticle.particle().momentum()).Vect().Unit() * ecalShowerDepth;

    pftrack.addPoint(PFTrajectoryPoint(-1,
                                       PFTrajectoryPoint::ECALShowerMax,
                                       meanShower,
                                       math::XYZTLorentzVector(theOutParticle.particle().momentum())));
  } else {
    if (PT > 5. && theOutParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_ && msgwarning)
      // Check consistent boundary as in CommonTools/BaseParticlePropagator/src/BaseParticlePropagator.cc
      // out of the HB/HE acceptance
      // eta = 3.0 -> cos^2(theta) = 0.99014 - cos**2(theta) is faster to determine than eta
      LogWarning("PFTrackTransformer") << "KF TRACK " << pftrack << " PROPAGATION TO THE ECAL HAS FAILED\n"
                                       << "theOutParticle.particle() pt,eta: " << theOutParticle.particle().pt() << " "
                                       << theOutParticle.particle().eta();
    PFTrajectoryPoint dummyECAL;
    pftrack.addPoint(dummyECAL);
    PFTrajectoryPoint dummyMaxSh;
    pftrack.addPoint(dummyMaxSh);
  }

  //HCAL entrance
  theOutParticle.propagateToHcalEntrance(false);
  if (theOutParticle.getSuccess() != 0)
    pftrack.addPoint(PFTrajectoryPoint(-1,
                                       PFTrajectoryPoint::HCALEntrance,
                                       math::XYZPoint(theOutParticle.particle().vertex()),
                                       math::XYZTLorentzVector(theOutParticle.particle().momentum())));
  else {
    if (PT > 5. && theOutParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_ && msgwarning)
      LogWarning("PFTrackTransformer") << "KF TRACK " << pftrack << " PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
    PFTrajectoryPoint dummyHCALentrance;
    pftrack.addPoint(dummyHCALentrance);
  }

  //HCAL exit
  // theOutParticle.setMagneticField(0); //Show we propagate as straight line inside HCAL ?
  theOutParticle.propagateToHcalExit(false);
  if (theOutParticle.getSuccess() != 0)
    pftrack.addPoint(PFTrajectoryPoint(-1,
                                       PFTrajectoryPoint::HCALExit,
                                       math::XYZPoint(theOutParticle.particle().vertex()),
                                       math::XYZTLorentzVector(theOutParticle.particle().momentum())));
  else {
    if (PT > 5. && theOutParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_ && msgwarning)
      LogWarning("PFTrackTransformer") << "KF TRACK " << pftrack << " PROPAGATION TO THE HCAL EXIT HAS FAILED";
    PFTrajectoryPoint dummyHCALexit;
    pftrack.addPoint(dummyHCALexit);
  }

  //HO layer0
  // if (abs(theOutParticle.particle().vertex().z())<550) {
  if (PT > 3.0) {  //Same value is used in PFBlockAlgo::link( case PFBlockLink::TRACKandHO:
    theOutParticle.setMagneticField(0);
    theOutParticle.propagateToHOLayer(false);
    if (theOutParticle.getSuccess() != 0) {
      pftrack.addPoint(PFTrajectoryPoint(-1,
                                         PFTrajectoryPoint::HOLayer,
                                         math::XYZPoint(theOutParticle.particle().vertex()),
                                         math::XYZTLorentzVector(theOutParticle.particle().momentum())));
    } else {
      if (PT > 5. && abs(theOutParticle.particle().Z()) < 700.25 && msgwarning)
        LogWarning("PFTrackTransformer") << "KF TRACK " << pftrack << " PROPAGATION TO THE HO HAS FAILED";
      PFTrajectoryPoint dummyHOLayer;
      pftrack.addPoint(dummyHOLayer);
    }
  } else {
    PFTrajectoryPoint dummyHOLayer;
    pftrack.addPoint(dummyHOLayer);
  }

  //VFcal(HF) entrance
  theOutParticle.setMagneticField(0);  // assume B=0 works ok for extrapolation to HF
  theOutParticle.propagateToVFcalEntrance(false);
  if (theOutParticle.getSuccess() != 0) {
    pftrack.addPoint(PFTrajectoryPoint(-1,
                                       PFTrajectoryPoint::VFcalEntrance,
                                       math::XYZPoint(theOutParticle.particle().vertex()),
                                       math::XYZTLorentzVector(theOutParticle.particle().momentum())));
  } else {
    PFTrajectoryPoint dummyVFcalentrance;
    pftrack.addPoint(dummyVFcalentrance);
  }

  return true;
}
bool PFTrackTransformer::addPointsAndBrems(reco::GsfPFRecTrack& pftrack,
                                           const reco::Track& track,
                                           const Trajectory& traj,
                                           const bool& GetMode) const {
  float PT = track.pt();
  // Trajectory for each trajectory point

  bool direction = (traj.direction() == alongMomentum);
  vector<TrajectoryMeasurement> measurements = traj.measurements();
  int iTrajFirst = (direction) ? 0 : measurements.size() - 1;
  int increment = (direction) ? +1 : -1;
  int iTrajLast = (direction) ? int(measurements.size()) : -1;

  unsigned int iTrajPos = 0;
  for (int iTraj = iTrajFirst; iTraj != iTrajLast; iTraj += increment) {
    GlobalPoint v = measurements[iTraj].updatedState().globalPosition();
    PFGsfHelper* PFGsf = new PFGsfHelper(measurements[iTraj]);
    //if (PFGsf->isValid()){
    bool ComputeMODE = GetMode;
    GlobalVector p = PFGsf->computeP(ComputeMODE);
    double DP = PFGsf->fittedDP();
    double SigmaDP = PFGsf->sigmafittedDP();
    unsigned int iid = measurements[iTraj].recHit()->det()->geographicalId().rawId();
    delete PFGsf;

    // --------------------------   Fill GSF Track -------------------------------------

    //    float pfmass= (pftrack.algoType()==reco::PFRecTrack::KF_ELCAND) ? 0.0005 : 0.139;
    float ptot = sqrt((p.x() * p.x()) + (p.y() * p.y()) + (p.z() * p.z()));
    float pfenergy = ptot;

    if (iTraj == iTrajFirst) {
      math::XYZTLorentzVector momClosest = math::XYZTLorentzVector(p.x(), p.y(), p.z(), ptot);
      const math::XYZPoint& posClosest = track.vertex();
      pftrack.addPoint(PFTrajectoryPoint(-1, PFTrajectoryPoint::ClosestApproach, posClosest, momClosest));

      BaseParticlePropagator theInnerParticle = BaseParticlePropagator(
          RawParticle(XYZTLorentzVector(p.x(), p.y(), p.z(), pfenergy),
                      XYZTLorentzVector(track.vertex().x(), track.vertex().y(), track.vertex().z(), 0.),
                      track.charge()),  //DANIELE Same thing v.x(),v.y(),v.()?
          0.,
          0.,
          B_.z());

      //BEAMPIPE
      theInnerParticle.setPropagationConditions(
          pfGeometry_.outerRadius(PFGeometry::BeamPipe), pfGeometry_.outerZ(PFGeometry::BeamPipe), false);
      theInnerParticle.propagate();
      if (theInnerParticle.getSuccess() != 0)
        pftrack.addPoint(PFTrajectoryPoint(-1,
                                           PFTrajectoryPoint::BeamPipeOrEndVertex,
                                           math::XYZPoint(theInnerParticle.particle().vertex()),
                                           math::XYZTLorentzVector(theInnerParticle.particle().momentum())));
      else {
        PFTrajectoryPoint dummyMaxSh;
        pftrack.addPoint(dummyMaxSh);
      }

      // First Point for the trajectory == Vertex ??
      pftrack.addPoint(PFTrajectoryPoint(
          iid, -1, math::XYZPoint(v.x(), v.y(), v.z()), math::XYZTLorentzVector(p.x(), p.y(), p.z(), p.mag())));
    }
    if (iTraj != iTrajFirst && iTraj != (abs(iTrajLast) - 1)) {
      pftrack.addPoint(PFTrajectoryPoint(
          iid, -1, math::XYZPoint(v.x(), v.y(), v.z()), math::XYZTLorentzVector(p.x(), p.y(), p.z(), p.mag())));
    }
    if (iTraj == (abs(iTrajLast) - 1)) {
      // Last Trajectory Meas
      pftrack.addPoint(PFTrajectoryPoint(
          iid, -1, math::XYZPoint(v.x(), v.y(), v.z()), math::XYZTLorentzVector(p.x(), p.y(), p.z(), p.mag())));

      BaseParticlePropagator theOutParticle =
          BaseParticlePropagator(RawParticle(XYZTLorentzVector(p.x(), p.y(), p.z(), pfenergy),
                                             XYZTLorentzVector(v.x(), v.y(), v.z(), 0.),
                                             track.charge()),
                                 0.,
                                 0.,
                                 B_.z());

      // ES1
      bool isBelowPS = false;
      theOutParticle.propagateToPreshowerLayer1(false);
      if (theOutParticle.getSuccess() != 0)
        pftrack.addPoint(PFTrajectoryPoint(-1,
                                           PFTrajectoryPoint::PS1,
                                           math::XYZPoint(theOutParticle.particle().vertex()),
                                           math::XYZTLorentzVector(theOutParticle.particle().momentum())));
      else {
        PFTrajectoryPoint dummyPS1;
        pftrack.addPoint(dummyPS1);
      }

      // ES2
      theOutParticle.propagateToPreshowerLayer2(false);
      if (theOutParticle.getSuccess() != 0) {
        pftrack.addPoint(PFTrajectoryPoint(-1,
                                           PFTrajectoryPoint::PS2,
                                           math::XYZPoint(theOutParticle.particle().vertex()),
                                           math::XYZTLorentzVector(theOutParticle.particle().momentum())));
        isBelowPS = true;
      } else {
        PFTrajectoryPoint dummyPS2;
        pftrack.addPoint(dummyPS2);
      }

      theOutParticle.propagateToEcalEntrance(false);

      if (theOutParticle.getSuccess() != 0) {
        pftrack.addPoint(PFTrajectoryPoint(-1,
                                           PFTrajectoryPoint::ECALEntrance,
                                           math::XYZPoint(theOutParticle.particle().vertex()),
                                           math::XYZTLorentzVector(theOutParticle.particle().momentum())));
        double ecalShowerDepth =
            PFCluster::getDepthCorrection(theOutParticle.particle().momentum().E(), isBelowPS, false);

        math::XYZPoint meanShower =
            math::XYZPoint(theOutParticle.particle().vertex()) +
            math::XYZTLorentzVector(theOutParticle.particle().momentum()).Vect().Unit() * ecalShowerDepth;

        pftrack.addPoint(PFTrajectoryPoint(-1,
                                           PFTrajectoryPoint::ECALShowerMax,
                                           meanShower,
                                           math::XYZTLorentzVector(theOutParticle.particle().momentum())));
      } else {
        if (PT > 5. && theOutParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
          LogWarning("PFTrackTransformer") << "GSF TRACK " << pftrack << " PROPAGATION TO THE ECAL HAS FAILED\n"
                                           << "theOutParticle.particle() pt,eta: " << theOutParticle.particle().pt()
                                           << " " << theOutParticle.particle().eta();
        PFTrajectoryPoint dummyECAL;
        pftrack.addPoint(dummyECAL);
        PFTrajectoryPoint dummyMaxSh;
        pftrack.addPoint(dummyMaxSh);
      }

      //HCAL entrance
      theOutParticle.propagateToHcalEntrance(false);
      if (theOutParticle.getSuccess() != 0)
        pftrack.addPoint(PFTrajectoryPoint(-1,
                                           PFTrajectoryPoint::HCALEntrance,
                                           math::XYZPoint(theOutParticle.particle().vertex()),
                                           math::XYZTLorentzVector(theOutParticle.particle().momentum())));
      else {
        if (PT > 5. && theOutParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
          LogWarning("PFTrackTransformer") << "GSF TRACK " << pftrack << " PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
        PFTrajectoryPoint dummyHCALentrance;
        pftrack.addPoint(dummyHCALentrance);
      }
      //HCAL exit
      theOutParticle.propagateToHcalExit(false);
      if (theOutParticle.getSuccess() != 0)
        pftrack.addPoint(PFTrajectoryPoint(-1,
                                           PFTrajectoryPoint::HCALExit,
                                           math::XYZPoint(theOutParticle.particle().vertex()),
                                           math::XYZTLorentzVector(theOutParticle.particle().momentum())));
      else {
        if (PT > 5. && theOutParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
          LogWarning("PFTrackTransformer") << "GSF TRACK " << pftrack << " PROPAGATION TO THE HCAL EXIT HAS FAILED";
        PFTrajectoryPoint dummyHCALexit;
        pftrack.addPoint(dummyHCALexit);
      }

      //HO Layer0
      if (abs(theOutParticle.particle().vertex().z()) < 550) {
        theOutParticle.setMagneticField(0);
        theOutParticle.propagateToHOLayer(false);
        if (theOutParticle.getSuccess() != 0)
          pftrack.addPoint(PFTrajectoryPoint(-1,
                                             PFTrajectoryPoint::HOLayer,
                                             math::XYZPoint(theOutParticle.particle().vertex()),
                                             math::XYZTLorentzVector(theOutParticle.particle().momentum())));
        else {
          if (PT > 5. && abs(theOutParticle.particle().Z()) < 700.25)
            LogWarning("PFTrackTransformer") << "GSF TRACK " << pftrack << " PROPAGATION TO THE HO HAS FAILED";
          PFTrajectoryPoint dummyHOLayer;
          pftrack.addPoint(dummyHOLayer);
        }
      }
    }

    // --------------------------   END GSF Track -------------------------------------

    // --------------------------   Fill Brem "Track" ---------------------------------
    // Fill the brem for each traj point

    //check that the vertex of the brem is in the tracker volume
    if ((v.perp() > 110) || (fabs(v.z()) > 280))
      continue;
    unsigned int iTrajPoint = iTrajPos + 2;
    if (iid % 2 == 1)
      iTrajPoint = 99;

    PFBrem brem(DP, SigmaDP, iTrajPoint);

    GlobalVector p_gamma = p * (fabs(DP) / p.mag());  // Direction from the electron (tangent), DP without any sign!;
    float e_gamma = fabs(DP);                         // DP = pout-pin so could be negative
    constexpr int gamma_charge = 0;
    BaseParticlePropagator theBremParticle =
        BaseParticlePropagator(RawParticle(XYZTLorentzVector(p_gamma.x(), p_gamma.y(), p_gamma.z(), e_gamma),
                                           XYZTLorentzVector(v.x(), v.y(), v.z(), 0.),
                                           gamma_charge),
                               0.,
                               0.,
                               B_.z());

    // add TrajectoryPoint for Brem, PS, ECAL, ECALShowMax, HCAL
    // Brem Entrance PS Layer1

    PFTrajectoryPoint dummyClosest;  // Added just to have the right number order in PFTrack.cc
    brem.addPoint(dummyClosest);

    PFTrajectoryPoint dummyBeamPipe;  // Added just to have the right number order in PFTrack.cc
    brem.addPoint(dummyBeamPipe);

    // ES1
    bool isBelowPS = false;
    theBremParticle.propagateToPreshowerLayer1(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::PS1,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      PFTrajectoryPoint dummyPS1;
      brem.addPoint(dummyPS1);
    }

    // ES2
    // Brem Entrance PS Layer 2
    theBremParticle.propagateToPreshowerLayer2(false);
    if (theBremParticle.getSuccess() != 0) {
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::PS2,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
      isBelowPS = true;
    } else {
      PFTrajectoryPoint dummyPS2;
      brem.addPoint(dummyPS2);
    }

    //ECAL entrance
    theBremParticle.propagateToEcalEntrance(false);
    if (theBremParticle.getSuccess() != 0) {
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::ECALEntrance,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
      double ecalShowerDepth =
          PFCluster::getDepthCorrection(theBremParticle.particle().momentum().E(), isBelowPS, false);

      math::XYZPoint meanShower =
          math::XYZPoint(theBremParticle.particle().vertex()) +
          math::XYZTLorentzVector(theBremParticle.particle().momentum()).Vect().Unit() * ecalShowerDepth;

      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::ECALShowerMax,
                                      meanShower,
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    } else {
      if ((DP > 5.) && ((DP / SigmaDP) > 3) && theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE ECAL HAS FAILED\n"
                                         << "theBremParticle.particle() pt,eta,cos2ThetaV: "
                                         << theBremParticle.particle().pt() << " " << theBremParticle.particle().eta()
                                         << " " << theBremParticle.particle().cos2ThetaV();
      PFTrajectoryPoint dummyECAL;
      brem.addPoint(dummyECAL);
      PFTrajectoryPoint dummyMaxSh;
      brem.addPoint(dummyMaxSh);
    }

    //HCAL entrance
    theBremParticle.propagateToHcalEntrance(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::HCALEntrance,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      if ((DP > 5.) && ((DP / SigmaDP) > 3) && theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
      PFTrajectoryPoint dummyHCALentrance;
      brem.addPoint(dummyHCALentrance);
    }

    //HCAL exit
    theBremParticle.propagateToHcalExit(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::HCALExit,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      if ((DP > 5.) && ((DP / SigmaDP) > 3) && theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE HCAL EXIT HAS FAILED";
      PFTrajectoryPoint dummyHCALexit;
      brem.addPoint(dummyHCALexit);
    }

    //HO Layer0
    if (abs(theBremParticle.particle().vertex().z()) < 550.0) {
      theBremParticle.setMagneticField(0);
      theBremParticle.propagateToHOLayer(false);
      if (theBremParticle.getSuccess() != 0)
        brem.addPoint(PFTrajectoryPoint(-1,
                                        PFTrajectoryPoint::HOLayer,
                                        math::XYZPoint(theBremParticle.particle().vertex()),
                                        math::XYZTLorentzVector(theBremParticle.particle().momentum())));
      else {
        if ((DP > 5.) && ((DP / SigmaDP) > 3) && abs(theBremParticle.particle().Z()) < 700.25)
          LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE H0 HAS FAILED";
        PFTrajectoryPoint dummyHOLayer;
        brem.addPoint(dummyHOLayer);
      }
    }
    brem.calculatePositionREP();
    pftrack.addBrem(brem);
    iTrajPos++;
  }
  return true;
}

bool PFTrackTransformer::addPointsAndBrems(reco::GsfPFRecTrack& pftrack,
                                           const reco::GsfTrack& track,
                                           const MultiTrajectoryStateTransform& mtjstate) const {
  //  float PT= track.pt();
  unsigned int iTrajPos = 0;
  unsigned int iid = 0;  // not anymore saved

  // *****************************   INNER State *************************************
  TrajectoryStateOnSurface inTSOS = mtjstate.innerStateOnSurface((track));
  TrajectoryStateOnSurface outTSOS = mtjstate.outerStateOnSurface((track));

  if (!inTSOS.isValid() || !outTSOS.isValid()) {
    if (!inTSOS.isValid())
      LogWarning("PFTrackTransformer") << " INNER TSOS NOT VALID ";
    if (!outTSOS.isValid())
      LogWarning("PFTrackTransformer") << " OUTER TSOS NOT VALID ";
    return false;
  }

  GlobalVector InMom;
  GlobalPoint InPos;
  if (inTSOS.isValid()) {
    multiTrajectoryStateMode::momentumFromModeCartesian(inTSOS, InMom);
    multiTrajectoryStateMode::positionFromModeCartesian(inTSOS, InPos);
  } else {
    InMom = GlobalVector(track.pxMode(), track.pyMode(), track.pzMode());
    InPos = GlobalPoint(0., 0., 0.);
  }

  //  float pfmass= (pftrack.algoType()==reco::PFRecTrack::KF_ELCAND) ? 0.0005 : 0.139;
  float ptot = sqrt((InMom.x() * InMom.x()) + (InMom.y() * InMom.y()) + (InMom.z() * InMom.z()));
  float pfenergy = ptot;

  math::XYZTLorentzVector momClosest = math::XYZTLorentzVector(InMom.x(), InMom.y(), InMom.z(), ptot);
  const math::XYZPoint& posClosest = track.vertex();
  pftrack.addPoint(PFTrajectoryPoint(-1, PFTrajectoryPoint::ClosestApproach, posClosest, momClosest));

  BaseParticlePropagator theInnerParticle = BaseParticlePropagator(
      RawParticle(XYZTLorentzVector(InMom.x(), InMom.y(), InMom.z(), pfenergy),
                  XYZTLorentzVector(track.vertex().x(), track.vertex().y(), track.vertex().z(), 0.),
                  track.charge()),  //DANIELE Same thing v.x(),v.y(),v.()?
      0.,
      0.,
      B_.z());
  //BEAMPIPE
  theInnerParticle.setPropagationConditions(
      pfGeometry_.outerRadius(PFGeometry::BeamPipe), pfGeometry_.outerZ(PFGeometry::BeamPipe), false);
  theInnerParticle.propagate();
  if (theInnerParticle.getSuccess() != 0)
    pftrack.addPoint(PFTrajectoryPoint(-1,
                                       PFTrajectoryPoint::BeamPipeOrEndVertex,
                                       math::XYZPoint(theInnerParticle.particle().vertex()),
                                       math::XYZTLorentzVector(theInnerParticle.particle().momentum())));
  else {
    PFTrajectoryPoint dummyBeam;
    pftrack.addPoint(dummyBeam);
  }

  // first tjpoint
  pftrack.addPoint(PFTrajectoryPoint(iid,
                                     -1,
                                     math::XYZPoint(InPos.x(), InPos.y(), InPos.z()),
                                     math::XYZTLorentzVector(InMom.x(), InMom.y(), InMom.z(), InMom.mag())));

  //######### Photon at INNER State ##########

  unsigned int iTrajPoint = iTrajPos + 2;
  double dp_tang = ptot;
  double sdp_tang = track.ptModeError() * (track.pMode() / track.ptMode());
  PFBrem brem(dp_tang, sdp_tang, iTrajPoint);
  constexpr int gamma_charge = 0;
  BaseParticlePropagator theBremParticle =
      BaseParticlePropagator(RawParticle(XYZTLorentzVector(InMom.x(), InMom.y(), InMom.z(), dp_tang),
                                         XYZTLorentzVector(InPos.x(), InPos.y(), InPos.z(), 0.),
                                         gamma_charge),
                             0.,
                             0.,
                             B_.z());
  // add TrajectoryPoint for Brem, PS, ECAL, ECALShowMax, HCAL
  // Brem Entrance PS Layer1
  PFTrajectoryPoint dummyClosest;  // Added just to have the right number order in PFTrack.cc
  brem.addPoint(dummyClosest);

  PFTrajectoryPoint dummyBeamPipe;  // Added just to have the right number order in PFTrack.cc
  brem.addPoint(dummyBeamPipe);

  bool isBelowPS = false;
  theBremParticle.propagateToPreshowerLayer1(false);
  if (theBremParticle.getSuccess() != 0)
    brem.addPoint(PFTrajectoryPoint(-1,
                                    PFTrajectoryPoint::PS1,
                                    math::XYZPoint(theBremParticle.particle().vertex()),
                                    math::XYZTLorentzVector(theBremParticle.particle().momentum())));
  else {
    PFTrajectoryPoint dummyPS1;
    brem.addPoint(dummyPS1);
  }

  // Brem Entrance PS Layer 2

  theBremParticle.propagateToPreshowerLayer2(false);
  if (theBremParticle.getSuccess() != 0) {
    brem.addPoint(PFTrajectoryPoint(-1,
                                    PFTrajectoryPoint::PS2,
                                    math::XYZPoint(theBremParticle.particle().vertex()),
                                    math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    isBelowPS = true;
  } else {
    PFTrajectoryPoint dummyPS2;
    brem.addPoint(dummyPS2);
  }

  //ECAL entrance
  theBremParticle.propagateToEcalEntrance(false);
  if (theBremParticle.getSuccess() != 0) {
    brem.addPoint(PFTrajectoryPoint(-1,
                                    PFTrajectoryPoint::ECALEntrance,
                                    math::XYZPoint(theBremParticle.particle().vertex()),
                                    math::XYZTLorentzVector(theBremParticle.particle().momentum())));

    //  for the first brem give a low default DP of 100 MeV.
    double EDepthCorr = 0.01;
    double ecalShowerDepth = PFCluster::getDepthCorrection(EDepthCorr, isBelowPS, false);

    math::XYZPoint meanShower =
        math::XYZPoint(theBremParticle.particle().vertex()) +
        math::XYZTLorentzVector(theBremParticle.particle().momentum()).Vect().Unit() * ecalShowerDepth;

    brem.addPoint(PFTrajectoryPoint(-1,
                                    PFTrajectoryPoint::ECALShowerMax,
                                    meanShower,
                                    math::XYZTLorentzVector(theBremParticle.particle().momentum())));
  } else {
    if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) &&
        theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
      LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE ECAL HAS FAILED\n"
                                       << "theBremParticle.particle() pt,eta,cos2ThetaV: "
                                       << theBremParticle.particle().pt() << " " << theBremParticle.particle().eta()
                                       << " " << theBremParticle.particle().cos2ThetaV();
    PFTrajectoryPoint dummyECAL;
    brem.addPoint(dummyECAL);
    PFTrajectoryPoint dummyMaxSh;
    brem.addPoint(dummyMaxSh);
  }

  //HCAL entrance
  theBremParticle.propagateToHcalEntrance(false);
  if (theBremParticle.getSuccess() != 0)
    brem.addPoint(PFTrajectoryPoint(-1,
                                    PFTrajectoryPoint::HCALEntrance,
                                    math::XYZPoint(theBremParticle.particle().vertex()),
                                    math::XYZTLorentzVector(theBremParticle.particle().momentum())));
  else {
    if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) &&
        theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
      LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
    PFTrajectoryPoint dummyHCALentrance;
    brem.addPoint(dummyHCALentrance);
  }

  //HCAL exit
  theBremParticle.propagateToHcalExit(false);
  if (theBremParticle.getSuccess() != 0)
    brem.addPoint(PFTrajectoryPoint(-1,
                                    PFTrajectoryPoint::HCALExit,
                                    math::XYZPoint(theBremParticle.particle().vertex()),
                                    math::XYZTLorentzVector(theBremParticle.particle().momentum())));
  else {
    if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) &&
        theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
      LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE HCAL EXIT HAS FAILED";
    PFTrajectoryPoint dummyHCALexit;
    brem.addPoint(dummyHCALexit);
  }

  //HO Layer0
  if (abs(theBremParticle.particle().vertex().z()) < 550) {
    theBremParticle.setMagneticField(0);
    theBremParticle.propagateToHOLayer(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::HOLayer,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) && abs(theBremParticle.particle().Z()) < 700.25)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE H0 HAS FAILED";
      PFTrajectoryPoint dummyHOLayer;
      brem.addPoint(dummyHOLayer);
    }
  }

  brem.calculatePositionREP();
  pftrack.addBrem(brem);
  iTrajPos++;

  // *****************************   INTERMIDIATE State *************************************
  //From the new Wolfgang code

  // To think if the cout should be removed.
  if (track.gsfExtra()->tangentsSize() == 0)
    LogError("PFTrackTransformer")
        << "BE CAREFUL: Gsf Tangents not stored in the event. You need to re-reco the particle-flow with "
           "RecoToDisplay_cfg.py and not RecoToDisplay_NoTracking_cfg.py ";

  vector<GsfTangent> gsftang = track.gsfExtra()->tangents();
  for (unsigned int iTang = 0; iTang < track.gsfExtra()->tangentsSize(); iTang++) {
    dp_tang = gsftang[iTang].deltaP().value();
    sdp_tang = gsftang[iTang].deltaP().error();

    //check that the vertex of the brem is in the tracker volume
    if ((sqrt(gsftang[iTang].position().x() * gsftang[iTang].position().x() +
              gsftang[iTang].position().y() * gsftang[iTang].position().y()) > 110) ||
        (fabs(gsftang[iTang].position().z()) > 280))
      continue;

    iTrajPoint = iTrajPos + 2;
    PFBrem brem(dp_tang, sdp_tang, iTrajPoint);

    GlobalVector p_tang =
        GlobalVector(gsftang[iTang].momentum().x(), gsftang[iTang].momentum().y(), gsftang[iTang].momentum().z());

    // ###### track tj points
    pftrack.addPoint(PFTrajectoryPoint(
        iid,
        -1,
        math::XYZPoint(gsftang[iTang].position().x(), gsftang[iTang].position().y(), gsftang[iTang].position().z()),
        math::XYZTLorentzVector(p_tang.x(), p_tang.y(), p_tang.z(), p_tang.mag())));

    //rescale
    GlobalVector p_gamma = p_tang * (fabs(dp_tang) / p_tang.mag());

    // GlobalVector

    double e_gamma = fabs(dp_tang);  // DP = pout-pin so could be negative
    theBremParticle = BaseParticlePropagator(
        RawParticle(
            XYZTLorentzVector(p_gamma.x(), p_gamma.y(), p_gamma.z(), e_gamma),
            XYZTLorentzVector(
                gsftang[iTang].position().x(), gsftang[iTang].position().y(), gsftang[iTang].position().z(), 0.),
            gamma_charge),
        0.,
        0.,
        B_.z());

    PFTrajectoryPoint dummyClosest;  // Added just to have the right number order in PFTrack.cc
    brem.addPoint(dummyClosest);

    PFTrajectoryPoint dummyBeamPipe;  // Added just to have the right number order in PFTrack.cc
    brem.addPoint(dummyBeamPipe);

    isBelowPS = false;
    theBremParticle.propagateToPreshowerLayer1(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::PS1,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      PFTrajectoryPoint dummyPS1;
      brem.addPoint(dummyPS1);
    }

    // Brem Entrance PS Layer 2

    theBremParticle.propagateToPreshowerLayer2(false);
    if (theBremParticle.getSuccess() != 0) {
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::PS2,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
      isBelowPS = true;
    } else {
      PFTrajectoryPoint dummyPS2;
      brem.addPoint(dummyPS2);
    }

    //ECAL entrance
    theBremParticle.propagateToEcalEntrance(false);
    if (theBremParticle.getSuccess() != 0) {
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::ECALEntrance,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));

      double ecalShowerDepth =
          PFCluster::getDepthCorrection(theBremParticle.particle().momentum().E(), isBelowPS, false);

      math::XYZPoint meanShower =
          math::XYZPoint(theBremParticle.particle().vertex()) +
          math::XYZTLorentzVector(theBremParticle.particle().momentum()).Vect().Unit() * ecalShowerDepth;

      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::ECALShowerMax,
                                      meanShower,
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    } else {
      if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) &&
          theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE ECAL HAS FAILED\n"
                                         << "theBremParticle.particle() pt,eta,cos2ThetaV: "
                                         << theBremParticle.particle().pt() << " " << theBremParticle.particle().eta()
                                         << " " << theBremParticle.particle().cos2ThetaV();
      PFTrajectoryPoint dummyECAL;
      brem.addPoint(dummyECAL);
      PFTrajectoryPoint dummyMaxSh;
      brem.addPoint(dummyMaxSh);
    }

    //HCAL entrance
    theBremParticle.propagateToHcalEntrance(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::HCALEntrance,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) &&
          theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
      PFTrajectoryPoint dummyHCALentrance;
      brem.addPoint(dummyHCALentrance);
    }

    //HCAL exit
    theBremParticle.propagateToHcalExit(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::HCALExit,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) &&
          theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE HCAL EXIT HAS FAILED";
      PFTrajectoryPoint dummyHCALexit;
      brem.addPoint(dummyHCALexit);
    }

    //HO Layer0
    if (abs(theBremParticle.particle().vertex().z()) < 550) {
      theBremParticle.setMagneticField(0);
      theBremParticle.propagateToHOLayer(false);
      if (theBremParticle.getSuccess() != 0)
        brem.addPoint(PFTrajectoryPoint(-1,
                                        PFTrajectoryPoint::HOLayer,
                                        math::XYZPoint(theBremParticle.particle().vertex()),
                                        math::XYZTLorentzVector(theBremParticle.particle().momentum())));
      else {
        if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) && abs(theBremParticle.particle().Z()) < 700.25)
          LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE H0 HAS FAILED";
        PFTrajectoryPoint dummyHOLayer;
        brem.addPoint(dummyHOLayer);
      }
    }

    brem.calculatePositionREP();
    pftrack.addBrem(brem);
    iTrajPos++;
  }

  // *****************************   OUTER State *************************************

  if (outTSOS.isValid()) {
    GlobalVector OutMom;
    GlobalPoint OutPos;

    // DANIELE ?????  if the out is not valid maybe take the last tangent?
    // From Wolfgang. It should be always valid

    multiTrajectoryStateMode::momentumFromModeCartesian(outTSOS, OutMom);
    multiTrajectoryStateMode::positionFromModeCartesian(outTSOS, OutPos);

    // last tjpoint
    pftrack.addPoint(PFTrajectoryPoint(iid,
                                       -1,
                                       math::XYZPoint(OutPos.x(), OutPos.y(), OutPos.z()),
                                       math::XYZTLorentzVector(OutMom.x(), OutMom.y(), OutMom.z(), OutMom.mag())));

    float ptot_out = sqrt((OutMom.x() * OutMom.x()) + (OutMom.y() * OutMom.y()) + (OutMom.z() * OutMom.z()));
    float pTtot_out = sqrt((OutMom.x() * OutMom.x()) + (OutMom.y() * OutMom.y()));
    float pfenergy_out = ptot_out;
    BaseParticlePropagator theOutParticle =
        BaseParticlePropagator(RawParticle(XYZTLorentzVector(OutMom.x(), OutMom.y(), OutMom.z(), pfenergy_out),
                                           XYZTLorentzVector(OutPos.x(), OutPos.y(), OutPos.z(), 0.),
                                           track.charge()),
                               0.,
                               0.,
                               B_.z());
    isBelowPS = false;
    theOutParticle.propagateToPreshowerLayer1(false);
    if (theOutParticle.getSuccess() != 0)
      pftrack.addPoint(PFTrajectoryPoint(-1,
                                         PFTrajectoryPoint::PS1,
                                         math::XYZPoint(theOutParticle.particle().vertex()),
                                         math::XYZTLorentzVector(theOutParticle.particle().momentum())));
    else {
      PFTrajectoryPoint dummyPS1;
      pftrack.addPoint(dummyPS1);
    }

    theOutParticle.propagateToPreshowerLayer2(false);
    if (theOutParticle.getSuccess() != 0) {
      pftrack.addPoint(PFTrajectoryPoint(-1,
                                         PFTrajectoryPoint::PS2,
                                         math::XYZPoint(theOutParticle.particle().vertex()),
                                         math::XYZTLorentzVector(theOutParticle.particle().momentum())));
      isBelowPS = true;
    } else {
      PFTrajectoryPoint dummyPS2;
      pftrack.addPoint(dummyPS2);
    }

    //ECAL entrance
    theOutParticle.propagateToEcalEntrance(false);
    if (theOutParticle.getSuccess() != 0) {
      pftrack.addPoint(PFTrajectoryPoint(-1,
                                         PFTrajectoryPoint::ECALEntrance,
                                         math::XYZPoint(theOutParticle.particle().vertex()),
                                         math::XYZTLorentzVector(theOutParticle.particle().momentum())));
      double EDepthCorr = 0.01;
      double ecalShowerDepth = PFCluster::getDepthCorrection(EDepthCorr, isBelowPS, false);

      math::XYZPoint meanShower =
          math::XYZPoint(theOutParticle.particle().vertex()) +
          math::XYZTLorentzVector(theOutParticle.particle().momentum()).Vect().Unit() * ecalShowerDepth;

      pftrack.addPoint(PFTrajectoryPoint(-1,
                                         PFTrajectoryPoint::ECALShowerMax,
                                         meanShower,
                                         math::XYZTLorentzVector(theOutParticle.particle().momentum())));
    } else {
      if (pTtot_out > 5. && theOutParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "GSF TRACK " << pftrack << " PROPAGATION TO THE ECAL HAS FAILED\n"
                                         << "theOutParticle.particle() pt,eta: " << theOutParticle.particle().pt()
                                         << " " << theOutParticle.particle().eta();

      PFTrajectoryPoint dummyECAL;
      pftrack.addPoint(dummyECAL);
      PFTrajectoryPoint dummyMaxSh;
      pftrack.addPoint(dummyMaxSh);
    }

    //HCAL entrance
    theOutParticle.propagateToHcalEntrance(false);
    if (theOutParticle.getSuccess() != 0)
      pftrack.addPoint(PFTrajectoryPoint(-1,
                                         PFTrajectoryPoint::HCALEntrance,
                                         math::XYZPoint(theOutParticle.particle().vertex()),
                                         math::XYZTLorentzVector(theOutParticle.particle().momentum())));
    else {
      if (pTtot_out > 5. && theOutParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "GSF TRACK " << pftrack << " PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
      PFTrajectoryPoint dummyHCALentrance;
      pftrack.addPoint(dummyHCALentrance);
    }
    //HCAL exit
    theOutParticle.propagateToHcalExit(false);
    if (theOutParticle.getSuccess() != 0)
      pftrack.addPoint(PFTrajectoryPoint(-1,
                                         PFTrajectoryPoint::HCALExit,
                                         math::XYZPoint(theOutParticle.particle().vertex()),
                                         math::XYZTLorentzVector(theOutParticle.particle().momentum())));
    else {
      if (pTtot_out > 5. && theOutParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "GSF TRACK " << pftrack << " PROPAGATION TO THE HCAL EXIT HAS FAILED";
      PFTrajectoryPoint dummyHCALexit;
      pftrack.addPoint(dummyHCALexit);
    }

    //HO Layer0
    if (abs(theOutParticle.particle().vertex().z()) < 550) {
      theOutParticle.setMagneticField(0);
      theOutParticle.propagateToHOLayer(false);
      if (theOutParticle.getSuccess() != 0)
        pftrack.addPoint(PFTrajectoryPoint(-1,
                                           PFTrajectoryPoint::HOLayer,
                                           math::XYZPoint(theOutParticle.particle().vertex()),
                                           math::XYZTLorentzVector(theOutParticle.particle().momentum())));
      else {
        if (pTtot_out > 5. && abs(theOutParticle.particle().Z()) < 700.25)
          LogWarning("PFTrackTransformer") << "GSF TRACK " << pftrack << " PROPAGATION TO THE HO HAS FAILED";
        PFTrajectoryPoint dummyHOLayer;
        pftrack.addPoint(dummyHOLayer);
      }
    }
    //######## Photon at the OUTER State ##########

    dp_tang = OutMom.mag();
    // for the moment same inner error just for semplicity
    sdp_tang = track.ptModeError() * (track.pMode() / track.ptMode());
    iTrajPoint = iTrajPos + 2;
    PFBrem brem(dp_tang, sdp_tang, iTrajPoint);

    theBremParticle = BaseParticlePropagator(RawParticle(XYZTLorentzVector(OutMom.x(), OutMom.y(), OutMom.z(), dp_tang),
                                                         XYZTLorentzVector(OutPos.x(), OutPos.y(), OutPos.z(), 0.),
                                                         gamma_charge),
                                             0.,
                                             0.,
                                             B_.z());

    PFTrajectoryPoint dummyClosest;  // Added just to have the right number order in PFTrack.cc
    brem.addPoint(dummyClosest);

    PFTrajectoryPoint dummyBeamPipe;  // Added just to have the right number order in PFTrack.cc
    brem.addPoint(dummyBeamPipe);

    isBelowPS = false;
    theBremParticle.propagateToPreshowerLayer1(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::PS1,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      PFTrajectoryPoint dummyPS1;
      brem.addPoint(dummyPS1);
    }

    // Brem Entrance PS Layer 2

    theBremParticle.propagateToPreshowerLayer2(false);
    if (theBremParticle.getSuccess() != 0) {
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::PS2,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
      isBelowPS = true;
    } else {
      PFTrajectoryPoint dummyPS2;
      brem.addPoint(dummyPS2);
    }

    //ECAL entrance
    theBremParticle.propagateToEcalEntrance(false);
    if (theBremParticle.getSuccess() != 0) {
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::ECALEntrance,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
      double ecalShowerDepth =
          PFCluster::getDepthCorrection(theBremParticle.particle().momentum().E(), isBelowPS, false);

      math::XYZPoint meanShower =
          math::XYZPoint(theBremParticle.particle().vertex()) +
          math::XYZTLorentzVector(theBremParticle.particle().momentum()).Vect().Unit() * ecalShowerDepth;

      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::ECALShowerMax,
                                      meanShower,
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    } else {
      if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) &&
          theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE ECAL HAS FAILED\n"
                                         << "theBremParticle.particle() pt,eta,cos2ThetaV: "
                                         << theBremParticle.particle().pt() << " " << theBremParticle.particle().eta()
                                         << " " << theBremParticle.particle().cos2ThetaV();
      PFTrajectoryPoint dummyECAL;
      brem.addPoint(dummyECAL);
      PFTrajectoryPoint dummyMaxSh;
      brem.addPoint(dummyMaxSh);
    }

    //HCAL entrance
    theBremParticle.propagateToHcalEntrance(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::HCALEntrance,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) &&
          theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE HCAL ENTRANCE HAS FAILED";
      PFTrajectoryPoint dummyHCALentrance;
      brem.addPoint(dummyHCALentrance);
    }
    //HCAL exit
    theBremParticle.propagateToHcalExit(false);
    if (theBremParticle.getSuccess() != 0)
      brem.addPoint(PFTrajectoryPoint(-1,
                                      PFTrajectoryPoint::HCALExit,
                                      math::XYZPoint(theBremParticle.particle().vertex()),
                                      math::XYZTLorentzVector(theBremParticle.particle().momentum())));
    else {
      if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) &&
          theBremParticle.particle().cos2ThetaV() < cos2ThetaV_Endcap_HiEnd_)
        LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE HCAL EXIT HAS FAILED";
      PFTrajectoryPoint dummyHCALexit;
      brem.addPoint(dummyHCALexit);
    }

    //HO Layer0
    if (abs(theBremParticle.particle().vertex().z()) < 550) {
      theBremParticle.setMagneticField(0);
      theBremParticle.propagateToHOLayer(false);
      if (theBremParticle.getSuccess() != 0)
        brem.addPoint(PFTrajectoryPoint(-1,
                                        PFTrajectoryPoint::HOLayer,
                                        math::XYZPoint(theBremParticle.particle().vertex()),
                                        math::XYZTLorentzVector(theBremParticle.particle().momentum())));
      else {
        if ((dp_tang > 5.) && ((dp_tang / sdp_tang) > 3) && abs(theBremParticle.particle().Z()) < 700.25)
          LogWarning("PFTrackTransformer") << "BREM " << brem << " PROPAGATION TO THE H0 HAS FAILED";
        PFTrajectoryPoint dummyHOLayer;
        brem.addPoint(dummyHOLayer);
      }
    }
    brem.calculatePositionREP();
    pftrack.addBrem(brem);
    iTrajPos++;
  }

  return true;
}
