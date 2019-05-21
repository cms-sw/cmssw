#include "Fireworks/ParticleFlow/interface/FWPFTrackUtils.h"

FWPFTrackSingleton *FWPFTrackSingleton::pInstance = nullptr;
bool FWPFTrackSingleton::instanceFlag = false;

//______________________________________________________________________________
FWPFTrackSingleton *FWPFTrackSingleton::Instance() {
  if (!instanceFlag)  // Instance doesn't exist yet
  {
    pInstance = new FWPFTrackSingleton();
    instanceFlag = true;
  }

  return pInstance;  // Pointer to sole instance
}

//______________________________________________________________________________
void FWPFTrackSingleton::initPropagator() {
  m_magField = new FWMagField();

  // Common propagator, helix stepper
  m_trackPropagator = new TEveTrackPropagator();
  m_trackPropagator->SetMagFieldObj(m_magField, false);
  m_trackPropagator->SetMaxR(FWPFGeom::caloR3());
  m_trackPropagator->SetMaxZ(FWPFGeom::caloZ2());
  m_trackPropagator->SetDelta(0.01);
  m_trackPropagator->SetProjTrackBreaking(TEveTrackPropagator::kPTB_UseLastPointPos);
  m_trackPropagator->SetRnrPTBMarkers(kTRUE);
  m_trackPropagator->IncDenyDestroy();

  // Tracker propagator
  m_trackerTrackPropagator = new TEveTrackPropagator();
  m_trackerTrackPropagator->SetStepper(TEveTrackPropagator::kRungeKutta);
  m_trackerTrackPropagator->SetMagFieldObj(m_magField, false);
  m_trackerTrackPropagator->SetDelta(0.01);
  m_trackerTrackPropagator->SetMaxR(FWPFGeom::caloR3());
  m_trackerTrackPropagator->SetMaxZ(FWPFGeom::caloZ2());
  m_trackerTrackPropagator->SetProjTrackBreaking(TEveTrackPropagator::kPTB_UseLastPointPos);
  m_trackerTrackPropagator->SetRnrPTBMarkers(kTRUE);
  m_trackerTrackPropagator->IncDenyDestroy();
}

//______________________________________________________________________________
FWPFTrackUtils::FWPFTrackUtils() { m_singleton = FWPFTrackSingleton::Instance(); }

//______________________________________________________________________________
TEveTrack *FWPFTrackUtils::getTrack(const reco::Track &iData) {
  TEveTrackPropagator *propagator =
      (!iData.extra().isAvailable()) ? m_singleton->getTrackerTrackPropagator() : m_singleton->getTrackPropagator();

  TEveRecTrack t;
  t.fBeta = 1;
  t.fP = TEveVector(iData.px(), iData.py(), iData.pz());
  t.fV = TEveVector(iData.vertex().x(), iData.vertex().y(), iData.vertex().z());
  t.fSign = iData.charge();
  TEveTrack *trk = new TEveTrack(&t, propagator);
  trk->MakeTrack();

  return trk;
}

//______________________________________________________________________________
TEveStraightLineSet *FWPFTrackUtils::setupLegoTrack(const reco::Track &iData) {
  using namespace FWPFGeom;

  // Declarations
  int wraps[3] = {-1, -1, -1};
  bool ECAL = false;
  TEveTrack *trk = getTrack(iData);
  std::vector<TEveVector> trackPoints(trk->GetN() - 1);
  const Float_t *points = trk->GetP();
  TEveStraightLineSet *legoTrack = new TEveStraightLineSet();

  if (m_singleton->getField()->getSource() == FWMagField::kNone) {
    if (fabs(iData.eta()) < 2.0 && iData.pt() > 0.5 && iData.pt() < 30) {
      double estimate = fw::estimate_field(iData, true);
      if (estimate >= 0)
        m_singleton->getField()->guessField(estimate);
    }
  }

  // Convert to Eta/Phi and store in vector
  for (Int_t i = 1; i < trk->GetN(); ++i) {
    int j = i * 3;
    TEveVector temp = TEveVector(points[j], points[j + 1], points[j + 2]);
    TEveVector vec = TEveVector(temp.Eta(), temp.Phi(), 0.001);

    trackPoints[i - 1] = vec;
  }

  // Add first point to ps if necessary
  for (Int_t i = 1; i < trk->GetN(); ++i) {
    int j = i * 3;
    TEveVector v1 = TEveVector(points[j], points[j + 1], points[j + 2]);

    if (!ECAL) {
      if (FWPFMaths::checkIntersect(v1, caloR1())) {
        TEveVector v2 = TEveVector(points[j - 3], points[j - 2], points[j - 1]);
        TEveVector xyPoint = FWPFMaths::lineCircleIntersect(v1, v2, caloR1());
        TEveVector zPoint;
        if (v1.fZ < 0)
          zPoint = TEveVector(xyPoint.fX, xyPoint.fY, v1.fZ - 50.f);
        else
          zPoint = TEveVector(xyPoint.fX, xyPoint.fY, v1.fZ + 50.f);

        TEveVector vec = FWPFMaths::lineLineIntersect(v1, v2, xyPoint, zPoint);
        legoTrack->AddMarker(vec.Eta(), vec.Phi(), 0.001, 0);

        wraps[0] = i;  // There is now a chance that the track will also reach the HCAL radius
        ECAL = true;
      } else if (fabs(v1.fZ) >= caloZ1()) {
        TEveVector p1, p2;
        TEveVector vec, v2 = TEveVector(points[j - 3], points[j - 2], points[j - 1]);
        float z, y = FWPFMaths::linearInterpolation(v2, v1, caloZ1());

        if (v2.fZ > 0)
          z = caloZ1();
        else
          z = caloZ1() * -1;

        p1 = TEveVector(v2.fX + 50, y, z);
        p2 = TEveVector(v2.fX - 50, y, z);
        vec = FWPFMaths::lineLineIntersect(v1, v2, p1, p2);

        legoTrack->AddMarker(vec.Eta(), vec.Phi(), 0.001, 0);
        wraps[0] = i;
        ECAL = true;
      }
    } else if (FWPFMaths::checkIntersect(v1, caloR2())) {
      TEveVector v2 = TEveVector(points[j - 3], points[j - 2], points[j - 1]);
      TEveVector xyPoint = FWPFMaths::lineCircleIntersect(v1, v2, caloR2());
      TEveVector zPoint;
      if (v1.fZ < 0)
        zPoint = TEveVector(xyPoint.fX, xyPoint.fY, v1.fZ - 50.f);
      else
        zPoint = TEveVector(xyPoint.fX, xyPoint.fY, v1.fZ + 50.f);

      TEveVector vec = FWPFMaths::lineLineIntersect(v1, v2, xyPoint, zPoint);
      legoTrack->AddMarker(vec.Eta(), vec.Phi(), 0.001, 1);

      wraps[1] = i;  // There is now a chance that the track will also reach the HCAL radius
      break;
    }
  }

  if (wraps[0] != -1)  //if( ECAL )
  {
    int i = (trk->GetN() - 1) * 3;
    int j = trk->GetN() - 2;  // This is equal to the last index in trackPoints vector
    TEveVector vec = TEveVector(points[i], points[i + 1], points[i + 2]);

    if (FWPFMaths::checkIntersect(vec, caloR3() - 1)) {
      legoTrack->AddMarker(vec.Eta(), vec.Phi(), 0.001, 2);
      wraps[2] = j;
    } else if (fabs(vec.fZ) >= caloZ2()) {
      legoTrack->AddMarker(vec.Eta(), vec.Phi(), 0.001, 2);
      wraps[2] = j;
    }
  }

  /* Handle phi wrapping */
  for (int i = 0; i < static_cast<int>(trackPoints.size() - 1); ++i) {
    if ((trackPoints[i + 1].fY - trackPoints[i].fY) > 1) {
      trackPoints[i + 1].fY -= TMath::TwoPi();
      if (i == wraps[0]) {
        TEveChunkManager::iterator mi(legoTrack->GetMarkerPlex());
        mi.next();  // First point
        TEveStraightLineSet::Marker_t &m = *(TEveStraightLineSet::Marker_t *)mi();
        m.fV[0] = trackPoints[i + 1].fX;
        m.fV[1] = trackPoints[i + 1].fY;
        m.fV[2] = 0.001;
      } else if (i == wraps[1]) {
        TEveChunkManager::iterator mi(legoTrack->GetMarkerPlex());
        mi.next();
        mi.next();  // Second point
        TEveStraightLineSet::Marker_t &m = *(TEveStraightLineSet::Marker_t *)mi();
        m.fV[0] = trackPoints[i + 1].fX;
        m.fV[1] = trackPoints[i + 1].fY;
        m.fV[2] = 0.001;
      }
    }

    if ((trackPoints[i].fY - trackPoints[i + 1].fY) > 1) {
      trackPoints[i + 1].fY += TMath::TwoPi();
      if (i == wraps[0]) {
        TEveChunkManager::iterator mi(legoTrack->GetMarkerPlex());
        mi.next();  // First point
        TEveStraightLineSet::Marker_t &m = *(TEveStraightLineSet::Marker_t *)mi();
        m.fV[0] = trackPoints[i + 1].fX;
        m.fV[1] = trackPoints[i + 1].fY;
        m.fV[2] = 0.001;
      } else if (i == wraps[1]) {
        TEveChunkManager::iterator mi(legoTrack->GetMarkerPlex());
        mi.next();
        mi.next();  // Second point
        TEveStraightLineSet::Marker_t &m = *(TEveStraightLineSet::Marker_t *)mi();
        m.fV[0] = trackPoints[i + 1].fX;
        m.fV[1] = trackPoints[i + 1].fY;
        m.fV[2] = 0.001;
      }
    }
  }

  int end = static_cast<int>(trackPoints.size() - 1);
  if (wraps[2] == end) {
    TEveChunkManager::iterator mi(legoTrack->GetMarkerPlex());
    mi.next();
    mi.next();
    mi.next();  // Third point
    TEveStraightLineSet::Marker_t &m = *(TEveStraightLineSet::Marker_t *)mi();
    m.fV[0] = trackPoints[end].fX;
    m.fV[1] = trackPoints[end].fY;
    m.fV[2] = 0.001;
  }

  // Set points on TEveLineSet object ready for displaying
  for (unsigned int i = 0; i < trackPoints.size() - 1; ++i)
    legoTrack->AddLine(trackPoints[i], trackPoints[i + 1]);

  legoTrack->SetDepthTest(false);
  legoTrack->SetMarkerStyle(4);
  legoTrack->SetMarkerSize(1);
  legoTrack->SetRnrMarkers(true);

  delete trk;  // Release memory that is no longer required

  return legoTrack;
}

//______________________________________________________________________________
TEveTrack *FWPFTrackUtils::setupTrack(const reco::Track &iData) {
  if (m_singleton->getField()->getSource() == FWMagField::kNone) {
    if (fabs(iData.eta()) < 2.0 && iData.pt() > 0.5 && iData.pt() < 30) {
      double estimate = fw::estimate_field(iData, true);
      if (estimate >= 0)
        m_singleton->getField()->guessField(estimate);
    }
  }

  TEveTrack *trk = getTrack(iData);

  return trk;
}

//______________________________________________________________________________
TEvePointSet *FWPFTrackUtils::getCollisionMarkers(const TEveTrack *trk) {
  using namespace FWPFGeom;

  bool ECAL = false;
  const Float_t *points = trk->GetP();
  TEvePointSet *ps = new TEvePointSet();

  for (Int_t i = 1; i < trk->GetN(); ++i) {
    int j = i * 3;
    TEveVector v1 = TEveVector(points[j], points[j + 1], points[j + 2]);

    if (!ECAL) {
      if (FWPFMaths::checkIntersect(v1, caloR1())) {
        TEveVector v2 = TEveVector(points[j - 3], points[j - 2], points[j - 1]);
        TEveVector xyPoint = FWPFMaths::lineCircleIntersect(v1, v2, caloR1());
        TEveVector zPoint;
        if (v1.fZ < 0)
          zPoint = TEveVector(xyPoint.fX, xyPoint.fY, v1.fZ - 50.f);
        else
          zPoint = TEveVector(xyPoint.fX, xyPoint.fY, v1.fZ + 50.f);

        TEveVector vec = FWPFMaths::lineLineIntersect(v1, v2, xyPoint, zPoint);
        ps->SetNextPoint(vec.fX, vec.fY, vec.fZ);

        ECAL = true;
      } else if (fabs(v1.fZ) >= caloZ1()) {
        TEveVector p1, p2;
        TEveVector vec, v2 = TEveVector(points[j - 3], points[j - 2], points[j - 1]);
        float z, y = FWPFMaths::linearInterpolation(v2, v1, caloZ1());

        if (v2.fZ > 0)
          z = caloZ1();
        else
          z = caloZ1() * -1;

        p1 = TEveVector(v2.fX + 50, y, z);
        p2 = TEveVector(v2.fX - 50, y, z);
        vec = FWPFMaths::lineLineIntersect(v1, v2, p1, p2);

        ps->SetNextPoint(vec.fX, vec.fY, vec.fZ);
        ECAL = true;
      }
    } else if (FWPFMaths::checkIntersect(v1, caloR2())) {
      TEveVector v2 = TEveVector(points[j - 3], points[j - 2], points[j - 1]);
      TEveVector xyPoint = FWPFMaths::lineCircleIntersect(v1, v2, caloR2());
      TEveVector zPoint;
      if (v1.fZ < 0)
        zPoint = TEveVector(xyPoint.fX, xyPoint.fY, v1.fZ - 50.f);
      else
        zPoint = TEveVector(xyPoint.fX, xyPoint.fY, v1.fZ + 50.f);

      TEveVector vec = FWPFMaths::lineLineIntersect(v1, v2, xyPoint, zPoint);
      ps->SetNextPoint(vec.fX, vec.fY, vec.fZ);
      break;  // ECAL and HCAL collisions found so stop looping
    }
  }

  // HCAL collisions (outer radius and endcap)
  int i = (trk->GetN() - 1) * 3;
  TEveVector vec = TEveVector(points[i], points[i + 1], points[i + 2]);

  if (FWPFMaths::checkIntersect(vec, caloR3() - 1))
    ps->SetNextPoint(vec.fX, vec.fY, vec.fZ);
  else if (fabs(vec.fZ) >= caloZ2())
    ps->SetNextPoint(vec.fX, vec.fY, vec.fZ);

  return ps;
}
