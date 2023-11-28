#include "RecoMET/METAlgorithms/interface/GlobalHaloAlgo.h"
namespace {
  constexpr float c_cm_per_ns = 29.9792458;
};
/*
  [class]:  GlobalHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: See GlobalHaloAlgo.h
  [date]: October 15, 2009
*/
using namespace std;
using namespace edm;
using namespace reco;

enum detectorregion { EB, EE, HB, HE };
int Phi_To_HcaliPhi(float phi) {
  phi = phi < 0 ? phi + 2. * TMath::Pi() : phi;
  float phi_degrees = phi * (360.) / (2. * TMath::Pi());
  int iPhi = (int)((phi_degrees / 5.) + 1.);

  return iPhi < 73 ? iPhi : 73;
}

int Phi_To_EcaliPhi(float phi) {
  phi = phi < 0 ? phi + 2. * TMath::Pi() : phi;
  float phi_degrees = phi * (360.) / (2. * TMath::Pi());
  int iPhi = (int)(phi_degrees + 1.);

  return iPhi < 361 ? iPhi : 360;
}

GlobalHaloAlgo::GlobalHaloAlgo() {
  // Defaults are "loose"
  Ecal_R_Min = 110.;  // Tight: 200.
  Ecal_R_Max = 330.;  // Tight: 250.
  Hcal_R_Min = 110.;  // Tight: 220.
  Hcal_R_Max = 490.;  // Tight: 350.
}

reco::GlobalHaloData GlobalHaloAlgo::Calculate(const CaloGeometry& TheCaloGeometry,
                                               const CSCGeometry& TheCSCGeometry,
                                               const reco::CaloMET& TheCaloMET,
                                               edm::Handle<edm::View<Candidate> >& TheCaloTowers,
                                               edm::Handle<CSCSegmentCollection>& TheCSCSegments,
                                               edm::Handle<CSCRecHit2DCollection>& TheCSCRecHits,
                                               edm::Handle<reco::MuonCollection>& TheMuons,
                                               const CSCHaloData& TheCSCHaloData,
                                               const EcalHaloData& TheEcalHaloData,
                                               const HcalHaloData& TheHcalHaloData,
                                               bool ishlt) {
  GlobalHaloData TheGlobalHaloData;
  float METOverSumEt = TheCaloMET.sumEt() ? TheCaloMET.pt() / TheCaloMET.sumEt() : 0;
  TheGlobalHaloData.SetMETOverSumEt(METOverSumEt);

  int EcalOverlapping_CSCRecHits[361] = {};
  int EcalOverlapping_CSCSegments[361] = {};
  int HcalOverlapping_CSCRecHits[73] = {};
  int HcalOverlapping_CSCSegments[73] = {};

  if (TheCSCSegments.isValid()) {
    for (CSCSegmentCollection::const_iterator iSegment = TheCSCSegments->begin(); iSegment != TheCSCSegments->end();
         iSegment++) {
      bool EcalOverlap[361];
      bool HcalOverlap[73];
      for (int i = 0; i < 361; i++) {
        EcalOverlap[i] = false;
        if (i < 73)
          HcalOverlap[i] = false;
      }

      std::vector<CSCRecHit2D> Hits = iSegment->specificRecHits();
      for (std::vector<CSCRecHit2D>::iterator iHit = Hits.begin(); iHit != Hits.end(); iHit++) {
        DetId TheDetUnitId(iHit->geographicalId());
        if (TheDetUnitId.det() != DetId::Muon)
          continue;
        if (TheDetUnitId.subdetId() != MuonSubdetId::CSC)
          continue;

        const GeomDetUnit* TheUnit = TheCSCGeometry.idToDetUnit(TheDetUnitId);
        LocalPoint TheLocalPosition = iHit->localPosition();
        const BoundPlane& TheSurface = TheUnit->surface();
        const GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);

        int Hcal_iphi = Phi_To_HcaliPhi(TheGlobalPosition.phi());
        int Ecal_iphi = Phi_To_EcaliPhi(TheGlobalPosition.phi());
        float x = TheGlobalPosition.x();
        float y = TheGlobalPosition.y();

        float r = TMath::Sqrt(x * x + y * y);

        if (r < Ecal_R_Max && r > Ecal_R_Min)
          EcalOverlap[Ecal_iphi] = true;
        if (r < Hcal_R_Max && r > Hcal_R_Max)
          HcalOverlap[Hcal_iphi] = true;
      }
      for (int i = 0; i < 361; i++) {
        if (EcalOverlap[i])
          EcalOverlapping_CSCSegments[i]++;
        if (i < 73 && HcalOverlap[i])
          HcalOverlapping_CSCSegments[i]++;
      }
    }
  }
  if (TheCSCRecHits.isValid()) {
    for (CSCRecHit2DCollection::const_iterator iCSCRecHit = TheCSCRecHits->begin(); iCSCRecHit != TheCSCRecHits->end();
         iCSCRecHit++) {
      DetId TheDetUnitId(iCSCRecHit->geographicalId());
      if (TheDetUnitId.det() != DetId::Muon)
        continue;
      if (TheDetUnitId.subdetId() != MuonSubdetId::CSC)
        continue;

      const GeomDetUnit* TheUnit = TheCSCGeometry.idToDetUnit(TheDetUnitId);
      LocalPoint TheLocalPosition = iCSCRecHit->localPosition();
      const BoundPlane& TheSurface = TheUnit->surface();
      const GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);

      int Hcaliphi = Phi_To_HcaliPhi(TheGlobalPosition.phi());
      int Ecaliphi = Phi_To_EcaliPhi(TheGlobalPosition.phi());
      float x = TheGlobalPosition.x();
      float y = TheGlobalPosition.y();

      float r = TMath::Sqrt(x * x + y * y);

      if (r < Ecal_R_Max && r > Ecal_R_Min)
        EcalOverlapping_CSCRecHits[Ecaliphi]++;
      if (r < Hcal_R_Max && r > Hcal_R_Max)
        HcalOverlapping_CSCRecHits[Hcaliphi]++;
    }
  }

  // In development....
  // Get Ecal Wedges
  std::vector<PhiWedge> EcalWedges = TheEcalHaloData.GetPhiWedges();

  // Get Hcal Wedges
  std::vector<PhiWedge> HcalWedges = TheHcalHaloData.GetPhiWedges();

  //Get Ref to CSC Tracks
  //edm::RefVector<reco::TrackCollection> TheCSCTracks = TheCSCHaloData.GetTracks();
  //for(unsigned int i = 0 ; i < TheCSCTracks.size() ; i++ )
  //edm::Ref<reco::TrackCollection> iTrack( TheCSCTracks, i );

  // Get global positions of central most rechit of CSC Halo tracks
  std::vector<GlobalPoint> TheGlobalPositions = TheCSCHaloData.GetCSCTrackImpactPositions();

  // Container to store Ecal/Hcal iPhi values matched to impact point of CSC tracks
  std::vector<int> vEcaliPhi, vHcaliPhi;

  for (std::vector<GlobalPoint>::iterator Pos = TheGlobalPositions.begin(); Pos != TheGlobalPositions.end(); Pos++) {
    // Calculate global phi coordinate for central most rechit in the track
    float global_phi = Pos->phi();
    float global_r = TMath::Sqrt(Pos->x() * Pos->x() + Pos->y() * Pos->y());

    // Convert global phi to iPhi
    int global_EcaliPhi = Phi_To_EcaliPhi(global_phi);
    int global_HcaliPhi = Phi_To_HcaliPhi(global_phi);

    //Loop over Ecal Phi Wedges
    for (std::vector<PhiWedge>::iterator iWedge = EcalWedges.begin(); iWedge != EcalWedges.end(); iWedge++) {
      if ((TMath::Abs(global_EcaliPhi - iWedge->iPhi()) <= 5) && (global_r > Ecal_R_Min && global_r < Ecal_R_Max)) {
        bool StoreWedge = true;
        for (unsigned int i = 0; i < vEcaliPhi.size(); i++)
          if (vEcaliPhi[i] == iWedge->iPhi())
            StoreWedge = false;

        if (StoreWedge) {
          PhiWedge NewWedge(*iWedge);
          NewWedge.SetOverlappingCSCSegments(EcalOverlapping_CSCSegments[iWedge->iPhi()]);
          NewWedge.SetOverlappingCSCRecHits(EcalOverlapping_CSCRecHits[iWedge->iPhi()]);
          vEcaliPhi.push_back(iWedge->iPhi());
          TheGlobalHaloData.GetMatchedEcalPhiWedges().push_back(NewWedge);
        }
      }
    }
    //Loop over Hcal Phi Wedges
    for (std::vector<PhiWedge>::iterator iWedge = HcalWedges.begin(); iWedge != HcalWedges.end(); iWedge++) {
      if ((TMath::Abs(global_HcaliPhi - iWedge->iPhi()) <= 2) && (global_r > Hcal_R_Min && global_r < Hcal_R_Max)) {
        bool StoreWedge = true;
        for (unsigned int i = 0; i < vHcaliPhi.size(); i++)
          if (vHcaliPhi[i] == iWedge->iPhi())
            StoreWedge = false;

        if (StoreWedge) {
          vHcaliPhi.push_back(iWedge->iPhi());
          PhiWedge NewWedge(*iWedge);
          NewWedge.SetOverlappingCSCSegments(HcalOverlapping_CSCSegments[iWedge->iPhi()]);
          NewWedge.SetOverlappingCSCRecHits(HcalOverlapping_CSCRecHits[iWedge->iPhi()]);
          PhiWedge wedge(*iWedge);
          TheGlobalHaloData.GetMatchedHcalPhiWedges().push_back(NewWedge);
        }
      }
    }
  }

  // Corrections to MEx, MEy
  float dMEx = 0.;
  float dMEy = 0.;
  // Loop over calotowers and correct the MET for the towers that lie in the trajectory of the CSC Halo Tracks
  for (edm::View<Candidate>::const_iterator iCandidate = TheCaloTowers->begin(); iCandidate != TheCaloTowers->end();
       iCandidate++) {
    const Candidate* c = &(*iCandidate);
    if (c) {
      const CaloTower* iTower = dynamic_cast<const CaloTower*>(c);
      if (iTower->et() < TowerEtThreshold)
        continue;
      if (abs(iTower->ieta()) > 24)
        continue;  // not in barrel/endcap
      int iphi = iTower->iphi();
      for (unsigned int x = 0; x < vEcaliPhi.size(); x++) {
        if (iphi == vEcaliPhi[x]) {
          dMEx += (TMath::Cos(iTower->phi()) * iTower->emEt());
          dMEy += (TMath::Sin(iTower->phi()) * iTower->emEt());
        }
      }
      for (unsigned int x = 0; x < vHcaliPhi.size(); x++) {
        if (iphi == vHcaliPhi[x]) {
          dMEx += (TMath::Cos(iTower->phi()) * iTower->hadEt());
          dMEy += (TMath::Sin(iTower->phi()) * iTower->hadEt());
        }
      }
    }
  }

  TheGlobalHaloData.SetMETCorrections(dMEx, dMEy);

  std::vector<HaloClusterCandidateECAL> hccandEB = TheEcalHaloData.getHaloClusterCandidatesEB();
  std::vector<HaloClusterCandidateECAL> hccandEE = TheEcalHaloData.getHaloClusterCandidatesEE();
  std::vector<HaloClusterCandidateHCAL> hccandHB = TheHcalHaloData.getHaloClusterCandidatesHB();
  std::vector<HaloClusterCandidateHCAL> hccandHE = TheHcalHaloData.getHaloClusterCandidatesHE();

  //CSC-calo matching
  bool ECALBmatched(false), ECALEmatched(false), HCALBmatched(false), HCALEmatched(false);

  if (TheCSCSegments.isValid()) {
    for (CSCSegmentCollection::const_iterator iSegment = TheCSCSegments->begin(); iSegment != TheCSCSegments->end();
         iSegment++) {
      CSCDetId iCscDetID = iSegment->cscDetId();
      bool Segment1IsGood = true;

      //avoid segments from collision muons
      if (TheMuons.isValid()) {
        for (reco::MuonCollection::const_iterator mu = TheMuons->begin(); mu != TheMuons->end() && (Segment1IsGood);
             mu++) {
          if (!mu->isTrackerMuon() && !mu->isGlobalMuon() && mu->isStandAloneMuon())
            continue;
          if (!mu->isGlobalMuon() && mu->isTrackerMuon() && mu->pt() < 3)
            continue;
          const std::vector<MuonChamberMatch> chambers = mu->matches();
          for (std::vector<MuonChamberMatch>::const_iterator kChamber = chambers.begin(); kChamber != chambers.end();
               kChamber++) {
            if (kChamber->detector() != MuonSubdetId::CSC)
              continue;
            for (std::vector<reco::MuonSegmentMatch>::const_iterator kSegment = kChamber->segmentMatches.begin();
                 kSegment != kChamber->segmentMatches.end();
                 kSegment++) {
              edm::Ref<CSCSegmentCollection> cscSegRef = kSegment->cscSegmentRef;
              CSCDetId kCscDetID = cscSegRef->cscDetId();

              if (kCscDetID == iCscDetID) {
                Segment1IsGood = false;
              }
            }
          }
        }
      }
      if (!Segment1IsGood)
        continue;

      // Get local direction vector; if direction runs parallel to beamline,
      // count this segment as beam halo candidate.
      LocalPoint iLocalPosition = iSegment->localPosition();
      LocalVector iLocalDirection = iSegment->localDirection();

      GlobalPoint iGlobalPosition = TheCSCGeometry.chamber(iCscDetID)->toGlobal(iLocalPosition);
      GlobalVector iGlobalDirection = TheCSCGeometry.chamber(iCscDetID)->toGlobal(iLocalDirection);

      float iTheta = iGlobalDirection.theta();
      if (iTheta > max_segment_theta && iTheta < TMath::Pi() - max_segment_theta)
        continue;

      float iPhi = iGlobalPosition.phi();
      float iR = sqrt(iGlobalPosition.perp2());
      float iZ = iGlobalPosition.z();
      float iT = iSegment->time();

      //CSC-calo matching:
      //Here, one checks if any halo cluster can be matched to a CSC segment.
      //The matching uses both geometric (dphi, dR) and timing information (dt).
      //The cut values depend on the subdetector considered (e.g. in HB, Rcalo-Rsegment is allowed to be very negative)

      bool ebmatched = SegmentMatchingEB(TheGlobalHaloData, hccandEB, iZ, iR, iT, iPhi, ishlt);
      bool eematched = SegmentMatchingEE(TheGlobalHaloData, hccandEE, iZ, iR, iT, iPhi, ishlt);
      bool hbmatched = SegmentMatchingHB(TheGlobalHaloData, hccandHB, iZ, iR, iT, iPhi, ishlt);
      bool hematched = SegmentMatchingHE(TheGlobalHaloData, hccandHE, iZ, iR, iT, iPhi, ishlt);

      ECALBmatched |= ebmatched;
      ECALEmatched |= eematched;
      HCALBmatched |= hbmatched;
      HCALEmatched |= hematched;
    }
  }

  TheGlobalHaloData.SetSegmentIsEBCaloMatched(ECALBmatched);
  TheGlobalHaloData.SetSegmentIsEECaloMatched(ECALEmatched);
  TheGlobalHaloData.SetSegmentIsHBCaloMatched(HCALBmatched);
  TheGlobalHaloData.SetSegmentIsHECaloMatched(HCALEmatched);

  //Now checking patterns from EcalHaloData and HcalHaloData:
  //Simply check whether any cluster has a halo pattern
  //In that case store the rhits in GlobalHaloData

  bool HaloPatternFoundInEB = false;
  for (auto& hcand : hccandEB) {
    if ((hcand.getIsHaloFromPattern() && !ishlt) || (hcand.getIsHaloFromPattern_HLT() && ishlt)) {
      HaloPatternFoundInEB = true;
      edm::RefVector<EcalRecHitCollection> bhrhcandidates = hcand.getBeamHaloRecHitsCandidates();
      AddtoBeamHaloEBEERechits(bhrhcandidates, TheGlobalHaloData, true);
    }
  }

  bool HaloPatternFoundInEE = false;
  for (auto& hcand : hccandEE) {
    if ((hcand.getIsHaloFromPattern() && !ishlt) || (hcand.getIsHaloFromPattern_HLT() && ishlt)) {
      HaloPatternFoundInEE = true;
      edm::RefVector<EcalRecHitCollection> bhrhcandidates = hcand.getBeamHaloRecHitsCandidates();
      AddtoBeamHaloEBEERechits(bhrhcandidates, TheGlobalHaloData, false);
    }
  }

  bool HaloPatternFoundInHB = false;
  for (auto& hcand : hccandHB) {
    if ((hcand.getIsHaloFromPattern() && !ishlt) || (hcand.getIsHaloFromPattern_HLT() && ishlt)) {
      HaloPatternFoundInHB = true;
      edm::RefVector<HBHERecHitCollection> bhrhcandidates = hcand.getBeamHaloRecHitsCandidates();
      AddtoBeamHaloHBHERechits(bhrhcandidates, TheGlobalHaloData);
    }
  }

  bool HaloPatternFoundInHE = false;
  for (auto& hcand : hccandHE) {
    if ((hcand.getIsHaloFromPattern() && !ishlt) || (hcand.getIsHaloFromPattern_HLT() && ishlt)) {
      HaloPatternFoundInHE = true;
      edm::RefVector<HBHERecHitCollection> bhrhcandidates = hcand.getBeamHaloRecHitsCandidates();
      AddtoBeamHaloHBHERechits(bhrhcandidates, TheGlobalHaloData);
    }
  }
  TheGlobalHaloData.SetHaloPatternFoundEB(HaloPatternFoundInEB);
  TheGlobalHaloData.SetHaloPatternFoundEE(HaloPatternFoundInEE);
  TheGlobalHaloData.SetHaloPatternFoundHB(HaloPatternFoundInHB);
  TheGlobalHaloData.SetHaloPatternFoundHE(HaloPatternFoundInHE);

  return TheGlobalHaloData;
}

bool GlobalHaloAlgo::SegmentMatchingEB(reco::GlobalHaloData& thehalodata,
                                       const std::vector<HaloClusterCandidateECAL>& haloclustercands,
                                       float iZ,
                                       float iR,
                                       float iT,
                                       float iPhi,
                                       bool ishlt) {
  bool rhmatchingfound = false;

  for (auto& hcand : haloclustercands) {
    if (!ApplyMatchingCuts(EB,
                           ishlt,
                           hcand.getSeedEt(),
                           iZ,
                           hcand.getSeedZ(),
                           iR,
                           hcand.getSeedR(),
                           iT,
                           hcand.getSeedTime(),
                           iPhi,
                           hcand.getSeedPhi()))
      continue;

    rhmatchingfound = true;

    edm::RefVector<EcalRecHitCollection> bhrhcandidates = hcand.getBeamHaloRecHitsCandidates();

    AddtoBeamHaloEBEERechits(bhrhcandidates, thehalodata, true);
  }

  return rhmatchingfound;
}

bool GlobalHaloAlgo::SegmentMatchingEE(reco::GlobalHaloData& thehalodata,
                                       const std::vector<HaloClusterCandidateECAL>& haloclustercands,
                                       float iZ,
                                       float iR,
                                       float iT,
                                       float iPhi,
                                       bool ishlt) {
  bool rhmatchingfound = false;

  for (auto& hcand : haloclustercands) {
    if (!ApplyMatchingCuts(EE,
                           ishlt,
                           hcand.getSeedEt(),
                           iZ,
                           hcand.getSeedZ(),
                           iR,
                           hcand.getSeedR(),
                           iT,
                           hcand.getSeedTime(),
                           iPhi,
                           hcand.getSeedPhi()))
      continue;

    rhmatchingfound = true;

    edm::RefVector<EcalRecHitCollection> bhrhcandidates = hcand.getBeamHaloRecHitsCandidates();

    AddtoBeamHaloEBEERechits(bhrhcandidates, thehalodata, false);
  }

  return rhmatchingfound;
}

bool GlobalHaloAlgo::SegmentMatchingHB(reco::GlobalHaloData& thehalodata,
                                       const std::vector<HaloClusterCandidateHCAL>& haloclustercands,
                                       float iZ,
                                       float iR,
                                       float iT,
                                       float iPhi,
                                       bool ishlt) {
  bool rhmatchingfound = false;

  for (auto& hcand : haloclustercands) {
    if (!ApplyMatchingCuts(HB,
                           ishlt,
                           hcand.getSeedEt(),
                           iZ,
                           hcand.getSeedZ(),
                           iR,
                           hcand.getSeedR(),
                           iT,
                           hcand.getSeedTime(),
                           iPhi,
                           hcand.getSeedPhi()))
      continue;

    rhmatchingfound = true;

    edm::RefVector<HBHERecHitCollection> bhrhcandidates = hcand.getBeamHaloRecHitsCandidates();

    AddtoBeamHaloHBHERechits(bhrhcandidates, thehalodata);
  }

  return rhmatchingfound;
}

bool GlobalHaloAlgo::SegmentMatchingHE(reco::GlobalHaloData& thehalodata,
                                       const std::vector<HaloClusterCandidateHCAL>& haloclustercands,
                                       float iZ,
                                       float iR,
                                       float iT,
                                       float iPhi,
                                       bool ishlt) {
  bool rhmatchingfound = false;

  for (auto& hcand : haloclustercands) {
    if (!ApplyMatchingCuts(HE,
                           ishlt,
                           hcand.getSeedEt(),
                           iZ,
                           hcand.getSeedZ(),
                           iR,
                           hcand.getSeedR(),
                           iT,
                           hcand.getSeedTime(),
                           iPhi,
                           hcand.getSeedPhi()))
      continue;

    rhmatchingfound = true;

    edm::RefVector<HBHERecHitCollection> bhrhcandidates = hcand.getBeamHaloRecHitsCandidates();

    AddtoBeamHaloHBHERechits(bhrhcandidates, thehalodata);
  }

  return rhmatchingfound;
}

bool GlobalHaloAlgo::ApplyMatchingCuts(int subdet,
                                       bool ishlt,
                                       double rhet,
                                       double segZ,
                                       double rhZ,
                                       double segR,
                                       double rhR,
                                       double segT,
                                       double rhT,
                                       double segPhi,
                                       double rhPhi) {
  //Std::Absolute time wrt BX
  double tBXrh = rhT + sqrt(rhR * rhR + rhZ * rhZ) / c_cm_per_ns;
  double tBXseg = segT + sqrt(segR * segR + segZ * segZ) / c_cm_per_ns;
  //Time at z=0, under beam halo hypothesis
  double tcorseg = tBXseg - std::abs(segZ) / c_cm_per_ns;       //Outgoing beam halo
  double tcorsegincbh = tBXseg + std::abs(segZ) / c_cm_per_ns;  //Ingoing beam halo
  double truedt[4] = {1000, 1000, 1000, 1000};
  //There are four types of segments associated to beam halo, test each hypothesis:
  //IT beam halo, ingoing track
  double twindow_seg = 15;
  if (std::abs(tcorsegincbh) < twindow_seg)
    truedt[0] = tBXrh - tBXseg - std::abs(rhZ - segZ) / c_cm_per_ns;
  //IT beam halo, outgoing track
  if (std::abs(tcorseg) < twindow_seg)
    truedt[1] = tBXseg - tBXrh - std::abs(rhZ - segZ) / c_cm_per_ns;
  //OT beam halo (from next BX), ingoing track
  if (tcorsegincbh > 25 - twindow_seg && std::abs(tcorsegincbh) < 25 + twindow_seg)
    truedt[2] = tBXrh - tBXseg - std::abs(rhZ - segZ) / c_cm_per_ns;
  //OT beam halo (from next BX), outgoing track
  if (tcorseg > 25 - twindow_seg && tcorseg < 25 + twindow_seg)
    truedt[3] = tBXseg - tBXrh - std::abs(rhZ - segZ) / c_cm_per_ns;

  if (subdet == EB) {
    if (rhet < et_thresh_rh_eb)
      return false;
    if (rhet < 20 && ishlt)
      return false;
    if (std::abs(deltaPhi(rhPhi, segPhi)) > dphi_thresh_segvsrh_eb)
      return false;
    if (rhR - segR < dr_lowthresh_segvsrh_eb)
      return false;
    if (rhR - segR > dr_highthresh_segvsrh_eb)
      return false;
    if (std::abs(truedt[0]) > dt_segvsrh_eb && std::abs(truedt[1]) > dt_segvsrh_eb &&
        std::abs(truedt[2]) > dt_segvsrh_eb && std::abs(truedt[3]) > dt_segvsrh_eb)
      return false;
    return true;
  }

  if (subdet == EE) {
    if (rhet < et_thresh_rh_ee)
      return false;
    if (rhet < 20 && ishlt)
      return false;
    if (std::abs(deltaPhi(rhPhi, segPhi)) > dphi_thresh_segvsrh_ee)
      return false;
    if (rhR - segR < dr_lowthresh_segvsrh_ee)
      return false;
    if (rhR - segR > dr_highthresh_segvsrh_ee)
      return false;
    if (std::abs(truedt[0]) > dt_segvsrh_ee && std::abs(truedt[1]) > dt_segvsrh_ee &&
        std::abs(truedt[2]) > dt_segvsrh_ee && std::abs(truedt[3]) > dt_segvsrh_ee)
      return false;
    return true;
  }

  if (subdet == HB) {
    if (rhet < et_thresh_rh_hb)
      return false;
    if (rhet < 20 && ishlt)
      return false;
    if (std::abs(deltaPhi(rhPhi, segPhi)) > dphi_thresh_segvsrh_hb)
      return false;
    if (rhR - segR < dr_lowthresh_segvsrh_hb)
      return false;
    if (rhR - segR > dr_highthresh_segvsrh_hb)
      return false;
    if (std::abs(truedt[0]) > dt_segvsrh_hb && std::abs(truedt[1]) > dt_segvsrh_hb &&
        std::abs(truedt[2]) > dt_segvsrh_hb && std::abs(truedt[3]) > dt_segvsrh_hb)
      return false;
    return true;
  }

  if (subdet == HE) {
    if (rhet < et_thresh_rh_he)
      return false;
    if (rhet < 20 && ishlt)
      return false;
    if (std::abs(deltaPhi(rhPhi, segPhi)) > dphi_thresh_segvsrh_he)
      return false;
    if (rhR - segR < dr_lowthresh_segvsrh_he)
      return false;
    if (rhR - segR > dr_highthresh_segvsrh_he)
      return false;
    if (std::abs(truedt[0]) > dt_segvsrh_he && std::abs(truedt[1]) > dt_segvsrh_he &&
        std::abs(truedt[2]) > dt_segvsrh_he && std::abs(truedt[3]) > dt_segvsrh_he)
      return false;
    return true;
  }

  return false;
}

void GlobalHaloAlgo::AddtoBeamHaloEBEERechits(edm::RefVector<EcalRecHitCollection>& bhtaggedrechits,
                                              reco::GlobalHaloData& thehalodata,
                                              bool isbarrel) {
  for (size_t ihit = 0; ihit < bhtaggedrechits.size(); ++ihit) {
    bool alreadyincl = false;
    edm::Ref<EcalRecHitCollection> rhRef(bhtaggedrechits[ihit]);
    edm::RefVector<EcalRecHitCollection> refrhcoll;
    if (isbarrel)
      refrhcoll = thehalodata.GetEBRechits();
    else
      refrhcoll = thehalodata.GetEERechits();
    for (size_t jhit = 0; jhit < refrhcoll.size(); jhit++) {
      edm::Ref<EcalRecHitCollection> rhitRef(refrhcoll[jhit]);
      if (rhitRef->detid() == rhRef->detid())
        alreadyincl = true;
      if (rhitRef->detid() == rhRef->detid())
        break;
    }
    if (!alreadyincl && isbarrel)
      thehalodata.GetEBRechits().push_back(rhRef);
    if (!alreadyincl && !isbarrel)
      thehalodata.GetEERechits().push_back(rhRef);
  }
}

void GlobalHaloAlgo::AddtoBeamHaloHBHERechits(edm::RefVector<HBHERecHitCollection>& bhtaggedrechits,
                                              reco::GlobalHaloData& thehalodata) {
  for (size_t ihit = 0; ihit < bhtaggedrechits.size(); ++ihit) {
    bool alreadyincl = false;
    edm::Ref<HBHERecHitCollection> rhRef(bhtaggedrechits[ihit]);
    edm::RefVector<HBHERecHitCollection> refrhcoll;
    refrhcoll = thehalodata.GetHBHERechits();
    for (size_t jhit = 0; jhit < refrhcoll.size(); jhit++) {
      edm::Ref<HBHERecHitCollection> rhitRef(refrhcoll[jhit]);
      if (rhitRef->detid() == rhRef->detid())
        alreadyincl = true;
      if (rhitRef->detid() == rhRef->detid())
        break;
    }
    if (!alreadyincl)
      thehalodata.GetHBHERechits().push_back(rhRef);
  }
}
