#include "L1Trigger/L1TMuon/src/Phase2/L1TDisplacedMuonStubRecovery.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "L1Trigger/L1TMuon/src/Phase2/GeometryHelpers.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "iostream"

using namespace L1TMuon;

L1TDisplacedMuonStubRecovery::L1TDisplacedMuonStubRecovery(const edm::ParameterSet& iConfig)
{
}

L1TDisplacedMuonStubRecovery::~L1TDisplacedMuonStubRecovery()
{
}

void L1TDisplacedMuonStubRecovery::recoverCSCLCT(const l1t::EMTFTrack& track,
                                                 const l1t::EMTFTrackCollection* emtfTracks,
                                                 const CSCCorrelatedLCTDigiCollection* lcts,
                                                 int station, CSCCorrelatedLCTDigiId& bestLCT) const
{
  const std::string functionName("L1TDisplacedMuonStubRecovery::recoverCSCLCT");

  const int trackTriggerSector(track.Sector());
  const int trackEndcap(track.Eta() < 0 ? -1 : +1 );
  const int trackBX(track.BX());

  for (int ring=1; ring<=4; ring++){

    // only station 1 has ring 3/4 chambers
    if ((station!=1 and ring==3) or
        (station!=1 and ring==4)) continue;

    for (int chamber=1; chamber<=36; chamber++){

      // do not consider invalid detids
      if ( (station==2 or station==3 or station==4) and
           (ring==1) and chamber>18) continue;

      // create the detid
      CSCDetId ch_id(trackEndcap, station, ring, chamber);

      // trigger sector must be the same
      if (trackTriggerSector != ch_id.triggerSector()) continue;

      // get the stubs in this detid
      const auto& range = lcts->get(ch_id);
      for (auto digiItr = range.first; digiItr != range.second; ++digiItr){

        // candidate stub
        const auto& stub(*digiItr);

        // BXs have to match
        const int deltaBX = std::abs(stub.getBX() - trackBX);
        if (deltaBX > 1) continue;

        // check that this stub is not already part of a CSC TF track
        if (stubInEMTFTracks(stub, *emtfTracks)) continue;

        // bestLCT = pickBestMatchingStub(allxs[ch_id.station()-1], allys[ch_id.station()-1],
        //                                         bestLCT, std::make_pair(ch_id, stub), 6);
      }
      if (bestLCT.second.isValid()) {
        // stub position
        const auto& gp = L1TMuon::GeometryHelpers::globalPositionOfCSCLCT(csc_g,
                                                                          bestLCT.second,
                                                                          bestLCT.first);

        // // extra selection - stub not too far from track
        // if (reco::deltaR(float(gp.eta()), normalizedPhi(float(gp.phi())),
        //                  muon_eta, muon_phi) < 1){

        // }
      }
    }
  }
}

void L1TDisplacedMuonStubRecovery::getBestMatchedME0(const l1t::Muon& l1mu,
                                                     const ME0SegmentCollection* me0Segments,
                                                     ME0Segment& seg) const
{
  const std::string functionName("L1TDisplacedMuonStubRecovery::getBestMatchedME0");
  float minDR_L1Mu_ME0 = 999;

  if(verbose_) cout << functionName << " -- Find best match:" << endl;

  for (auto thisSegment = me0Segments->begin(); thisSegment != me0Segments->end(); ++thisSegment){

    const GlobalPoint& SegPos = L1TMuon::GeometryHelpers::globalPositionOfME0LCT(me0_g, seg);

    if(verbose_>1) cout << functionName << " -- Candidate segment " << *thisSegment << endl;

    if (l1mu.eta() * SegPos.eta() < 0){
      if(verbose_>1) cout << functionName << " -- Incorrect endcap" << endl;
      continue;
    }

    // const auto& tsos_ME0_st2 = propagateFromME0ToCSC(*thisSegment, event_.CSCTF_pt[j], event_.CSCTF_charge[j], 2);
    // if (not tsos_ME0_st2.isValid()) {
    //   if(verbose_>1) cout << functionName << " -- Cannot propagate to ME2Incorrect endcap" << endl;
    //   continue;
    // }

    // const GlobalPoint& gp_ME0_st2(tsos_ME0_st2.globalPosition());

  }

  /*
    // BXs have to match
    int deltaBX = std::abs(getME0SegmentBX(*thisSegment) - (event_.CSCTF_bx[j]));
    if (deltaBX > 0) continue; // segment must be in time!

    if(verbose) {
      std::cout <<"ME0gp eta "<< SegPos.eta() <<" phi "<< SegPos.phi() <<" gp_ME0_st2 eta "<< gp_ME0_st2.eta()<<" phi "<< gp_ME0_st2.phi()<<" L1Mu eta "<< event_.CSCTF_eta[j] <<" phi "<< event_.CSCTF_phi[j]  <<" pt "<< event_.CSCTF_pt[j] <<" quality "<< event_.CSCTF_quality[j] << std::endl;
    }

    if (std::fabs(SegPos.eta() - event_.CSCTF_eta[j])>0.4){
      std::cout << "ALARM!!!! Too large deltaEta, Seg eta "<< SegPos.eta() <<" L1Mu_eta "<< event_.CSCTF_eta[j] << std::endl;
      continue;
    }

    // cannot be too far in phi
    if (std::fabs(reco::deltaPhi((float) gp_ME0_st2.phi(), (float) event_.CSCTF_phi[j])) > 0.7) {
      std::cout << "ALARM!!!! Too large deltaPhi, Seg phi "<< gp_ME0_st2.phi() <<" L1Mu_phi "<< event_.CSCTF_phi[j] << std::endl;
      continue;
    }

    if(verbose) {
      std::cout << "ME0DetId " << id << std::endl;
      std::cout<<"Candidate " << *thisSegment << std::endl;
      std::cout<<"eta phi " << SegPos.eta() << " " << normalizedPhi((float)SegPos.phi()) << std::endl;
      if (event_.CSCTF_phi1[j] != 99 and  std::fabs(reco::deltaPhi(float(event_.CSCTF_phi1[j]), float(SegPos.phi()))) > 0.3)
        std::cout << "ALARM!!!! Too large deltaPhi, Seg phi "<< SegPos.phi()<<" L1Mu_phi "<< event_.CSCTF_phi[j] <<" CSCTF_phi1 "<< event_.CSCTF_phi1[j] <<" propagate ME0 to st2 phi "<< gp_ME0_st2.phi() << std::endl;

    }

    float deltaR = std::fabs(reco::deltaPhi((float) gp_ME0_st2.phi(), (float) event_.CSCTF_phi[j]));
    if (deltaR < minDR_L1Mu_ME0){
      minDR_L1Mu_ME0 = deltaR;
      bestMatchingME0Segment = *thisSegment;
      std::cout <<"\t so far bestMatchingME0Segment, minDR_L1Mu_ME0 "<< minDR_L1Mu_ME0 << std::endl;
    }
  }
  */
}

bool L1TDisplacedMuonStubRecovery::stubInEMTFTracks(const CSCCorrelatedLCTDigi& stub,
                                                    const l1t::EMTFTrackCollection& l1Tracks) const
{
  // check all tracks
  for (const auto& tftrack: l1Tracks){

    // check all track hits
    for (const auto& hit : tftrack.Hits()){

      // check only CSC stubs
      if (not hit.Is_CSC()) continue;

      // does this stub match?
      if (stub == hit.CSC_LCTDigi()) return true;
    }
  }
  return false;
}
