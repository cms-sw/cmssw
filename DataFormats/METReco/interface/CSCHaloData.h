#ifndef DATAFORMATS_METRECO_CSCHALODATA_H
#define DATAFORMATS_METRECO_CSCHALODATA_H
#include "TMath.h"
#include <vector>

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/METReco/interface/HaloData.h"

#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

/*
  [class]:  CSCHaloData
  [authors]: R. Remington, The University of Florida
  [description]: Container class to store beam halo data specific to the CSC subdetector
  [date]: October 15, 2009
*/

namespace reco {
  
  class CSCHaloData{
    
  public:
    // Default constructor
    CSCHaloData();

    virtual ~CSCHaloData(){}

    // Number of HaloTriggers in +/- endcap
    int NumberOfHaloTriggers (HaloData::Endcap z= HaloData::both) const ;
    int NumberOfHaloTriggers_TrkMuUnVeto (HaloData::Endcap z= HaloData::both) const ;
    int NHaloTriggers(HaloData::Endcap z = HaloData::both ) const { return NumberOfHaloTriggers(z);}

    // Number of Halo Tracks in +/-  endcap
    int NumberOfHaloTracks(HaloData::Endcap z= HaloData::both) const ;
    int NHaloTracks(HaloData::Endcap z = HaloData::both) const { return NumberOfHaloTracks(z) ;}

    // Halo trigger bit from the HLT  
    bool CSCHaloHLTAccept() const {return HLTAccept;}

    // Number of chamber-level triggers with non-collision timing
    short int NumberOfOutOfTimeTriggers(HaloData::Endcap z = HaloData::both ) const;
    short int NOutOfTimeTriggers(HaloData::Endcap z = HaloData::both) const {return NumberOfOutOfTimeTriggers(z);}
    // Number of CSCRecHits with non-collision timing
    short int NumberOfOutTimeHits() const { return nOutOfTimeHits;}
    short int NOutOfTimeHits() const {return nOutOfTimeHits;}
    // Look at number of muons with timing consistent with incoming particles
    short int NTracksSmalldT() const { return nTracks_Small_dT;}
    short int NTracksSmallBeta() const{ return nTracks_Small_beta; }
    short int NTracksSmallBetaAndSmalldT() const { return nTracks_Small_dT_Small_beta; }

    // MLR
    short int NFlatHaloSegments() const{ return nFlatHaloSegments; }
    bool GetSegmentsInBothEndcaps() const{ return segments_in_both_endcaps; }
    bool GetSegmentIsCaloMatched() const{ return segmentiscalomatched; }
    // End MLR
    short int NFlatHaloSegments_TrkMuUnVeto() const{ return nFlatHaloSegments_TrkMuUnVeto; }
    bool GetSegmentsInBothEndcaps_Loose_TrkMuUnVeto() const{ return segments_in_both_endcaps_loose_TrkMuUnVeto;}
    bool GetSegmentsInBothEndcaps_Loose_dTcut_TrkMuUnVeto() const{ return segments_in_both_endcaps_loose_dtcut_TrkMuUnVeto;}


    // Get Reference to the Tracks
    edm::RefVector<reco::TrackCollection>& GetTracks(){return TheTrackRefs;}
    const edm::RefVector<reco::TrackCollection>& GetTracks()const {return TheTrackRefs;}
    
    // Set Number of Halo Triggers
    void SetNumberOfHaloTriggers(int PlusZ,  int MinusZ ){ nTriggers_PlusZ =PlusZ; nTriggers_MinusZ = MinusZ ;}
    void SetNumberOfHaloTriggers_TrkMuUnVeto(int PlusZ,  int MinusZ ){ nTriggers_PlusZ_TrkMuUnVeto =PlusZ; nTriggers_MinusZ_TrkMuUnVeto = MinusZ ;}
    // Set number of chamber-level triggers with non-collision timing
    void SetNOutOfTimeTriggers(short int PlusZ,short int MinusZ){ nOutOfTimeTriggers_PlusZ = PlusZ ; nOutOfTimeTriggers_MinusZ = MinusZ;}
    // Set number of CSCRecHits with non-collision timing
    void SetNOutOfTimeHits(short int num){ nOutOfTimeHits = num ;}
    // Set number of tracks with timing consistent with incoming particles
    void SetNIncomingTracks(short int n_small_dT, short int n_small_beta, short int n_small_both) {  nTracks_Small_dT = n_small_dT; 
      nTracks_Small_beta = n_small_beta; nTracks_Small_dT_Small_beta = n_small_both;}

    // Set HLT Bit
    void SetHLTBit(bool status) { HLTAccept = status ;} 

    // Get GlobalPoints of CSC tracking rechits nearest to the calorimeters
    //std::vector<const GlobalPoint>& GetCSCTrackImpactPositions() const {return TheGlobalPositions;}
    const std::vector<GlobalPoint>& GetCSCTrackImpactPositions() const {return TheGlobalPositions;}
    std::vector<GlobalPoint>& GetCSCTrackImpactPositions() {return TheGlobalPositions;}

    // MLR
    // Set # of CSCSegments that appear to be part of a halo muon
    // If there is more than 1 muon, this is the number of segments in the halo muon
    // with the largest number of segments that pass the cut.
    void SetNFlatHaloSegments(short int nSegments) {nFlatHaloSegments = nSegments;}
    void SetSegmentsBothEndcaps(bool b) { segments_in_both_endcaps = b; }
    // End MLR
    void SetNFlatHaloSegments_TrkMuUnVeto(short int nSegments) {nFlatHaloSegments_TrkMuUnVeto = nSegments;}
    void SetSegmentsBothEndcaps_Loose_TrkMuUnVeto(bool b) { segments_in_both_endcaps_loose_TrkMuUnVeto = b; }
    void SetSegmentsBothEndcaps_Loose_dTcut_TrkMuUnVeto(bool b) { segments_in_both_endcaps_loose_dtcut_TrkMuUnVeto = b; }
    void SetSegmentIsCaloMatched(bool b) { segmentiscalomatched = b; }

  private:
    edm::RefVector<reco::TrackCollection> TheTrackRefs;

    // The GlobalPoints from constituent rechits nearest to the calorimeter of CSC tracks
    std::vector<GlobalPoint> TheGlobalPositions;
    int nTriggers_PlusZ;
    int nTriggers_MinusZ;
    int nTriggers_PlusZ_TrkMuUnVeto;
    int nTriggers_MinusZ_TrkMuUnVeto;
    // CSC halo trigger reported by the HLT
    bool HLTAccept;
   
    int nTracks_PlusZ;
    int nTracks_MinusZ;

    // number of  out-of-time chamber-level triggers (assumes the event triggered at the bx of the beam crossing)
    short int nOutOfTimeTriggers_PlusZ;
    short int nOutOfTimeTriggers_MinusZ;
    // number of out-of-time CSCRecHit2Ds (assumes the event triggered at the bx of the beam crossing)
    short int nOutOfTimeHits;
    // number of cosmic muon outer (CSC) tracks with (T_segment_outer - T_segment_inner) < max_dt_muon_segment
    short int nTracks_Small_dT;
    // number of cosmic muon outer (CSC) tracks with free inverse beta < max_free_inverse_beta
    short int nTracks_Small_beta;
    // number of cosmic muon outer (CSC) tracks with both 
    // (T_segment_outer - T_segment_inner) <  max_dt_muon_segment and free inverse beta < max_free_inverse_beta
    short int nTracks_Small_dT_Small_beta;

    // MLR
    // number of CSCSegments that are flat and have the same (r,phi)
    short int nFlatHaloSegments;
    bool segments_in_both_endcaps;
    // end MLR
    short int nFlatHaloSegments_TrkMuUnVeto;
    bool segments_in_both_endcaps_loose_TrkMuUnVeto;
    bool segments_in_both_endcaps_loose_dtcut_TrkMuUnVeto;
    bool segmentiscalomatched ;
  };


  
}
  

#endif
