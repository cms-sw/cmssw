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

    // Get Reference to the Tracks
    edm::RefVector<reco::TrackCollection>& GetTracks(){return TheTrackRefs;}
    const edm::RefVector<reco::TrackCollection>& GetTracks()const {return TheTrackRefs;}
    
    // Set Number of Halo Triggers
    void SetNumberOfHaloTriggers(int PlusZ,  int MinusZ ){ nTriggers_PlusZ =PlusZ; nTriggers_MinusZ = MinusZ ;}

    // Set number of chamber-level triggers with non-collision timing
    void SetNOutOfTimeTriggers(short int PlusZ,short int MinusZ){ nOutOfTimeTriggers_PlusZ = PlusZ ; nOutOfTimeTriggers_MinusZ = MinusZ;}
    // Set number of CSCRecHits with non-collision timing
    void SetNOutOfTimeHits(short int num){ nOutOfTimeHits = num ;}

    // Set HLT Bit
    void SetHLTBit(bool status) { HLTAccept = status ;} 

    // Get GlobalPoints of CSC tracking rechits nearest to the calorimeters
    //std::vector<const GlobalPoint>& GetCSCTrackImpactPositions() const {return TheGlobalPositions;}
    const std::vector<GlobalPoint>& GetCSCTrackImpactPositions() const {return TheGlobalPositions;}
    std::vector<GlobalPoint>& GetCSCTrackImpactPositions() {return TheGlobalPositions;}
    
  private:
    edm::RefVector<reco::TrackCollection> TheTrackRefs;

    // The GlobalPoints from constituent rechits nearest to the calorimeter of CSC tracks
    std::vector<GlobalPoint> TheGlobalPositions;
    int nTriggers_PlusZ;
    int nTriggers_MinusZ;

    // CSC halo trigger reported by the HLT
    bool HLTAccept;
   
    int nTracks_PlusZ;
    int nTracks_MinusZ;

    // number of  out-of-time chamber-level triggers (assumes the event triggered at the bx of the beam crossing)
    short int nOutOfTimeTriggers_PlusZ;
    short int nOutOfTimeTriggers_MinusZ;
    // number of out-of-time CSCRecHit2Ds (assumes the event triggered at the bx of the beam crossing)
    short int nOutOfTimeHits;

  };


  
}
  

#endif
