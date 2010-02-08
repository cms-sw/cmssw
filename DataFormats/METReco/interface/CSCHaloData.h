#ifndef DATAFORMATS_METRECO_CSCHALODATA_H
#define DATAFORMATS_METRECO_CSCHALODATA_H
#include "TMath.h"
#include <vector>

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
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
    int NumberOfHaloTriggers (int z=0) const ;
    // Number of Halo Tracks in +/-  endcap
    int NumberOfHaloTracks(int z=0) const ;

    // Halo trigger bit from the HLT  
    bool CSCHaloHLTAccept() const {return HLTAccept;}

    // Get Reference to the Tracks
    edm::RefVector<reco::TrackCollection>& GetTracks(){return TheTrackRefs;}
    const edm::RefVector<reco::TrackCollection>& GetTracks()const {return TheTrackRefs;}
    
    // Set Number of Halo Triggers
    void SetNumberOfHaloTriggers(int PlusZ,  int MinusZ ){ nTriggers_PlusZ =PlusZ; nTriggers_MinusZ = MinusZ ;}

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
  };
  
}
  

#endif
