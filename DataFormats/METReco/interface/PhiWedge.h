#ifndef DATAFORMATS_METRECO_PHIWEDGE_H
#define DATAFORMATS_METRECO_PHIWEDGE_H
/*
  [class]:  PhiWedge
  [authors]: R. Remington, The University of Florida
  [description]: Simple class analogous to CaloTower but in z-direction.  Stores basic information related to Hcal and Ecal rechits within constant 5-degree phi windows.  The idea will be to match these reconstructed phi-wedges with csc tracks for BeamHalo identification.
  [date]: October 15, 2009
*/
#include "TMath.h"
#include <vector>
namespace reco {

  class PhiWedge {
  public:
    // Constructors
    PhiWedge();
    PhiWedge(float E, int iphi, int constituents);
    PhiWedge(float E, int iphi, int constituents, float min_time, float max_time);
    PhiWedge(const PhiWedge&);
    // Destructors

    ~PhiWedge() {}

    // Energy sum of all rechits above threshold in this 5-degree window
    float Energy() const { return energy_; }

    // Number of rechits above threshold in this 5-degree window
    int NumberOfConstituents() const { return constituents_; }

    // iPhi value of this 5-degree window
    int iPhi() const { return iphi_; }

    // Global phi lower bound of this 5-degree window (between 0 and 2Pi)
    float PhiLow() const { return 2. * TMath::Pi() * (float)((iphi_ * 5) - (5.)); }

    // Global phi upper bound of this 5-degree window (between 0 and 2Pi
    float PhiHigh() const { return 2. * TMath::Pi() * (float)((iphi_ * 5)); }

    // Get Min/Max Time
    float MinTime() const { return min_time_; }
    float MaxTime() const { return max_time_; }

    // Get halo direction confidence based on time ordering of the rechits ( must be within range of -1 to + 1 )
    // Direction is calculated by counting the number of pair-wise time-ascending rechits from -Z to +Z and then normalizing this count by number of pair-wise combinations
    // If all pair-wise combinations are consistent with a common z-direction, then this value will be plus or minus 1 exactly.  Otherwise it will be somewhere in between.
    float ZDirectionConfidence() const { return (1. - PlusZOriginConfidence_) * 2. - 1.; }
    float PlusZDirectionConfidence() const { return 1. - PlusZOriginConfidence_; }
    float MinusZDirectionConfidence() const { return PlusZOriginConfidence_; }

    // Get halo origin confidence based on time ordering of the rechits
    float PlusZOriginConfidence() const { return PlusZOriginConfidence_; }
    float MinusZOriginConfidence() const { return 1. - PlusZOriginConfidence_; }

    // To be filled later or removed
    int OverlappingCSCTracks() const { return OverlappingCSCTracks_; }
    int OverlappingCSCSegments() const { return OverlappingCSCSegments_; }
    int OverlappingCSCRecHits() const { return OverlappingCSCRecHits_; }
    int OverlappingCSCHaloTriggers() const { return OverlappingCSCHaloTriggers_; }

    // Setters
    void SetOverlappingCSCTracks(int x) { OverlappingCSCTracks_ = x; }
    void SetOverlappingCSCSegments(int x) { OverlappingCSCSegments_ = x; }
    void SetOverlappingCSCRecHits(int x) { OverlappingCSCRecHits_ = x; }
    void SetOverlappingCSCHaloTriggers(int x) { OverlappingCSCHaloTriggers_ = x; }
    void SetMinMaxTime(float min, float max) {
      min_time_ = min;
      max_time_ = max;
    }
    void SetPlusZOriginConfidence(float x) { PlusZOriginConfidence_ = x; }

  private:
    float energy_;
    int iphi_;
    int constituents_;
    float min_time_;
    float max_time_;
    float PlusZOriginConfidence_;
    int OverlappingCSCTracks_;
    int OverlappingCSCSegments_;
    int OverlappingCSCRecHits_;
    int OverlappingCSCHaloTriggers_;
  };
  typedef std::vector<PhiWedge> PhiWedgeCollection;
}  // namespace reco
#endif
