#ifndef L1Trigger_TrackFindingTMTT_TrkRZfilter_h
#define L1Trigger_TrackFindingTMTT_TrkRZfilter_h

#include "L1Trigger/TrackFindingTMTT/interface/L1track2D.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"

#include <vector>
#include <list>

//=== This class runs filters in track candidates previously found by the r-phi Hough transform,
//=== which check that each track's stubs are consistent with a straight line in the r-z plane.
//===
//=== The filtering removes inconsistent stubs from the track candidates, & also kills some track candidates
//=== altogether if the filter leaves them with too few stubs.
//===
//=== The r-z filters also add an estimate of the r-z helix parameters to the selected track candidates,
//=== The filtered tracks are returned at L1track3D type, since they contain this information.
//===
//=== It does NOT contain filters such as the bend filter, which are so simple that the firmware can run them
//=== INSIDE the r-phi HT. Simple filters of this kind are in class HTcell.
//===
//=== After creating the L1track3D tracks, TrkRZfilter can optionally run duplicate removal on them,
//=== before they are output.

namespace tmtt {

  class Settings;
  class Stub;

  class TrkRZfilter {
  public:
    // Initialize configuration parameters, and note sector number, eta range covered by sector and phi coordinate of its centre.
    TrkRZfilter(const Settings* settings,
                unsigned int iPhiSec,
                unsigned int iEtaReg,
                float etaMinSector,
                float etaMaxSector,
                float phiCentreSector);

    // Filters track candidates (found by the r-phi Hough transform), removing inconsistent stubs from the tracks,
    // also killing some of the tracks altogether if they are left with too few stubs.
    // Also adds an estimate of r-z helix parameters to the selected track objects, returning the tracks as L1track3D type.
    //
    std::list<L1track3D> filterTracks(const std::list<L1track2D>& tracks);

    //=== Extra information about each track input to filter. (Only use after you have first called filterTracks).

    // Number of seed combinations considered by the Seed Filter for each input track.
    const std::vector<unsigned int>& numSeedCombsPerTrk() const { return numSeedCombsPerTrk_; }
    const std::vector<unsigned int>& numGoodSeedCombsPerTrk() const {
      return numGoodSeedCombsPerTrk_;
    }  // Only counts seeds compatible with beam-spot.

  private:
    //--- Filters returning filtered stubs based on input ones.

    // Use Seed Filter to produce a filtered collection of stubs on this track candidate that are consistent with a straight line
    // in r-z using tracklet algo.
    std::vector<Stub*> seedFilter(const std::vector<Stub*>& stubs, float trkQoverPt, bool print);

    //--- Estimate r-z helix parameters from centre of eta-sector if no better estimate provided by r-z filter.
    void estRZhelix();

  private:
    //=== Configuration parameters

    const Settings* settings_;

    unsigned int iPhiSec_;  // Sector number.
    unsigned int iEtaReg_;
    float etaMinSector_;  // rapidity range of this sector.
    float etaMaxSector_;
    float phiCentreSector_;  // phi coordinate of its centre.

    // Track (z0, tan_lambda) estimate from r-z filter or centre of eta sector, and boolean to indicate if this data is filled.
    float rzHelix_z0_;
    float rzHelix_tanL_;
    bool rzHelix_set_;

    // Useful info for r-z filters.
    float chosenRofZ_;     // Radius used to defined zTrkMinSector and zTrkMaxSector.
    float zTrkMinSector_;  // corresponding range of this sector specified as z coordinate of track at given radius.
    float zTrkMaxSector_;
    float beamWindowZ_;  // Assumed length of beam spot in z.

    // Name of r-z track filter algorithm to run.
    std::string rzFilterName_;

    // Filter stubs in cell using Seed Filter? (a tracklet-like algorithm in r-z plane).
    bool useSeedFilter_;

    // Options for Seed filter.
    bool keepAllSeed_;
    float seedResCut_;

    // Number of seed combinations considered by the Seed Filter, for each input track.
    std::vector<unsigned int> numSeedCombsPerTrk_;
    std::vector<unsigned int> numGoodSeedCombsPerTrk_;
    unsigned int maxSeedCombinations_;
    unsigned int maxGoodSeedCombinations_;
    unsigned int maxSeedsPerStub_;
    bool zTrkSectorCheck_;

    // For debugging
    unsigned int minNumMatchLayers_;
  };

}  // namespace tmtt

#endif
