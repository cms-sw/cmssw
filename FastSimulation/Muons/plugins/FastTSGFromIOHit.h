#ifndef FastSimulation_Muons_FastTSGFromIOHit_H
#define FastSimulation_Muons_FastTSGFromIOHit_H

/** \class FastTSGFromIOHit
 *  Tracker Seed Generator by propagating and updating a standAlone muon
 *  to the first pixel hits in tracker system 
 *
 *  Emulate TSGFromIOHit in RecoMuon
 *
 *  \author Adam Everett - Purdue University 
 */

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include <vector>

class RectangularEtaPhiTrackingRegion;
class TrackingRegion;
class SimTrack;

class FastTSGFromIOHit : public TrackerSeedGenerator {

public:
  /// constructor
  FastTSGFromIOHit(const edm::ParameterSet &pset,edm::ConsumesCollector& iC);

  /// destructor
  ~FastTSGFromIOHit() override;

  /// generate seed(s) for a track
    void  trackerSeeds(const TrackCand&, const TrackingRegion&, const TrackerTopology *tTopo, std::vector<TrajectorySeed>&) override;

 private:
  bool clean(reco::TrackRef muRef,
  	     const RectangularEtaPhiTrackingRegion& region,
  	     const BasicTrajectorySeed* aSeed,
  	     const SimTrack& theSimTrack); 

private:
  std::string theCategory;
    
    edm::EDGetTokenT<edm::SimTrackContainer> simTracksTk;
    std::vector<edm::EDGetTokenT<TrajectorySeedCollection> > seedsTks;
    double thePtCut;

};

#endif 
