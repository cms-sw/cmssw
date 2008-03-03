#ifndef RecoMuon_TrackerSeedCleaner_H
#define RecoMuon_TrackerSeedCleaner_H

/** \class TrackerSeedCleaner
 *  Seeds Cleaner based on direction
 *  $Date: 2008/02/28 22:17:48 $
 *  $Revision: 1.0 $
    \author A. Grelli -  Purdue University, Pavia University
 */ 

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

class MuonServiceProxy;
class TSGFromL2Muon;
class MuonTrackingRegionBuilder;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class TrackerSeedCleaner{

public:

  typedef std::vector<TrajectorySeed> tkSeeds;

  TrackerSeedCleaner(const edm::ParameterSet& pset) : theProxyService(0),theEvent(0) {
                   builderName = pset.getParameter<std::string>("TTRHBuilder");
                   theBeamSpotTag = pset.getParameter<edm::InputTag>("beamSpot");
  }

  //intizialization
  virtual void init(const MuonServiceProxy *service);
  
  /// destructor
  virtual ~TrackerSeedCleaner() {}
  //the cleaner
  virtual std::vector<TrajectorySeed >  clean(const reco::TrackRef& , const RectangularEtaPhiTrackingRegion& region, tkSeeds&);

  virtual void setEvent(const edm::Event&);

private:

  const MuonServiceProxy * theProxyService;
  const edm::Event * theEvent;

  edm::InputTag theBeamSpotTag; //beam spot
  std::string builderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder; 
};

#endif

