#ifndef FastSimulation_Muons_FastTSGFromL2Muon_H
#define FastSimulation_Muons_FastTSGFromL2Muon_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"


#include <vector>
namespace edm { class ParameterSet; class Event; class EventSetup; }

class MuonServiceProxy;
class MuonTrackingRegionBuilder;
class RectangularEtaPhiTrackingRegion;
class SimTrack;
//class TH1F;

//
// generate seeds corresponding to L2 muons
//

class FastTSGFromL2Muon : public edm::stream::EDProducer <> {

 public:

  FastTSGFromL2Muon(const edm::ParameterSet& cfg);
  virtual ~FastTSGFromL2Muon();
  virtual void beginRun(edm::Run const& run, edm::EventSetup const& es) override;
  virtual void produce(edm::Event& ev, const edm::EventSetup& es) override;
  
 private:

  bool clean(reco::TrackRef muRef,
	     RectangularEtaPhiTrackingRegion* region,
	     const BasicTrajectorySeed* aSeed, 
	     const SimTrack& theSimTrack); 

 private:

  edm::ParameterSet theConfig;
  edm::InputTag theSimTrackCollectionLabel;
  edm::InputTag theL2CollectionLabel;
  std::vector<edm::InputTag> theSeedCollectionLabels;

  // bool useTFileService_;

  MuonServiceProxy* theService;
  double thePtCut;
  MuonTrackingRegionBuilder* theRegionBuilder;

  // TH1F* h_nSeedPerTrack;
  // TH1F* h_nGoodSeedPerTrack;
  // TH1F* h_nGoodSeedPerEvent;

};
#endif
