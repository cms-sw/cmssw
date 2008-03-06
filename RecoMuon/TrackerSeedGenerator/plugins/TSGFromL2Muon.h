#ifndef RecoMuon_TrackerSeedGenerator_TSGFromL2Muon_H
#define RecoMuon_TrackerSeedGenerator_TSGFromL2Muon_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



namespace edm { class ParameterSet; class Event; class EventSetup; }

class MuonServiceProxy;
class TrackerSeedGenerator;
class MuonTrackingRegionBuilder;
class TrackerSeedCleaner;
class TH1F;

//
// generate seeds corresponding to L2 muons
//

class TSGFromL2Muon : public edm::EDProducer {
public:
  TSGFromL2Muon(const edm::ParameterSet& cfg);
  virtual ~TSGFromL2Muon();
  virtual void beginJob(const edm::EventSetup& es);
  virtual void produce(edm::Event& ev, const edm::EventSetup& es);
 
private:
  edm::ParameterSet theConfig;
  edm::InputTag theL2CollectionLabel;

  bool useTFileService_;

  MuonServiceProxy* theService;
  double thePtCut;
  MuonTrackingRegionBuilder* theRegionBuilder;
  TrackerSeedGenerator* theTkSeedGenerator;
  TrackerSeedCleaner* theSeedCleaner;

  TH1F* h_nSeedPerTrack;
  TH1F* h_nGoodSeedPerTrack;
  TH1F* h_nGoodSeedPerEvent;
};
#endif
