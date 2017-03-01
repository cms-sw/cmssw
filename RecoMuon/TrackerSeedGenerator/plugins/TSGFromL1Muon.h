#ifndef RecoMuon_TrackerSeedGenerator_TSGFromL1Muon_H
#define RecoMuon_TrackerSeedGenerator_TSGFromL1Muon_H

/** \class TSGFromL1Muon
 * Description: 
 * EDPRoducer to generate L3MuonTracjectorySeed from L1MuonParticles
 * \author Marcin Konecki
*/


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

namespace edm { class Event; class EventSetup; }
class L1MuonRegionProducer;
class L1MuonPixelTrackFitter;
class OrderedHitsGenerator;
class PixelTrackFilter;
class L1MuonSeedsMerger;


class TSGFromL1Muon : public edm::stream::EDProducer<> {
public:
  TSGFromL1Muon(const edm::ParameterSet& cfg);
  virtual ~TSGFromL1Muon();
  virtual void beginRun(const edm::Run & run, const edm::EventSetup&es) override;
  virtual void produce(edm::Event& ev, const edm::EventSetup& es) override;
private:
 
private:
  edm::ParameterSet theConfig;
  edm::InputTag theSourceTag;
  edm::EDGetTokenT<l1extra::L1MuonParticleCollection> theSourceToken; 
  edm::EDGetTokenT<PixelTrackFilter> theFilterToken;

  std::unique_ptr<L1MuonRegionProducer> theRegionProducer;
  OrderedHitsGenerator * theHitGenerator;
  std::unique_ptr<L1MuonPixelTrackFitter> theFitter;
  L1MuonSeedsMerger * theMerger;

};
#endif
