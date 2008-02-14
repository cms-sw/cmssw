#ifndef RecoMuon_TrackerSeedGenerator_TSGFromL1Muon_H
#define RecoMuon_TrackerSeedGenerator_TSGFromL1Muon_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm { class Event; class EventSetup; }
class PixelFitter;
class OrderedHitsGenerator;

//
// generate seeds corresponding to L1 muons
//

class TSGFromL1Muon : public edm::EDProducer {
public:
  TSGFromL1Muon(const edm::ParameterSet& cfg);
  virtual ~TSGFromL1Muon();
  virtual void beginJob(const edm::EventSetup& es);
  virtual void produce(edm::Event& ev, const edm::EventSetup& es);
private:
  float deltaPhi(float phi1, float phi2) const;
  float getPt(float phi0, float phiL1, float eta, float charge) const;
  float getBending(float eta, float pt, float charge) const;
  void param(float eta, float &p1, float& p2) const;
 
private:
  edm::ParameterSet theConfig;
  edm::InputTag theSourceTag;
  OrderedHitsGenerator * theHitGenerator;
  const PixelFitter       * theFitter;

};
#endif
