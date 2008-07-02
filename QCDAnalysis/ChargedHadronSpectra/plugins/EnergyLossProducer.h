#ifndef EnergyLossProducer_H
#define EnergyLossProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;

namespace edm { class Event; class EventSetup; }
class TrackerGeometry;

class EnergyLossProducer : public edm::EDProducer
{
public:
  explicit EnergyLossProducer(const edm::ParameterSet& ps);
  ~EnergyLossProducer();
  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:
  void beginJob(const edm::EventSetup& es);

  string trackProducer;
  double pixelToStripMultiplier, pixelToStripExponent;
  const TrackerGeometry * theTracker;
};
#endif

