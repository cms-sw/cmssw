#ifndef RecoTracker_TrackProducer_FamosRecHitTest_h
#define RecoTracker_TrackProducer_FamosRecHitTest_h 

#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackerGeometry;

class FamosRecHitTest : public edm::stream::EDAnalyzer <>
{
public:

  explicit FamosRecHitTest(const edm::ParameterSet& pset);

  virtual ~FamosRecHitTest();
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);

private:

};
#endif
