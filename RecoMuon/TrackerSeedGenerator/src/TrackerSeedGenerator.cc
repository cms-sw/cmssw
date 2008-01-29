#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"


#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

void TrackerSeedGenerator::init(const MuonServiceProxy *service)
{
  theProxyService = service;
}

TrackerSeedGenerator::BTSeedCollection TrackerSeedGenerator::trackerSeeds(const TrackCand& can, 
   const TrackingRegion& region)
{
  BTSeedCollection result;
  const edm::EventSetup & es = theProxyService->eventSetup();
  run(result, *theEvent, es, region); 
  return result;
}

void TrackerSeedGenerator::setEvent(const edm::Event& event)
{
  theEvent = &event;
}
