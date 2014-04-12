#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

void TrackerSeedGenerator::init(const MuonServiceProxy *service)
{
  theProxyService = service;
}

void  TrackerSeedGenerator::trackerSeeds(const TrackCand& can, 
					 const TrackingRegion& region, 
					 const TrackerTopology *tTopo, TrackerSeedGenerator::BTSeedCollection & result)
{
  const edm::EventSetup & es = theProxyService->eventSetup();
  run(result, *theEvent, es, region); 
  return ;
}
void TrackerSeedGenerator::setEvent(const edm::Event& event)
{
  theEvent = &event;
}
