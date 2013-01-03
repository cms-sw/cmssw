#ifndef _HitInfo_h_
#define _HitInfo_h_

#include "FWCore/Framework/interface/ESHandle.h"

class DetId;
class TrackerTopology;
class TrackingRecHit;
class PSimHit;

#include <vector>
#include <string>

class HitInfo
{
public:
  HitInfo();
  ~HitInfo();

  static std::string getInfo(const DetId & id, const TrackerTopology* tTopo);
  static std::string getInfo(const TrackingRecHit & recHit, const TrackerTopology* tTopo);
  static std::string getInfo(std::vector<const TrackingRecHit *> recHits, const TrackerTopology* tTopo);
  static std::string getInfo(const PSimHit & simHit, const TrackerTopology* tTopo);
};

#endif
