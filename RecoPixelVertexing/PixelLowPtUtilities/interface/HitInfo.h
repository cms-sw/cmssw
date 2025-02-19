#ifndef _HitInfo_h_
#define _HitInfo_h_

class DetId;
class TrackingRecHit;
class PSimHit;

#include <vector>
#include <string>

class HitInfo
{
public:
  HitInfo();
  ~HitInfo();

  static std::string getInfo(const DetId & id);
  static std::string getInfo(const TrackingRecHit & recHit);
  static std::string getInfo(std::vector<const TrackingRecHit *> recHits);
  static std::string getInfo(const PSimHit & simHit);
};

#endif
