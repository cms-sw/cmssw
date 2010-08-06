// !!
#include "QCDAnalysis/ChargedHadronSpectra/interface/HitInfo.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <sstream>
using namespace std;

/*****************************************************************************/
HitInfo::HitInfo()
{
}

/*****************************************************************************/
HitInfo::~HitInfo()
{
}

/*****************************************************************************/
string HitInfo::getInfo(const DetId & id)
{
  std::string info;

  if(id.subdetId() == int(PixelSubdetector::PixelBarrel))
  {
    // 0 + (layer-1)<<1 + (ladder-1)%2 : 0-5
    PXBDetId pid(id);
    ostringstream o;
    o << " (" << pid.layer()  << "|" << pid.ladder()
      <<  "|" << pid.module() << ")";
    info += o.str();
  }
  else
  {
    // 6 + (disk-1)<<1 + (panel-1)%2
    PXFDetId pid(id);
    ostringstream o;
    o << " (" << pid.side()   << "|" << pid.disk()
      <<  "|" << pid.blade()  << "|" << pid.panel()
      <<  "|" << pid.module() << ")";
    info += o.str();
  }

  return info;
}

/*****************************************************************************/
string HitInfo::getInfo(const TrackingRecHit & recHit)
{
  DetId id(recHit.geographicalId());

  return getInfo(id);
}

/*****************************************************************************/
string HitInfo::getInfo(std::vector<const TrackingRecHit *> recHits)
{
  std::string info;

  for(std::vector<const TrackingRecHit *>::const_iterator
        recHit = recHits.begin();
        recHit!= recHits.end(); recHit++)
     info += getInfo(**recHit);

  return info;
}

/*****************************************************************************/
string HitInfo::getInfo(const PSimHit & simHit)
{
  std::string info;

  DetId id = DetId(simHit.detUnitId());

  {
    ostringstream o;
    o << simHit.particleType();

    info += " | pid=" + o.str();
  }

  {
    ostringstream o;
//    o << theTracker->idToDet(id)->subDetector();
    o << id.subdetId();

    info += " | " + o.str();
  }

  return info + getInfo(id);;
}

