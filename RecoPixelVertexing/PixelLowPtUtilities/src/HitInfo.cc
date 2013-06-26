#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

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
string HitInfo::getInfo(const DetId & id, const TrackerTopology *tTopo)
{
  string info;

  if(id.subdetId() == int(PixelSubdetector::PixelBarrel))
  {
    // 0 + (layer-1)<<1 + (ladder-1)%2 : 0-5
    
    ostringstream o;
    o << " (" << tTopo->pxbLayer(id)  << "|" << tTopo->pxbLadder(id)
      <<  "|" << tTopo->pxbModule(id) << ")";
    info += o.str();
  }
  else
  {
    // 6 + (disk-1)<<1 + (panel-1)%2
    
    ostringstream o;
    o << " (" << tTopo->pxfSide(id)   << "|" << tTopo->pxfDisk(id)
      <<  "|" << tTopo->pxfBlade(id)  << "|" << tTopo->pxfPanel(id)
      <<  "|" << tTopo->pxfModule(id) << ")";
    info += o.str();
  }

  return info;
}

/*****************************************************************************/
string HitInfo::getInfo(const TrackingRecHit & recHit, const TrackerTopology *tTopo)
{
  DetId id(recHit.geographicalId());

  return getInfo(id, tTopo);
}

/*****************************************************************************/
string HitInfo::getInfo(vector<const TrackingRecHit *> recHits, const TrackerTopology *tTopo)
{
  string info;

  for(vector<const TrackingRecHit *>::const_iterator
        recHit = recHits.begin();
        recHit!= recHits.end(); recHit++)
    info += getInfo(**recHit, tTopo);

  return info;
}

/*****************************************************************************/
string HitInfo::getInfo(const PSimHit & simHit, const TrackerTopology *tTopo)
{
  string info;

  DetId id = DetId(simHit.detUnitId());

  {
    ostringstream o;
    o << simHit.particleType();

    info += " | pid=" + o.str();
  }

  {
    ostringstream o;
    o << id.subdetId();

    info += " | " + o.str();
  }

  return info + getInfo(id, tTopo);
}

