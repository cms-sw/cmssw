#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ValidHitPairFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

using namespace std;

enum { BPix1=1, BPix2=2, BPix3=3,
       FPix1_neg=4, FPix2_neg=5,
       FPix1_pos=6, FPix2_pos=7 };

/*****************************************************************************/
float spin(float ph)
{
  if(ph < 0) ph += 2 * M_PI;
  return ph;
}

/*****************************************************************************/
ValidHitPairFilter::ValidHitPairFilter
  (const edm::ParameterSet& ps, edm::ConsumesCollector& iC)
{
}

/*****************************************************************************/
ValidHitPairFilter::~ValidHitPairFilter()
{
}
/*****************************************************************************/
void ValidHitPairFilter::update(const edm::Event& ev, const edm::EventSetup& es) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<TrackerTopologyRcd>().get(tTopo);


  // Get tracker
  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  theTracker = trackerHandle.product();

  // Get geometric search tracker
  edm::ESHandle<GeometricSearchTracker> geometricSearchTrackerHandle;
  es.get<TrackerRecoGeometryRecord>().get(geometricSearchTrackerHandle);
  theGSTracker = geometricSearchTrackerHandle.product();

  // Get magnetic field
  edm::ESHandle<MagneticField> magneticFieldHandle;
  es.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  theMagneticField = magneticFieldHandle.product();

  // Get propagator
  edm::ESHandle<Propagator> thePropagatorHandle;
  es.get<TrackingComponentsRecord>().get("AnalyticalPropagator",
                                          thePropagatorHandle);
  thePropagator = thePropagatorHandle.product();

  // Bounds, hardwired FIXME
  rzBounds[0].resize(8); phBounds[0].resize(20);
  rzBounds[1].resize(8); phBounds[1].resize(32);
  rzBounds[2].resize(8); phBounds[2].resize(44);

  rzBounds[3].resize(7); phBounds[3].resize(24);
  rzBounds[4].resize(7); phBounds[4].resize(24);
  rzBounds[5].resize(7); phBounds[5].resize(24);
  rzBounds[6].resize(7); phBounds[6].resize(24);

  LogTrace("MinBiasTracking")
    << " [ValidHitPairFilter] initializing pixel barrel";

  for(TrackerGeometry::DetContainer::const_iterator
      det = theTracker->detsPXB().begin();
      det!= theTracker->detsPXB().end(); det++)
  {
    
    DetId pid=(*det)->geographicalId();
    int il  = tTopo->pxbLayer(pid)  - 1;
    int irz = tTopo->pxbModule(pid) - 1;
    int iph = tTopo->pxbLadder(pid) - 1;
  
    rzBounds[il][irz] = (*det)->position().z();
    phBounds[il][iph] = spin((*det)->position().phi());
  }

  LogTrace("MinBiasTracking")
    << " [ValidHitPairFilter] initializing pixel endcap";

  for(TrackerGeometry::DetContainer::const_iterator
      det = theTracker->detsPXF().begin();
      det!= theTracker->detsPXF().end(); det++)
  {
    

    DetId pid=(*det)->geographicalId();
    int il  = BPix3 + ((tTopo->pxfSide(pid)  -1) << 1) + (tTopo->pxfDisk(pid) -1);
    int irz =         ((tTopo->pxfModule(pid)-1) << 1) + (tTopo->pxfPanel(pid)-1);
    int iph =          (tTopo->pxfBlade(pid) -1);

    rzBounds[il][irz] = (*det)->position().perp();
    phBounds[il][iph] = spin((*det)->position().phi());
  }

  for(int i = 0; i < 7; i++)
  {
    sort(rzBounds[i].begin(), rzBounds[i].end());
    sort(phBounds[i].begin(), phBounds[i].end());
  }
}

/*****************************************************************************/
int ValidHitPairFilter::getLayer(const TrackingRecHit & recHit, const TrackerTopology *tTopo) const
{
  DetId id(recHit.geographicalId());

  if(id.subdetId() == int(PixelSubdetector::PixelBarrel))
  {
    
    return tTopo->pxbLayer(id);
  }
  else
  {
    
    return BPix3 + ((tTopo->pxfSide(id)-1) << 1) + tTopo->pxfDisk(id);
  }
}

/*****************************************************************************/
vector<int> ValidHitPairFilter::getMissingLayers(int a, int b) const
{
  vector<int> l;
  pair<int,int> c(a,b);

  if(c == pair<int,int>(BPix1,BPix2))     { l.push_back(int(BPix3));
                                            l.push_back(int(FPix1_pos));
                                            l.push_back(int(FPix1_neg)); return l; }
  if(c == pair<int,int>(BPix1,BPix3))     { l.push_back(int(BPix2));     return l; }
  if(c == pair<int,int>(BPix2,BPix3))     { l.push_back(int(BPix1));     return l; }
  if(c == pair<int,int>(BPix1,FPix1_pos)) { l.push_back(int(BPix2));
                                            l.push_back(int(FPix2_pos)); return l; }
  if(c == pair<int,int>(BPix1,FPix1_neg)) { l.push_back(int(BPix2)); 
                                            l.push_back(int(FPix2_neg)); return l; }
  if(c == pair<int,int>(BPix1,FPix2_pos)) { l.push_back(int(BPix2)); 
                                            l.push_back(int(FPix1_pos)); return l; }
  if(c == pair<int,int>(BPix1,FPix2_neg)) { l.push_back(int(BPix2)); 
                                            l.push_back(int(FPix1_neg)); return l; }
  if(c == pair<int,int>(BPix2,FPix1_pos)) { l.push_back(int(BPix1)); 
                                            l.push_back(int(FPix2_pos)); return l; }
  if(c == pair<int,int>(BPix2,FPix1_neg)) { l.push_back(int(BPix1)); 
                                            l.push_back(int(FPix2_neg)); return l; }
  if(c == pair<int,int>(BPix2,FPix2_pos)) { l.push_back(int(BPix1));
                                            l.push_back(int(FPix1_pos)); return l; }
  if(c == pair<int,int>(BPix2,FPix2_neg)) { l.push_back(int(BPix1)); 
                                            l.push_back(int(FPix1_neg)); return l; }
  if(c == pair<int,int>(FPix1_pos,FPix2_pos)) { l.push_back(int(BPix1)); return l; }
  if(c == pair<int,int>(FPix1_neg,FPix2_neg)) { l.push_back(int(BPix1)); return l; }

  return l;
}

/*****************************************************************************/
FreeTrajectoryState ValidHitPairFilter::getTrajectory
  (const reco::Track & track) const
{
  GlobalPoint position(track.vertex().x(),
                       track.vertex().y(),
                       track.vertex().z());

  GlobalVector momentum(track.momentum().x(),
                        track.momentum().y(),
                        track.momentum().z());
    
  GlobalTrajectoryParameters gtp(position, momentum,
                                 track.charge(), theMagneticField);
  
  FreeTrajectoryState fts(gtp);

  return fts;
}

/*****************************************************************************/
vector<const GeomDet *> ValidHitPairFilter::getCloseDets
  (int il,
   float rz, const vector<float>& rzB,
   float ph, const vector<float>& phB,
   const TrackerTopology *tTopo) const
{
  vector<int> rzVec, phVec;

  // Radius or z
  vector<float>::const_iterator irz = lower_bound(rzB.begin(),rzB.end(), rz);
  if(irz > rzB.begin()) rzVec.push_back(irz - rzB.begin() - 1);
  if(irz < rzB.end()  ) rzVec.push_back(irz - rzB.begin()    );

  // Phi, circular
  vector<float>::const_iterator iph = lower_bound(phB.begin(),phB.end(), ph);
  if(iph > phB.begin()) phVec.push_back(iph         - phB.begin() - 1);
                   else phVec.push_back(phB.end()   - phB.begin() - 1);
  if(iph < phB.end()  ) phVec.push_back(iph         - phB.begin()    );
                   else phVec.push_back(phB.begin() - phB.begin()    );

  // Detectors
  vector<const GeomDet *> dets;

  for(vector<int>::const_iterator irz = rzVec.begin(); irz!= rzVec.end(); irz++)
  for(vector<int>::const_iterator iph = phVec.begin(); iph!= phVec.end(); iph++)
  {
    if(il+1 <= BPix3)
    {
      int layer  =  il  + 1;
      int ladder = *iph + 1;
      int module = *irz + 1;

      LogTrace("MinBiasTracking")
	<< "  [ValidHitPairFilter]  third ("<<layer<< "|"<<ladder<<"|"<<module<<")";

      DetId id=tTopo->pxbDetId(layer,ladder,module);
      dets.push_back(theTracker->idToDet(id));
    }
    else
    {
      int side = (il - BPix3) / 2 +1;  
      int disk   = (il - BPix3) % 2 + 1;
      int blade  =  *iph + 1;
      int panel  = (*irz) % 2 + 1;
      int module = (*irz) / 2 + 1;

      LogTrace("MinBiasTracking")
	<< "  [ValidHitPairFilter]  third ("<<side<<"|"<<disk<<"|"<<blade<<"|"<<panel<<"|"<<module<<")";

     
      DetId id=tTopo->pxfDetId(side,disk,blade,panel,module);
      dets.push_back(theTracker->idToDet(id));
      
    }
  }

  return dets;
}

/*****************************************************************************/
bool ValidHitPairFilter::operator() 
  (const reco::Track * track, const vector<const TrackingRecHit *>& recHits,
   const TrackerTopology *tTopo) const
{
  bool hasGap = true;

  if(recHits.size() == 2)
  {
    LogTrace("MinBiasTracking")
      << "  [ValidHitPairFilter] pair" << HitInfo::getInfo(*(recHits[0]),tTopo)
      << HitInfo::getInfo(*(recHits[1]),tTopo);

    float tol = 0.1; // cm
    float sc  = -1.; // scale, allow 'tol' of edge to count as outside
    LocalError le(tol*tol, tol*tol, tol*tol);

    // determine missing layers
    vector<int> missingLayers = getMissingLayers(getLayer(*(recHits[0]),tTopo),
                                                 getLayer(*(recHits[1]),tTopo));

    for(vector<int>::const_iterator missingLayer = missingLayers.begin();
                                    missingLayer!= missingLayers.end();
                                    missingLayer++)
    {
      int il = *missingLayer - 1;

      // propagate to missing layer
      FreeTrajectoryState fts = getTrajectory(*track);
      TrajectoryStateOnSurface tsos;
      float rz = 0.;
  
      if(il < BPix3)
      { // barrel
        const BarrelDetLayer * layer =
          (theGSTracker->pixelBarrelLayers())[il];
  
        tsos = thePropagator->propagate(fts, layer->surface());

        if(tsos.isValid())
          rz = tsos.globalPosition().z();
      }
      else
      { // endcap
        const ForwardDetLayer * layer;
        if(il - BPix3 < 2)
          layer = (theGSTracker->negPixelForwardLayers())[il - BPix3    ];
        else
          layer = (theGSTracker->posPixelForwardLayers())[il - BPix3 - 2];

        tsos = thePropagator->propagate(fts, layer->surface());

        if(tsos.isValid())
          rz = tsos.globalPosition().perp();
      }

      if(tsos.isValid())
      {
        float phi = spin(tsos.globalPosition().phi());

        // check close dets
        vector<const GeomDet *> closeDets =
          getCloseDets(il, rz ,rzBounds[il], phi,phBounds[il], tTopo);

        for(vector<const GeomDet *>::const_iterator det = closeDets.begin();
                                                    det!= closeDets.end();
                                                    det++)
        {
          TrajectoryStateOnSurface tsos =
            thePropagator->propagate(fts, (*det)->surface());

          if(tsos.isValid())
            if((*det)->surface().bounds().inside(tsos.localPosition(), le, sc))
              hasGap = false;
        }
      }
    }
  }


  if(hasGap)
    LogTrace("MinBiasTracking") << " [ValidHitPairFilter] has gap --> good";
  else
    LogTrace("MinBiasTracking") << " [ValidHitPairFilter] no gap --> rejected";

  return hasGap;
}

