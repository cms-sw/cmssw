#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotSimTracks.h"
#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotUtils.h"
#include "QCDAnalysis/ChargedHadronSpectra/interface/HitInfo.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

using namespace std;

/*****************************************************************************/
struct sortByPabs
{
  bool operator() (const PSimHit& a, const PSimHit& b) const
  {
    return (a.pabs() > b.pabs());
  }
};

/*****************************************************************************/
struct sortByTof
{
  bool operator() (const PSimHit& a, const PSimHit& b) const
  {
    return (a.tof() < b.tof());
  }
};

/*****************************************************************************/
PlotSimTracks::PlotSimTracks
  (const edm::EventSetup& es, ofstream& file_) : file(file_)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  theTracker = trackerHandle.product();
}

/*****************************************************************************/
PlotSimTracks::~PlotSimTracks()
{
}

/*****************************************************************************/
void PlotSimTracks::printSimTracks(const edm::Event& ev)
{
  edm::Handle<TrackingParticleCollection> simTrackHandle;
  ev.getByLabel("trackingtruthprod",      simTrackHandle);
  const TrackingParticleCollection* simTracks = simTrackHandle.product();

  PlotUtils plotUtils;

  file << ", If[st, {RGBColor[0.5,0.5,0.5]";

  for(TrackingParticleCollection::const_iterator simTrack = simTracks->begin();
                                                 simTrack!= simTracks->end();
                                                 simTrack++)
  {
    vector<PSimHit> simHits;

    simHits = simTrack->trackPSimHit();

    // reorder with help of momentum
    sort(simHits.begin(), simHits.end(), sortByPabs());

    for(vector<PSimHit>::const_iterator simHit = simHits.begin();
                                        simHit!= simHits.end(); simHit++)
    {
      DetId id = DetId(simHit->detUnitId());

      GlobalPoint  p1 =
        theTracker->idToDetUnit(id)->toGlobal(simHit->localPosition()); 
      GlobalVector v1 =
        theTracker->idToDetUnit(id)->toGlobal(simHit->localDirection());

      // simHit
      file << ", Point[{" << p1.x() << "," << p1.y() << "," << p1.z() << "*z}]"
           << endl;
      file << ", Text[StyleForm[\"s\", URL->\"Ekin=" << simTrack->energy() -
simTrack->mass()
           << " GeV | parent: source=" << simTrack->parentVertex()->nSourceTracks() 
           << "daughter=" << simTrack->parentVertex()->nDaughterTracks()
           << HitInfo::getInfo(*simHit) << "\"], {"
           << p1.x() << "," << p1.y() << "," << p1.z() << "*z}, {0,1}]"
           << endl;

      // det
      double x = theTracker->idToDet(id)->surface().bounds().width() /2;
      double y = theTracker->idToDet(id)->surface().bounds().length()/2;
      double z = 0.;
  
      GlobalPoint p00 =  theTracker->idToDet(id)->toGlobal(LocalPoint(-x,-y,z));
      GlobalPoint p01 =  theTracker->idToDet(id)->toGlobal(LocalPoint(-x, y,z));
      GlobalPoint p10 =  theTracker->idToDet(id)->toGlobal(LocalPoint( x,-y,z));
      GlobalPoint p11 =  theTracker->idToDet(id)->toGlobal(LocalPoint( x, y,z));

      if(theTracker->idToDet(id)->subDetector() ==
           GeomDetEnumerators::PixelBarrel ||
         theTracker->idToDet(id)->subDetector() ==
           GeomDetEnumerators::PixelEndcap)
        file << ", If[sd, {RGBColor[0.6,0.6,0.6], ";
      else
        file << ", If[sd, {RGBColor[0.8,0.8,0.8], ";

      file       <<"Line[{{"<< p00.x()<<","<<p00.y()<<","<<p00.z()<<"*z}, "
                       <<"{"<< p01.x()<<","<<p01.y()<<","<<p01.z()<<"*z}, "
                       <<"{"<< p11.x()<<","<<p11.y()<<","<<p11.z()<<"*z}, "
                       <<"{"<< p10.x()<<","<<p10.y()<<","<<p10.z()<<"*z}, "
                       <<"{"<< p00.x()<<","<<p00.y()<<","<<p00.z()<<"*z}}]}]"
        << endl;

      if(simHit == simHits.begin()) // vertex to first point
      {
        GlobalPoint p0(simTrack->vertex().x(),
                       simTrack->vertex().y(),
                       simTrack->vertex().z());
        plotUtils.printHelix(p0,p1,v1, file, simTrack->charge());
      }

      if(simHit+1 != simHits.end()) // if not last
      {
        DetId id = DetId((simHit+1)->detUnitId());
        GlobalPoint  p2 =
          theTracker->idToDetUnit(id)->toGlobal((simHit+1)->localPosition());
        GlobalVector v2 =
          theTracker->idToDetUnit(id)->toGlobal((simHit+1)->localDirection());

        plotUtils.printHelix(p1,p2,v2, file, simTrack->charge());
      }
    }
  }

  file << "}]";
}

