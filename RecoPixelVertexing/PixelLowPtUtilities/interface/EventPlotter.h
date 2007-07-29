#ifndef _EventPlotter_h_
#define _EventPlotter_h_

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"


#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/VZero/interface/VZero.h"
#include "DataFormats/VZero/interface/VZeroFwd.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include <fstream>

#include <vector>
using namespace std;

class EventPlotter
{
  public:
    EventPlotter(const edm::EventSetup& es);
    ~EventPlotter();
    void printEvent(const edm::Event& ev);

  private:
    void printHelix(const GlobalPoint& p1, const GlobalPoint& p2, const
GlobalVector& n2, ofstream& outFile, int charge);
    void getGlobal(const PSimHit& simHit, GlobalPoint& p, GlobalVector& v, bool& isPixel);
    void printSimTracks(const TrackingParticleCollection* simTracks);
    void printRecTracks(const reco::TrackCollection* recTracks,
                        const vector<Trajectory>*    recTrajes);
    void printPixelRecHit
      (const SiPixelRecHit * recHit, ofstream& pixelDetUnits,
                                     ofstream& pixelHits);
    void printPixelRecHits(const edm::Event& ev);

    void printStripRecHit
      (const SiStripRecHit2D * recHit, ofstream& stripDetUnits,
                                       ofstream& stripHits);
    void printStripRecHits(const edm::Event& ev);

    FreeTrajectoryState getTrajectory(const reco::Track& track);
    void printVZeros(const reco::VZeroCollection* vZeros);
    void printVertices(const reco::VertexCollection* vertices);

    const TrackerGeometry* theTracker;
    const MagneticField* theMagField;
};

#endif
