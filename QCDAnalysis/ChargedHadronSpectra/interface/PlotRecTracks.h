#ifndef _ChargedHadronSpectra_PlotRecTracks_h_
#define _ChargedHadronSpectra_PlotRecTracks_h_

#include <fstream>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"

namespace edm { class Event; class EventSetup; }
class TrackingRecHit;
class TrackerGeometry;
class TrackerHitAssociator;
class TrackerTopology;
class FreeTrajectoryState;
class MagneticField;
class Propagator;

class TrajectoryFitter;

namespace reco { class Track; }

class PlotRecTracks
{
  public:
    explicit PlotRecTracks(const edm::EventSetup& es_,
                           std::string trackProducer_, std::ofstream& file_);
    ~PlotRecTracks();
    void printRecTracks(const edm::Event& ev, const edm::EventSetup& es);

  private:
    std::string getPixelInfo(const TrackingRecHit* recHit,
                             const TrackerTopology* tTopo,
                             const std::ostringstream& o,
                             const std::ostringstream& d);
    std::string getStripInfo(const TrackingRecHit* recHit,
                             const TrackerTopology* tTopo,
                             const std::ostringstream& o,
                             const std::ostringstream& d);
    FreeTrajectoryState getTrajectoryAtOuterPoint(const reco::Track& track);

    const edm::EventSetup& es;
    std::string trackProducer;
    std::ofstream& file;

    const TrackerGeometry* theTracker;
    const MagneticField* theMagField;
    const Propagator*  thePropagator;
    const TrajectoryFitter* theFitter;

    TrackerHitAssociator * theHitAssociator;
};

#endif
