#ifndef _ChargedHadronSpectra_PlotSimTracks_h_
#define _ChargedHadronSpectra_PlotSimTracks_h_

#include <fstream>

namespace edm { class Event; class EventSetup; }
class TrackingRecHit;
class TrackerGeometry;
class CaloGeometry;
class PSimHit;

class PlotSimTracks
{
  public:
    explicit PlotSimTracks(const edm::EventSetup& es, std::ofstream& file_);
    ~PlotSimTracks();
    void printSimTracks(const edm::Event& ev, const edm::EventSetup& es);

  private:
    std::ofstream& file;
    const TrackerGeometry* theTracker;
    const CaloGeometry* theCaloGeometry;
};

#endif
