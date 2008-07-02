#ifndef _ChargedHadronSpectra_PlotRecHits_h_
#define _ChargedHadronSpectra_PlotRecHits_h_

#include <fstream>

namespace edm { class Event; class EventSetup; }
class SiPixelRecHit;
class SiStripRecHit2D;
class TrackerGeometry;

class PlotRecHits
{
  public:
    explicit PlotRecHits(const edm::EventSetup& es, std::ofstream& file_);
    ~PlotRecHits();
    void printRecHits(const edm::Event& ev);
    void printPixelRecHit (const SiPixelRecHit   * recHit);
    void printStripRecHit (const SiStripRecHit2D * recHit);

  private:
    void printPixelRecHits(const edm::Event& ev);
    void printStripRecHits(const edm::Event& ev);

    std::ofstream& file;
    const TrackerGeometry* theTracker;
};

#endif
