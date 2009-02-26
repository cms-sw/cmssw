#ifndef _ChargedHadronSpectra_PlotEcalRecHits_h_
#define _ChargedHadronSpectra_PlotEcalRecHits_h_

#include <fstream>

namespace edm { class Event; class EventSetup; }
class CaloCellGeometry;
class CaloGeometry;

class PlotEcalRecHits
{
  public:
    explicit PlotEcalRecHits(const edm::EventSetup & es, std::ofstream & file_);
    ~PlotEcalRecHits();

    void printEcalRecHits(const edm::Event& ev);

  private:
    void printEcalRecHit (const CaloCellGeometry* cell, float energy);

    std::ofstream & file;
    const CaloGeometry * theCaloGeometry;
};

#endif
