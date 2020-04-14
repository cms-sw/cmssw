#ifndef DTOccupancyEfficiency_H
#define DTOccupancyEfficiency_H

/** \class DTOccupancyEfficiency
 *  DQM Analysis of 4D DT segments, DTdigis and DTRecHits : <br>
 *   
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <string>
#include <map>
#include <vector>

class DTOccupancyEfficiency : public DQMEDAnalyzer {
public:
  /// Constructor
  DTOccupancyEfficiency(const edm::ParameterSet& pset);

  /// Destructor
  ~DTOccupancyEfficiency() override;

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override;

protected:
  // Book the histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  // Switch for verbosity
  bool debug;

  // Label for Digis
  edm::EDGetTokenT<DTDigiCollection> dtDigiToken_;

  // Label of 4D segments in the event
  edm::EDGetTokenT<DTRecSegment4DCollection> recHits4DToken_;

  // Lable of 1D rechits in the event
  edm::EDGetTokenT<DTRecHitCollection> recHitToken_;

  edm::ParameterSet parameters;

  MonitorElement* timeBoxesPerEvent;
  MonitorElement* digisPerEvent;
  MonitorElement* segments4DPerEvent;
  MonitorElement* recHitsPerEvent;
  MonitorElement* recHitsPer4DSegment;
  MonitorElement* t0From4DPhiSegment;
  MonitorElement* t0From4DZSegment;
  // station, wheel for ints
  std::map<int, std::map<int, MonitorElement*> > timeBoxesPerRing;
  std::map<int, std::map<int, MonitorElement*> > digisPerRing;
};
#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
