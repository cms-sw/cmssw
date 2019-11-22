#ifndef DTNoiseTask_H
#define DTNoiseTask_H

/** \class DTNoiseTask
 *  No description available.
 *
 *  \authors G. Mila , G. Cerminara - INFN Torino
 */

#include <DQMServices/Core/interface/DQMOneEDAnalyzer.h>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"

#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/MuonDetId/interface/DTSuperLayerId.h>
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include <CondFormats/DTObjects/interface/DTTtrig.h>

// RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
  class Event;
}  // namespace edm

class DTGeometry;

//-class DTNoiseTask : public edm::EDAnalyzer {
class DTNoiseTask : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  /// Constructor
  DTNoiseTask(const edm::ParameterSet& ps);

  /// Destructor
  ~DTNoiseTask() override;

  // Operations
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

protected:
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& setup) final {}
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& setup) override;

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  void bookHistos(DQMStore::IBooker&, DTChamberId chId);
  void bookHistos(DQMStore::IBooker&, DTSuperLayerId slId);

  // The label to retrieve the digis
  edm::EDGetTokenT<DTDigiCollection> dtDigiToken_;
  // counter of processed events
  int evtNumber;
  //switch for time boxes filling
  bool doTimeBoxHistos;
  // Lable of 4D segments in the event
  edm::EDGetTokenT<DTRecSegment4DCollection> recHits4DToken_;
  //switch for segment veto
  bool doSegmentVeto;

  edm::ESHandle<DTGeometry> dtGeom;
  edm::ESHandle<DTTtrig> tTrigMap;

  //tTrig map per Station
  std::map<DTChamberId, double> tTrigStMap;

  //the noise histos (Hz)
  std::map<DTChamberId, MonitorElement*> noiseHistos;

  //map for histo normalization
  std::map<DTChamberId, int> mapEvt;

  //the time boxes
  std::map<DTSuperLayerId, MonitorElement*> tbHistos;

  MonitorElement* nEventMonitor;

  // safe margin (ns) between ttrig and beginning of counting area
  double safeMargin;
};
#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
