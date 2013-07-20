#ifndef DTNoiseTask_H
#define DTNoiseTask_H

/** \class DTNoiseTask
 *  No description available.
 *
 *  $Date: 2011/06/10 13:23:26 $
 *  $Revision: 1.9 $
 *  \authors G. Mila , G. Cerminara - INFN Torino
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/MuonDetId/interface/DTSuperLayerId.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Utilities/interface/InputTag.h"


namespace edm {
  class ParameterSet;
  class EventSetup;
  class Event;
}

class DQMStore;
class DTGeometry;


class DTNoiseTask : public edm::EDAnalyzer {
public:
  /// Constructor
  DTNoiseTask(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTNoiseTask();

  // Operations

protected:
  /// BeginJob
  void beginJob();

  void beginRun(const edm::Run&, const edm::EventSetup&);

  void beginLuminosityBlock(const edm::LuminosityBlock&  lumiSeg, const edm::EventSetup& context);
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& setup);
  

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  /// Endjob
  void endJob();

private:
  
  void bookHistos(DTChamberId chId);
  void bookHistos(DTSuperLayerId slId);

  // The label to retrieve the digis 
  edm::InputTag dtDigiLabel;
  // counter of processed events
  int evtNumber;
  //switch for time boxes filling
  bool doTimeBoxHistos;
  // Lable of 4D segments in the event
  std::string theRecHits4DLabel;
  //switch for segment veto
  bool doSegmentVeto;

  DQMStore *dbe;
  edm::ESHandle<DTGeometry> dtGeom;

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

