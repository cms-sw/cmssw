#ifndef DTNoiseTask_H
#define DTNoiseTask_H

/** \class DTNoiseTask
 *  No description available.
 *
 *  $Date: 2008/06/30 12:50:03 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <FWCore/Framework/interface/ESHandle.h>



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
  void beginJob(const edm::EventSetup& c);

  /// To reset the MEs
  void beginLuminosityBlock(const edm::LuminosityBlock&  lumiSeg, const edm::EventSetup& context);
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  /// Endjob
  void endJob();

private:
  
  void bookHistos(DTChamberId chId);

  // counter of processed events
  int evtNumber;
  DQMStore *dbe;
  edm::ESHandle<DTGeometry> dtGeom;

  //tTrig map per Station
  std::map<DTChamberId, int> tTrigStMap;

  //the noise histo (Hz)
  std::map<DTChamberId, MonitorElement*> noiseHistos;

  //map for histo normalization
  std::map<DTChamberId, int> mapEvt;

};
#endif

