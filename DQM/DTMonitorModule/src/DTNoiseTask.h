#ifndef DTNoiseTask_H
#define DTNoiseTask_H

/** \class DTNoiseTask
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>

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
  int nevents;
  DQMStore *dbe;
  edm::ESHandle<DTGeometry> muonGeom;
};
#endif

