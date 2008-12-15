#ifndef DQM_RPCMonitorModule_RPCMonitorRaw_H
#define DQM_RPCMonitorModule_RPCMonitorRaw_H
/** \class RPCMonitorRaw 
 **  Analyse errors in raw data stream as RPCRawDataCounts produced by RPCRawToDigi  
 **/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class RPCMonitorRaw : public edm::EDAnalyzer {
public:
  
  explicit RPCMonitorRaw( const edm::ParameterSet& cfg) : theConfig(cfg) {}
  virtual ~RPCMonitorRaw();

  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();

  /// get data, convert to digis attach againe to Event
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:

  edm::ParameterSet theConfig;
  
  bool theWriteHistos;
  
  MonitorElement* me_t[3];
  MonitorElement* me_e[3];
  MonitorElement* me_mapGoodEvents;
  MonitorElement* me_mapBadEvents;
};

#endif


