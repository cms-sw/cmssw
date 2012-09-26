#ifndef DQM_RPCMonitorModule_RPCMonitorRaw_H
#define DQM_RPCMonitorModule_RPCMonitorRaw_H
/** \class RPCMonitorRaw 
 **  Analyse errors in raw data stream as RPCRawDataCounts produced by RPCRawToDigi  
 **/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <bitset>

class RPCMonitorRaw : public edm::EDAnalyzer {
public:
  
  explicit RPCMonitorRaw( const edm::ParameterSet& cfg);
  virtual ~RPCMonitorRaw();

  virtual void beginJob( );
  virtual void endJob();

  /// get data, convert to digis attach againe to Event
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:

  
  MonitorElement* me_t[3];
  MonitorElement* me_e[3];
  MonitorElement* me_mapGoodEvents;
  MonitorElement* me_mapBadEvents;

  edm::ParameterSet theConfig;
  std::vector<MonitorElement* > theWatchedErrorHistos[3]; // histos with physical locations 
                                                          // (RMB,LINK)of selected  ReadoutErrors

  unsigned int theWatchedErrorHistoPos[10];               // for a give error type show its position
                                                          // (1..10) in theWatchedErrorHistos
                                                          // to get index one has to subtract -1
                                                          // 0 is not selected error type 
                                                  
  edm::InputTag rpcRawDataCountsTag_;
};

#endif
