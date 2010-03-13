#ifndef DQM_RPCMonitorClient_RPCMonitorLinkSynchro_H
#define DQM_RPCMonitorClient_RPCMonitorLinkSynchro_H

/** \class RPCMonitorLinkSynchro
 ** Monitor and anlyse synchro counts () produced by R2D. 
 **/
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroStat.h"


class MonitorElement;
namespace edm { class Event; class EventSetup; class Run;}


class RPCMonitorLinkSynchro : public edm::EDAnalyzer {
public:
  explicit RPCMonitorLinkSynchro( const edm::ParameterSet& cfg);
  virtual ~RPCMonitorLinkSynchro();
  virtual void beginJob();
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock&,const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);  
  virtual void endJob();

private:
  edm::ParameterSet theConfig;
  edm::ESWatcher<RPCEMapRcd> theCablingWatcher;
  RPCLinkSynchroStat theSynchroStat;

  MonitorElement* me_delaySummary;
  MonitorElement* me_delaySpread;
  MonitorElement* me_notComplete[3];

};

#endif


