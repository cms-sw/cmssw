#ifndef DQM_RPCMonitorClient_RPCMonitorLinkSynchro_H
#define DQM_RPCMonitorClient_RPCMonitorLinkSynchro_H

/** \class RPCMonitorLinkSynchro
 ** Monitor and anlyse synchro counts () produced by R2D. 
 **/
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"


class MonitorElement;
class RPCReadOutMapping;
namespace edm { class Event; class EventSetup; }


class RPCMonitorLinkSynchro : public edm::EDAnalyzer {
public:
  explicit RPCMonitorLinkSynchro( const edm::ParameterSet& cfg);
  virtual ~RPCMonitorLinkSynchro();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock&,const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);  
  virtual void endJob();

private:
  edm::ParameterSet theConfig;

  edm::ESWatcher<RPCEMapRcd> theCablingWatcher;
  const RPCReadOutMapping* theCabling;
  RPCRawSynchro theSynchro; 

  MonitorElement* me_delaySummary;
  MonitorElement* me_delaySpread;
//  MonitorElement* me_linksLowStat;
//  MonitorElement* me_linksBadSynchro;
//  MonitorElement* me_linksMostNoisy;
};

#endif


