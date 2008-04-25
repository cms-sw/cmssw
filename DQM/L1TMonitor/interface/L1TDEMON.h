#ifndef L1TDEMON_H
#define L1TDEMON_H

/*\class L1TDEMON
 *\description L1 trigger data|emulation comparison DQM interface 
               produces DQM monitorable elements
 *\author Nuno Leonardo (CERN)
 *\date 07.07
 */

// system, common includes
#include <memory>
#include <string>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// dqm includes
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
// l1 dataformats, d|e record includes
#include "L1Trigger/HardwareValidation/interface/DEtrait.h"

using dedefs::DEnsys;

class L1TDEMON : public edm::EDAnalyzer {

 public:

  explicit L1TDEMON(const edm::ParameterSet&);
  ~L1TDEMON();

 protected:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

 private:

  // input d|e record
  edm::InputTag DEsource_;

  // debug verbose level
  int verbose_;
  int verbose() {return verbose_;}

  // counters
  int nEvt_;
  int deSysCount[DEnsys];
  int nEvtWithSys[DEnsys];

  // root output file name
  std::string histFile_;

  // dqm histogram folder
  std::string histFolder_;

  // dqm common
  DQMStore* dbe;
  bool monitorDaemon_;
 
  /// monitoring elements

  // global
  MonitorElement* sysrates;
  MonitorElement* sysncand[2];
  MonitorElement* errordist;
  MonitorElement* errortype[DEnsys];

  // localization
  MonitorElement* etaphi[DEnsys];
  MonitorElement* eta[DEnsys];
  MonitorElement* phi[DEnsys];
  MonitorElement* x3 [DEnsys];
  MonitorElement* etaData[DEnsys];
  MonitorElement* phiData[DEnsys];
  MonitorElement*  x3Data[DEnsys];
  MonitorElement* rnkData[DEnsys];

  // trigger data word
  MonitorElement* dword [DEnsys];
  MonitorElement* eword [DEnsys];
  MonitorElement* deword[DEnsys];
  MonitorElement* masked[DEnsys];

  // subsytem correlations
  MonitorElement* CORR[DEnsys][DEnsys][3];
};

#endif
