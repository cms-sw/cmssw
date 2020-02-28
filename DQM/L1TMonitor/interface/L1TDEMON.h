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
// l1 dataformats, d|e record includes
#include "L1Trigger/HardwareValidation/interface/DEtrait.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class L1TDEMON : public DQMEDAnalyzer {
public:
  explicit L1TDEMON(const edm::ParameterSet&);
  ~L1TDEMON() override;

protected:
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // input d|e record
  edm::InputTag DEsource_;
  bool hasRecord_;

  // debug verbose level
  int verbose_;
  int verbose() { return verbose_; }

  // root output file name
  std::string histFile_;

  // dqm histogram folder
  std::string histFolder_;

  // dqm common
  bool monitorDaemon_;

  // running in filter farm? (use reduced set of me's)
  bool runInFF_;

  // counters
  int nEvt_;
  int deSysCount[dedefs::DEnsys];
  int nEvtWithSys[dedefs::DEnsys];

  // system status (enabled / disabled)
  // similar to COMPARE_COLLS HardwareValidation/L1Comparator, probably a more elegant solution
  // possible TODO

  bool m_doSys[dedefs::DEnsys];

  /// monitoring elements

  // global
  MonitorElement* sysrates;
  MonitorElement* sysncand[2];
  MonitorElement* errordist;
  MonitorElement* errortype[dedefs::DEnsys];

  // localization
  MonitorElement* etaphi[dedefs::DEnsys];
  MonitorElement* eta[dedefs::DEnsys];
  MonitorElement* phi[dedefs::DEnsys];
  MonitorElement* x3[dedefs::DEnsys];
  MonitorElement* etaData[dedefs::DEnsys];
  MonitorElement* phiData[dedefs::DEnsys];
  MonitorElement* x3Data[dedefs::DEnsys];
  MonitorElement* rnkData[dedefs::DEnsys];

  // trigger data word
  MonitorElement* dword[dedefs::DEnsys];
  MonitorElement* eword[dedefs::DEnsys];
  MonitorElement* deword[dedefs::DEnsys];
  MonitorElement* masked[dedefs::DEnsys];

  //define Token(-s)
  edm::EDGetTokenT<L1DataEmulRecord> DEsourceToken_;
};

#endif
