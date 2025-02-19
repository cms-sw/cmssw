#ifndef L1TdeGCT_H
#define L1TdeGCT_H

/*\class L1TdeGCT
 *\description GCT data|emulation comparison DQM interface 
               produces expert level DQM monitorable elements
 *\author N.Leonardo
 *\date 08.09
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

class L1TdeGCT : public edm::EDAnalyzer {

 public:

  explicit L1TdeGCT(const edm::ParameterSet&);
  ~L1TdeGCT();

 protected:

  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

 private:

  // input d|e record
  edm::InputTag DEsource_;
  bool hasRecord_;

  // debug verbose level
  int verbose_;
  int verbose() {return verbose_;}

  // root output file name
  std::string histFile_;

 // dqm histogram folder
  std::string histFolder_;

  // dqm common
  DQMStore* dbe;
  bool monitorDaemon_;
 
  // (em) iso, no-iso, (jets) cen, for, tau & energy sums.
  static const int nGctColl_ = dedefs::GCThfbit-dedefs::GCTisolaem+1; 

  // counters
  int colCount[nGctColl_];
  int nWithCol[nGctColl_];

  // MEs
  MonitorElement* sysrates;
  MonitorElement* sysncand[2];
  MonitorElement* errortype[nGctColl_];
  // location
  MonitorElement* etaphi [nGctColl_];
  MonitorElement* eta    [nGctColl_];
  MonitorElement* phi    [nGctColl_];
  MonitorElement* rnk    [nGctColl_];
  MonitorElement* etaData[nGctColl_];
  MonitorElement* phiData[nGctColl_];
  MonitorElement* rnkData[nGctColl_];

  // trigger data word
  MonitorElement* dword [nGctColl_];
  MonitorElement* eword [nGctColl_];
  MonitorElement* deword[nGctColl_];
  MonitorElement* masked[nGctColl_];

 public:

};

#endif
