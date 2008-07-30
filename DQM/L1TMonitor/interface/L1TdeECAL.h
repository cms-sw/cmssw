#ifndef L1TdeECAL_H
#define L1TdeECAL_H

/*\class L1TdeECAL
 *\description ECAL TPG data|emulation comparison DQM interface 
               produces expert level DQM monitorable elements
 *\authors P.Paganini, N.Leonardo
 *\note et trigger tower map inspired from code in
        DQM/EcalBarrelMonitorTasks
 *\date 07.11
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

class L1TdeECAL : public edm::EDAnalyzer {

 public:

  explicit L1TdeECAL(const edm::ParameterSet&);
  ~L1TdeECAL();

 protected:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

 public:

  static const int nSM    = 36; 
  static const int nTTEta = 17; 
  static const int nTTPhi = 4;
  
 private:

  // input d|e record
  edm::InputTag DEsource_;

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
 
  // et eta-phi map, for data and emul, for individual supermodules
  std::vector<MonitorElement*> etmapData;
  std::vector<MonitorElement*> etmapEmul;
  std::vector<MonitorElement*> etmapDiff;
  MonitorElement * EcalEtMapDiff ;
  MonitorElement * EcalFGMapDiff ;

 public:

  //auxiliary converters
  int iEtaiPhiToSMid(int, int);
  int TCCidToSMid(int);

};

#endif
