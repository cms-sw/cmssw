#ifndef BxTiming_H
#define BxTiming_H

/*\class BxTiming
 *\description common FED timing DQM module
 *\author N.Leonardo, A.Holzner, T.Christiansen, I.Mikulec
 *\date 08.03
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


class BxTiming : public edm::EDAnalyzer {

 public:

  explicit BxTiming(const edm::ParameterSet&);
  ~BxTiming();

 protected:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

 private:

  // input
  edm::InputTag fedSource_;
  edm::InputTag gtSource_;

  // debug verbose level
  int verbose_;
  int verbose() {return verbose_;}

  // counters
  int nEvt_;

  // root output file name
  std::string histFile_;

  // dqm histogram folder
  std::string histFolder_;

  // dqm common
  DQMStore* dbe;
 
  // readout l1 systems
  enum nsys {NSYS=9}; 
  enum syslist {ETP=0, HTP, GCT, CTP, CTF, DTP, DTF, RPC, GLT};
  std::pair<int,int> fedRange_[NSYS];
  int nfed_;   // number of feds
  int fedRef_; // reference fed

  /// histograms
  MonitorElement* hBxDiffAllFed;
  MonitorElement* hBxDiffSysFed[NSYS];
  MonitorElement* hBxOccyAllFed;
  MonitorElement**hBxOccyOneFed;

};

#endif
