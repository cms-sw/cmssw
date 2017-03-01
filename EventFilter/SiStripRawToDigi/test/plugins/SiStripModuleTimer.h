#ifndef EventFilter_SiStripRawToDigi_SiStripModuleTimer_H
#define EventFilter_SiStripRawToDigi_SiStripModuleTimer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "DQM/HLTEvF/interface/PathTimerService.h"
#include "TFile.h"
#include "TTree.h"

class SiStripModuleTimer : public edm::EDAnalyzer {

  public:

  SiStripModuleTimer(const edm::ParameterSet&);
  ~SiStripModuleTimer();

  void beginJob();
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();

 private:

  std::vector< std::string > moduleLabels_;
  std::vector<double> times_;
  TFile* file_;
  TTree* tree_;
};

#endif
