#ifndef CentralityDQM_H
#define CentralityDQM_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h" 

#include "DQMServices/Core/interface/MonitorElement.h"

class DQMStore;
 
class CentralityDQM: public edm::EDAnalyzer{

public:

  CentralityDQM(const edm::ParameterSet& ps);
  virtual ~CentralityDQM();
  
protected:

  virtual void beginJob();
  virtual void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;
  virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  virtual void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void endJob();
  
private:

  void bookHistos(DQMStore * bei );
  
  int nLumiSecs_;
  
  DQMStore* bei_;  

  ///////////////////////////
  // Histograms
  ///////////////////////////

  // Histograms - Centrality
  MonitorElement* h_hiNpix;
  MonitorElement* h_hiNpixelTracks;
  MonitorElement* h_hiNtracks;
  MonitorElement* h_hiNtracksPtCut;
  MonitorElement* h_hiNtracksEtaCut;
  MonitorElement* h_hiNtracksEtaPtCut;
  MonitorElement* h_hiHF;
  MonitorElement* h_hiHFplus;
  MonitorElement* h_hiHFminus;
  MonitorElement* h_hiHFplusEta4;
  MonitorElement* h_hiHFminusEta4;
  MonitorElement* h_hiHFhit;
  MonitorElement* h_hiHFhitPlus;
  MonitorElement* h_hiHFhitMinus;
  MonitorElement* h_hiEB;
  MonitorElement* h_hiET;
  MonitorElement* h_hiEE;
  MonitorElement* h_hiEEplus;
  MonitorElement* h_hiEEminus;
  MonitorElement* h_hiZDC;
  MonitorElement* h_hiZDCplus;
  MonitorElement* h_hiZDCminus;

};


#endif
