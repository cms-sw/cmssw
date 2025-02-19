//authors:  Francesco Costanza (DESY)
//          Dirk Kruecker (DESY)
//date:     05/05/11

#ifndef DQMOFFLINE_JETMET_SUSYDQM_ANALYZER_H
#define DQMOFFLINE_JETMET_SUSYDQM_ANALYZER_H (1)

#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class SUSYDQMAnalyzer: public edm::EDAnalyzer {
 public:
  explicit SUSYDQMAnalyzer(const edm::ParameterSet&);
  ~SUSYDQMAnalyzer();
 
 private:
  edm::ParameterSet iConfig;

  virtual void beginJob();
  virtual void beginRun(const edm::Run&, const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& , const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  edm::InputTag theCaloMETCollectionLabel;
  edm::InputTag thePFMETCollectionLabel;
  edm::InputTag theTCMETCollectionLabel;

  edm::InputTag theCaloJetCollectionLabel;
  edm::InputTag thePFJetCollectionLabel;
  edm::InputTag theJPTJetCollectionLabel;

  double _ptThreshold;
  double _maxNJets;
  double _maxAbsEta;

  std::string SUSYFolder;
  static const char* messageLoggerCatregory;
  
  DQMStore* dqm;

  //Susy DQM storing elements

  MonitorElement* hCaloHT;
  MonitorElement* hJPTHT;
  MonitorElement* hPFHT;

  MonitorElement* hCaloMET;
  MonitorElement* hPFMET;
  MonitorElement* hTCMET;
  
  MonitorElement* hCaloMHT;
  MonitorElement* hJPTMHT;
  MonitorElement* hPFMHT;  

  MonitorElement* hCaloAlpha_T;
  MonitorElement* hJPTAlpha_T;
  MonitorElement* hPFAlpha_T;
  
};

#endif
