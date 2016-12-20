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
#include "DataFormats/JetReco/interface/PFJet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <string>


class SUSYDQMAnalyzer: public DQMEDAnalyzer {
 public:
  explicit SUSYDQMAnalyzer(const edm::ParameterSet&);
  ~SUSYDQMAnalyzer();
 
 private:
  edm::ParameterSet iConfig;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event& , const edm::EventSetup&);

  edm::EDGetTokenT<reco::PFMETCollection> thePFMETCollectionToken;
  edm::EDGetTokenT<std::vector<reco::PFJet> > thePFJetCollectionToken;
  edm::EDGetTokenT<reco::CaloJetCollection> theCaloJetCollectionToken;

  edm::EDGetTokenT<reco::CaloMETCollection> theCaloMETCollectionToken;
  //edm::EDGetTokenT<reco::JPTJetCollection> theJPTJetCollectionToken;
  //edm::EDGetTokenT<reco::METCollection> theTCMETCollectionToken;

  double _ptThreshold;
  double _maxNJets;
  double _maxAbsEta;

  std::string SUSYFolder;
  static const char* messageLoggerCatregory;

  //Susy DQM storing elements
  //remove TCMET and JPT related variables

  MonitorElement* hCaloHT;
  //MonitorElement* hJPTHT;
  MonitorElement* hPFHT;

  MonitorElement* hCaloMET;
  MonitorElement* hPFMET;
  //MonitorElement* hTCMET;
  
  MonitorElement* hCaloMHT;
  //MonitorElement* hJPTMHT;
  MonitorElement* hPFMHT;  

  MonitorElement* hCaloAlpha_T;
  //MonitorElement* hJPTAlpha_T;
  MonitorElement* hPFAlpha_T;
  
};

#endif
