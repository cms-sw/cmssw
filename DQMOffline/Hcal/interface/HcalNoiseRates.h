#ifndef _DQMOFFLINE_HCAL_HCALNOISERATES_H_
#define _DQMOFFLINE_HCAL_HCALNOISERATES_H_


//
// NoiseRates.h
//
//    description: Makes plots to calculate the anomalous noise rates
//
//    author: K. Hatakeyama, H. Liu, Baylor
//
//

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/METReco/interface/HcalNoiseRBX.h"

//Hcal Hoise Summary
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"

//
// class declaration
//

class HcalNoiseRates : public DQMEDAnalyzer {
 public:
  explicit HcalNoiseRates(const edm::ParameterSet&);
  ~HcalNoiseRates();
  
  
 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  std::string outputFile_;

  // parameters
  edm::InputTag rbxCollName_;          // label for the rbx collection
  edm::EDGetTokenT<reco::HcalNoiseRBXCollection> tok_rbx_;
  double minRBXEnergy_;                // RBX energy threshold
  double minHitEnergy_;                // RecHit energy threshold
  bool   useAllHistos_;

  //Hcal Noise Summary Parameters
  edm::EDGetTokenT<HcalNoiseSummary> noisetoken_;

  MonitorElement* hLumiBlockCount_;
  MonitorElement* hRBXEnergy_;
  MonitorElement* hRBXEnergyType1_;
  MonitorElement* hRBXEnergyType2_;
  MonitorElement* hRBXEnergyType3_;
  MonitorElement* hRBXNHits_;

  // count lumi segments
  std::map<int, int> lumiCountMap_;

  //Hcal Noise Summary Plots

  MonitorElement* nNNumChannels_;
  MonitorElement* nNSumE_;
  MonitorElement* nNSumEt_;

  MonitorElement* sNNumChannels_;
  MonitorElement* sNSumE_;
  MonitorElement* sNSumEt_;

  MonitorElement* iNNumChannels_;
  MonitorElement* iNSumE_;
  MonitorElement* iNSumEt_;

  MonitorElement* hNoise_maxZeros_;
  MonitorElement* hNoise_maxHPDHits_;
  MonitorElement* hNoise_maxHPDNoOtherHits_;

};


#endif
