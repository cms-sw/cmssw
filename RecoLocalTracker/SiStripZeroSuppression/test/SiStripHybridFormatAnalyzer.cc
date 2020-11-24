// -*- C++ -*-
//
// Package:    SiStripHybridFormatAnalyzer
// Class:      SiStripHybridFormatAnalyzer
//
/**\class SiStripHybridFormatAnalyzer SiStripHybridFormatAnalyzer.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ivan Amos Cali
//         Created:  March 20 2018
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

//ROOT inclusion
#include "TH1F.h"

//
// class decleration
//

class SiStripHybridFormatAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  explicit SiStripHybridFormatAnalyzer(const edm::ParameterSet&);
  ~SiStripHybridFormatAnalyzer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override;

  edm::EDGetTokenT<edm::DetSetVector<SiStripDigi>> srcDigis_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripProcessedRawDigi>> srcAPVCM_;
  edm::Service<TFileService> fs_;
  edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalsToken_;

  TH1F* h1Digis_;
  TH1F* h1APVCM_;
  TH1F* h1BadAPVperEvent_;
  TH1F* h1BadAPVperModule_;
  TH1F* h1BadAPVperModuleOnlyBadModule_;
  TH1F* h1Pedestals_;

  TFileDirectory sdDigis_;
  TFileDirectory sdMisc_;

  uint16_t nModuletoDisplay_;
  uint16_t actualModule_;

  bool plotAPVCM_;

  //this to plot the pedestals distribution
  uint32_t peds_cache_id_;
};

SiStripHybridFormatAnalyzer::SiStripHybridFormatAnalyzer(const edm::ParameterSet& conf) {
  usesResource(TFileService::kSharedResource);

  srcDigis_ = consumes<edm::DetSetVector<SiStripDigi>>(conf.getParameter<edm::InputTag>("srcDigis"));
  srcAPVCM_ = consumes<edm::DetSetVector<SiStripProcessedRawDigi>>(conf.getParameter<edm::InputTag>("srcAPVCM"));
  pedestalsToken_ = esConsumes();
  nModuletoDisplay_ = conf.getParameter<uint32_t>("nModuletoDisplay");
  plotAPVCM_ = conf.getParameter<bool>("plotAPVCM");

  sdDigis_ = fs_->mkdir("Digis");
  sdMisc_ = fs_->mkdir("Miscellanea");

  h1APVCM_ = sdMisc_.make<TH1F>("APV CM", "APV CM", 1601, -100.5, 1500.5);
  h1APVCM_->SetXTitle("APV CM [adc]");
  h1APVCM_->SetYTitle("Entries");
  h1APVCM_->SetLineWidth(2);
  h1APVCM_->SetLineStyle(2);

  h1BadAPVperEvent_ = sdMisc_.make<TH1F>("BadAPV/Event", "BadAPV/Event", 72786, -0.5, 72785.5);
  h1BadAPVperEvent_->SetXTitle("# Bad APVs");
  h1BadAPVperEvent_->SetYTitle("Entries");
  h1BadAPVperEvent_->SetLineWidth(2);
  h1BadAPVperEvent_->SetLineStyle(2);

  h1BadAPVperModule_ = sdMisc_.make<TH1F>("BadAPV/Module", "BadAPV/Module", 7, -0.5, 6.5);
  h1BadAPVperModule_->SetXTitle("# Bad APVs");
  h1BadAPVperModule_->SetYTitle("Entries");
  h1BadAPVperModule_->SetLineWidth(2);
  h1BadAPVperModule_->SetLineStyle(2);

  h1BadAPVperModuleOnlyBadModule_ =
      sdMisc_.make<TH1F>("BadAPV/Module Only Bad Modules", "BadAPV/Module Only Bad Modules", 7, -0.5, 6.5);
  h1BadAPVperModuleOnlyBadModule_->SetXTitle("# Bad APVs");
  h1BadAPVperModuleOnlyBadModule_->SetYTitle("Entries");
  h1BadAPVperModuleOnlyBadModule_->SetLineWidth(2);
  h1BadAPVperModuleOnlyBadModule_->SetLineStyle(2);

  h1Pedestals_ = sdMisc_.make<TH1F>("Pedestals", "Pedestals", 2048, -1023.5, 1023.5);
  h1Pedestals_->SetXTitle("Pedestals [adc]");
  h1Pedestals_->SetYTitle("Entries");
  h1Pedestals_->SetLineWidth(2);
  h1Pedestals_->SetLineStyle(2);
}

SiStripHybridFormatAnalyzer::~SiStripHybridFormatAnalyzer() {}

void SiStripHybridFormatAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcDigis", edm::InputTag("siStripZeroSuppression", "VirginRaw"));
  desc.add<edm::InputTag>("srcAPVCM", edm::InputTag("siStripZeroSuppression", "APVCMVirginRaw"));
  desc.add<uint32_t>("nModuletoDisplay", 10000);
  desc.add<bool>("plotAPVCM", true);
  descriptions.add("siStripHybridFormatAnalyzer", desc);
}

void SiStripHybridFormatAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& es) {
  using namespace edm;

  //plotting pedestals
  //------------------------------------------------------------------
  if (actualModule_ == 0) {
    const auto& pedestalsObj = es.getData(pedestalsToken_);
    std::vector<uint32_t> detIdV;
    pedestalsObj.getDetIds(detIdV);
    std::vector<int> pedestals;
    for (const auto det : detIdV) {
      pedestals.clear();
      SiStripPedestals::Range pedestalsRange = pedestalsObj.getRange(det);
      pedestals.resize((pedestalsRange.second - pedestalsRange.first) * 0.8);
      pedestalsObj.allPeds(pedestals, pedestalsRange);
      for (const int ped : pedestals) {
        h1Pedestals_->Fill(ped);
      }
    }
  }

  //plotting CMN
  //------------------------------------------------------------------

  if (plotAPVCM_) {
    edm::Handle<edm::DetSetVector<SiStripProcessedRawDigi>> moduleCM;
    e.getByToken(srcAPVCM_, moduleCM);

    for (const auto& set : *moduleCM) {
      for (const auto& itCM : set)
        h1APVCM_->Fill(itCM.adc());
    }
  }

  //plotting digis histograms
  //------------------------------------------------------------------
  uint32_t nBadAPVevent = 0;

  edm::Handle<edm::DetSetVector<SiStripDigi>> moduleDigis;
  e.getByToken(srcDigis_, moduleDigis);

  edm::DetSetVector<SiStripDigi>::const_iterator itDigiDetSetV = moduleDigis->begin();
  for (; itDigiDetSetV != moduleDigis->end(); ++itDigiDetSetV) {
    uint32_t detId = itDigiDetSetV->id;
    edm::RunNumber_t const run = e.id().run();
    edm::EventNumber_t const event = e.id().event();

    char detIds[20];
    char evs[20];
    char runs[20];

    if (actualModule_ < nModuletoDisplay_) {
      sprintf(detIds, "%ul", detId);
      sprintf(evs, "%llu", event);
      sprintf(runs, "%u", run);
      char* dHistoName = Form("Id_%s_run_%s_ev_%s", detIds, runs, evs);
      h1Digis_ = sdDigis_.make<TH1F>(dHistoName, dHistoName, 768, -0.5, 767.5);
      h1Digis_->SetXTitle("strip #");
      h1Digis_->SetYTitle("adc");
      h1Digis_->SetLineWidth(2);
      h1Digis_->SetLineStyle(2);
    }
    uint16_t stripsPerAPV[6] = {0, 0, 0, 0, 0, 0};
    for (const auto& itDigi : *itDigiDetSetV) {
      uint16_t strip = itDigi.strip();
      uint16_t adc = itDigi.adc();
      if (actualModule_ < nModuletoDisplay_)
        h1Digis_->Fill(strip, adc);
      actualModule_++;
      //std::cout << "detID " << detId << " strip " << strip << " adc " << adc << std::endl;

      stripsPerAPV[strip / 128]++;
    }

    uint16_t nBadAPVmodule = 0;
    for (uint16_t apvN = 0; apvN < 6; apvN++) {
      if (stripsPerAPV[apvN] > 64) {
        nBadAPVevent++;
        nBadAPVmodule++;
      }
    }
    h1BadAPVperModule_->Fill(nBadAPVmodule);
    if (nBadAPVmodule)
      h1BadAPVperModuleOnlyBadModule_->Fill(nBadAPVmodule);
  }
  h1BadAPVperEvent_->Fill(nBadAPVevent);
}

void SiStripHybridFormatAnalyzer::beginRun(edm::Run const& iEvent, edm::EventSetup const&) { actualModule_ = 0; }

void SiStripHybridFormatAnalyzer::endRun(edm::Run const& iEvent, edm::EventSetup const&) {}

// ------------ method called once each job just before starting event loop  ------------
void SiStripHybridFormatAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiStripHybridFormatAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripHybridFormatAnalyzer);
