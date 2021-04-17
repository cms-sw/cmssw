// -*- C++ -*-
//
// Package:    SiStripBaselineAnalyzer
// Class:      SiStripBaselineAnalyzer
//
/**\class SiStripBaselineAnalyzer SiStripBaselineAnalyzer.cc Validation/SiStripAnalyzer/src/SiStripBaselineAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ivan Amos Cali
//         Created:  Mon Jul 28 14:10:52 CEST 2008
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

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

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

//ROOT inclusion
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TList.h"
#include "TString.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "THStack.h"

//
// class decleration
//

class SiStripBaselineAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiStripBaselineAnalyzer(const edm::ParameterSet&);
  ~SiStripBaselineAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  std::unique_ptr<SiStripPedestalsSubtractor> subtractorPed_;
  edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalsToken_;
  const SiStripPedestals* pedestalsHandle;
  edm::ESWatcher<SiStripPedestalsRcd> pedestalsWatcher_;
  std::vector<int> pedestals;

  bool plotClusters_;
  bool plotBaseline_;
  bool plotBaselinePoints_;
  bool plotRawDigi_;
  bool plotDigis_;
  bool plotAPVCM_;
  bool plotPedestals_;

  edm::InputTag srcBaseline_;
  edm::InputTag srcBaselinePoints_;
  edm::InputTag srcAPVCM_;
  edm::InputTag srcProcessedRawDigi_;
  edm::InputTag srcDigis_;
  edm::Service<TFileService> fs_;

  TH1F* h1BadAPVperEvent_;

  TH1F* h1ProcessedRawDigis_;
  TH1F* h1Baseline_;
  TH1F* h1Clusters_;
  TH1F* h1APVCM_;
  TH1F* h1Pedestals_;

  TCanvas* Canvas_;
  std::vector<TH1F> vProcessedRawDigiHisto_;
  std::vector<TH1F> vBaselineHisto_;
  std::vector<TH1F> vBaselinePointsHisto_;
  std::vector<TH1F> vClusterHisto_;

  uint16_t nModuletoDisplay_;
  uint16_t actualModule_;
};

SiStripBaselineAnalyzer::SiStripBaselineAnalyzer(const edm::ParameterSet& conf) {
  usesResource("TFileService");

  pedestalsToken_ = esConsumes();
  srcBaseline_ = conf.getParameter<edm::InputTag>("srcBaseline");
  srcBaselinePoints_ = conf.getParameter<edm::InputTag>("srcBaselinePoints");
  srcProcessedRawDigi_ = conf.getParameter<edm::InputTag>("srcProcessedRawDigi");
  srcDigis_ = conf.getParameter<edm::InputTag>("srcDigis");
  srcAPVCM_ = conf.getParameter<edm::InputTag>("srcAPVCM");
  subtractorPed_ = SiStripRawProcessingFactory::create_SubtractorPed(conf.getParameter<edm::ParameterSet>("Algorithms"),
                                                                     consumesCollector());
  nModuletoDisplay_ = conf.getParameter<uint32_t>("nModuletoDisplay");
  plotClusters_ = conf.getParameter<bool>("plotClusters");
  plotBaseline_ = conf.getParameter<bool>("plotBaseline");
  plotBaselinePoints_ = conf.getParameter<bool>("plotBaselinePoints");
  plotRawDigi_ = conf.getParameter<bool>("plotRawDigi");
  plotAPVCM_ = conf.getParameter<bool>("plotAPVCM");
  plotPedestals_ = conf.getParameter<bool>("plotPedestals");
  plotDigis_ = conf.getParameter<bool>("plotDigis");

  h1BadAPVperEvent_ = fs_->make<TH1F>("BadAPV/Event", "BadAPV/Event", 2001, -0.5, 2000.5);
  h1BadAPVperEvent_->SetXTitle("# Modules with Bad APVs");
  h1BadAPVperEvent_->SetYTitle("Entries");
  h1BadAPVperEvent_->SetLineWidth(2);
  h1BadAPVperEvent_->SetLineStyle(2);

  h1APVCM_ = fs_->make<TH1F>("APV CM", "APV CM", 2048, -1023.5, 1023.5);
  h1APVCM_->SetXTitle("APV CM [adc]");
  h1APVCM_->SetYTitle("Entries");
  h1APVCM_->SetLineWidth(2);
  h1APVCM_->SetLineStyle(2);

  h1Pedestals_ = fs_->make<TH1F>("Pedestals", "Pedestals", 2048, -1023.5, 1023.5);
  h1Pedestals_->SetXTitle("Pedestals [adc]");
  h1Pedestals_->SetYTitle("Entries");
  h1Pedestals_->SetLineWidth(2);
  h1Pedestals_->SetLineStyle(2);
}

SiStripBaselineAnalyzer::~SiStripBaselineAnalyzer() {}

void SiStripBaselineAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& es) {
  using namespace edm;
  if (plotPedestals_ && actualModule_ == 0) {
    if (pedestalsWatcher_.check(es)) {
      pedestalsHandle = &es.getData(pedestalsToken_);
    }

    std::vector<uint32_t> detIdV;
    pedestalsHandle->getDetIds(detIdV);

    for (uint32_t i = 0; i < detIdV.size(); ++i) {
      pedestals.clear();
      SiStripPedestals::Range pedestalsRange = pedestalsHandle->getRange(detIdV[i]);
      pedestals.resize((pedestalsRange.second - pedestalsRange.first) * 8 / 10);
      pedestalsHandle->allPeds(pedestals, pedestalsRange);
      for (uint32_t it = 0; it < pedestals.size(); ++it)
        h1Pedestals_->Fill(pedestals[it]);
    }
  }

  if (plotAPVCM_) {
    edm::Handle<edm::DetSetVector<SiStripProcessedRawDigi> > moduleCM;
    edm::InputTag CMLabel("siStripZeroSuppression:APVCM");
    e.getByLabel(srcAPVCM_, moduleCM);

    edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator itCMDetSetV = moduleCM->begin();
    for (; itCMDetSetV != moduleCM->end(); ++itCMDetSetV) {
      edm::DetSet<SiStripProcessedRawDigi>::const_iterator itCM = itCMDetSetV->begin();
      for (; itCM != itCMDetSetV->end(); ++itCM)
        h1APVCM_->Fill(itCM->adc());
    }
  }

  subtractorPed_->init(es);

  edm::Handle<edm::DetSetVector<SiStripRawDigi> > moduleRawDigi;
  if (plotRawDigi_)
    e.getByLabel(srcProcessedRawDigi_, moduleRawDigi);

  edm::Handle<edm::DetSetVector<SiStripProcessedRawDigi> > moduleBaseline;
  if (plotBaseline_)
    e.getByLabel(srcBaseline_, moduleBaseline);

  edm::Handle<edm::DetSetVector<SiStripDigi> > moduleBaselinePoints;
  if (plotBaselinePoints_)
    e.getByLabel(srcBaseline_, moduleBaselinePoints);

  edm::Handle<edm::DetSetVector<SiStripDigi> > moduleDigis;
  if (plotDigis_)
    e.getByLabel(srcDigis_, moduleDigis);

  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
  if (plotClusters_) {
    edm::InputTag clusLabel("siStripClusters");
    e.getByLabel(clusLabel, clusters);
  }

  char detIds[20];
  char evs[20];
  char runs[20];

  TFileDirectory sdProcessedRawDigis_ = fs_->mkdir("ProcessedRawDigis");
  TFileDirectory sdBaseline_ = fs_->mkdir("Baseline");
  TFileDirectory sdBaselinePoints_ = fs_->mkdir("BaselinePoints");
  TFileDirectory sdClusters_ = fs_->mkdir("Clusters");
  TFileDirectory sdDigis_ = fs_->mkdir("Digis");

  edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator itDSBaseline;
  if (plotBaseline_)
    itDSBaseline = moduleBaseline->begin();
  edm::DetSetVector<SiStripRawDigi>::const_iterator itRawDigis = moduleRawDigi->begin();

  uint32_t NBabAPVs = moduleRawDigi->size();
  std::cout << "Number of module with HIP in this event: " << NBabAPVs << std::endl;
  h1BadAPVperEvent_->Fill(NBabAPVs);

  for (; itRawDigis != moduleRawDigi->end(); ++itRawDigis) {
    if (actualModule_ > nModuletoDisplay_)
      return;
    uint32_t detId = itRawDigis->id;

    if (plotBaseline_) {
      //std::cout << "bas id: " << itDSBaseline->id << " raw id: " << detId << std::endl;
      if (itDSBaseline->id != detId) {
        std::cout << "Collections out of Synch. Something of fishy is going on ;-)" << std::endl;
        return;
      }
    }

    actualModule_++;
    edm::RunNumber_t const run = e.id().run();
    edm::EventNumber_t const event = e.id().event();
    //std::cout << "processing module N: " << actualModule_<< " detId: " << detId << " event: "<< event << std::endl;

    edm::DetSet<SiStripRawDigi>::const_iterator itRaw = itRawDigis->begin();
    bool restAPV[6] = {false, false, false, false, false, false};
    int strip = 0, totADC = 0;
    int minAPVRes = 7, maxAPVRes = -1;
    for (; itRaw != itRawDigis->end(); ++itRaw, ++strip) {
      float adc = itRaw->adc();
      totADC += adc;
      if (strip % 127 == 0) {
        //std::cout << "totADC " << totADC << std::endl;
        int APV = strip / 128;
        if (totADC != 0) {
          restAPV[APV] = true;
          totADC = 0;
          if (APV > maxAPVRes)
            maxAPVRes = APV;
          if (APV < minAPVRes)
            minAPVRes = APV;
        }
      }
    }

    uint16_t bins = 768;
    float minx = -0.5, maxx = 767.5;
    if (minAPVRes != 7) {
      minx = minAPVRes * 128 - 0.5;
      maxx = maxAPVRes * 128 + 127.5;
      bins = maxx - minx;
    }

    sprintf(detIds, "%ul", detId);
    sprintf(evs, "%llu", event);
    sprintf(runs, "%u", run);
    char* dHistoName = Form("Id:%s_run:%s_ev:%s", detIds, runs, evs);
    h1ProcessedRawDigis_ = sdProcessedRawDigis_.make<TH1F>(dHistoName, dHistoName, bins, minx, maxx);

    if (plotBaseline_) {
      h1Baseline_ = sdBaseline_.make<TH1F>(dHistoName, dHistoName, bins, minx, maxx);
      h1Baseline_->SetXTitle("strip#");
      h1Baseline_->SetYTitle("ADC");
      h1Baseline_->SetMaximum(1024.);
      h1Baseline_->SetMinimum(-300.);
      h1Baseline_->SetLineWidth(2);
      h1Baseline_->SetLineStyle(2);
      h1Baseline_->SetLineColor(2);
    }

    if (plotClusters_) {
      h1Clusters_ = sdClusters_.make<TH1F>(dHistoName, dHistoName, bins, minx, maxx);

      h1Clusters_->SetXTitle("strip#");
      h1Clusters_->SetYTitle("ADC");
      h1Clusters_->SetMaximum(1024.);
      h1Clusters_->SetMinimum(-300.);
      h1Clusters_->SetLineWidth(2);
      h1Clusters_->SetLineStyle(2);
      h1Clusters_->SetLineColor(3);
    }

    h1ProcessedRawDigis_->SetXTitle("strip#");
    h1ProcessedRawDigis_->SetYTitle("ADC");
    h1ProcessedRawDigis_->SetMaximum(1024.);
    h1ProcessedRawDigis_->SetMinimum(-300.);
    h1ProcessedRawDigis_->SetLineWidth(2);

    std::vector<int16_t> ProcessedRawDigis(itRawDigis->size());
    subtractorPed_->subtract(*itRawDigis, ProcessedRawDigis);

    edm::DetSet<SiStripProcessedRawDigi>::const_iterator itBaseline;
    if (plotBaseline_)
      itBaseline = itDSBaseline->begin();
    std::vector<int16_t>::const_iterator itProcessedRawDigis;

    strip = 0;
    for (itProcessedRawDigis = ProcessedRawDigis.begin(); itProcessedRawDigis != ProcessedRawDigis.end();
         ++itProcessedRawDigis) {
      if (restAPV[strip / 128]) {
        float adc = *itProcessedRawDigis;
        h1ProcessedRawDigis_->Fill(strip, adc);
        if (plotBaseline_) {
          h1Baseline_->Fill(strip, itBaseline->adc());
          ++itBaseline;
        }
      }
      ++strip;
    }

    if (plotBaseline_)
      ++itDSBaseline;
    if (plotClusters_) {
      edmNew::DetSetVector<SiStripCluster>::const_iterator itClusters = clusters->begin();
      for (; itClusters != clusters->end(); ++itClusters) {
        for (edmNew::DetSet<SiStripCluster>::const_iterator clus = itClusters->begin(); clus != itClusters->end();
             ++clus) {
          if (itClusters->id() == detId) {
            int firststrip = clus->firstStrip();
            //std::cout << "Found cluster in detId " << detId << " " << firststrip << " " << clus->amplitudes().size() << " -----------------------------------------------" << std::endl;
            strip = 0;
            for (auto itAmpl = clus->amplitudes().begin(); itAmpl != clus->amplitudes().end(); ++itAmpl) {
              h1Clusters_->Fill(firststrip + strip, *itAmpl);
              ++strip;
            }
          }
        }
      }
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void SiStripBaselineAnalyzer::beginJob() { actualModule_ = 0; }

// ------------ method called once each job just after ending the event loop  ------------
void SiStripBaselineAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripBaselineAnalyzer);
