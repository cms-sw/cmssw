#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiPixelPerformanceSummaryRcd.h"

#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoReader.h"

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <sys/time.h>


using namespace cms;


SiPixelHistoricInfoReader::SiPixelHistoricInfoReader(const edm::ParameterSet& iConfig) 
			 : printDebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)), 
                           outputDir_(iConfig.getUntrackedParameter<std::string>("outputDir",".")), 
			   presentRun_(0) {}


SiPixelHistoricInfoReader::~SiPixelHistoricInfoReader() {}


void SiPixelHistoricInfoReader::beginJob(const edm::EventSetup& iSetup) {
  TString outputFilename(outputDir_); 
  outputFilename += "/SiPixelHistoricInfoReader.root"; 
  outputFile = new TFile(outputFilename,"RECREATE");

  allDetIds.clear(); // allDetIds.push_back(369345800);
  edm::ESHandle<SiPixelPerformanceSummary> SiPixelPerformanceSummary_;
  iSetup.get<SiPixelPerformanceSummaryRcd>().get(SiPixelPerformanceSummary_);
  SiPixelPerformanceSummary_->getAllDetIds(allDetIds);

  AllDetHistograms = new TObjArray();
  for (std::vector<uint32_t>::const_iterator iDet = allDetIds.begin(); iDet!=allDetIds.end(); ++iDet) {
    TString hisID("NumberOfDigis_DetID"); hisID += *iDet; hisID += "_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);
  }
}


void SiPixelHistoricInfoReader::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  edm::ESHandle<SiPixelPerformanceSummary> SiPixelPerformanceSummary_;
  iSetup.get<SiPixelPerformanceSummaryRcd>().get(SiPixelPerformanceSummary_);

  if (presentRun_!=SiPixelPerformanceSummary_->getRunNumber()) { 
    presentRun_ = SiPixelPerformanceSummary_->getRunNumber();
    
    TString hisID("NumberOfDigis_AllDets_Run"); hisID += presentRun_; 
    // std::ostringstream oRunID; oRunID << run.id(); hisID += oRunID.str(); was using RunID. 
    NumberOfDigisAllDets = new TH1F(hisID, hisID, 1, 0, 1); 
    NumberOfDigisAllDets->SetBit(TH1::kCanRebin);
 
    SiPixelPerformanceSummary_->print();
    fillHistograms(SiPixelPerformanceSummary_);
  }
}


void SiPixelHistoricInfoReader::analyze(const edm::Event& event, const edm::EventSetup& iSetup) {}


void SiPixelHistoricInfoReader::fillHistograms(edm::ESHandle<SiPixelPerformanceSummary> spps) {  
  for (std::vector<uint32_t>::const_iterator iDet=allDetIds.begin(); 
       iDet!=allDetIds.end(); ++iDet) {
    std::vector<float> performances; 
    performances.clear();
    spps->getDetSummary(*iDet, performances);
    
    if (performances.size()==3) {
      TString sDet; sDet += *iDet; 
      NumberOfDigisAllDets->Fill(sDet, performances[0]);
      int jBin = NumberOfDigisAllDets->GetXaxis()->FindBin(sDet);
      NumberOfDigisAllDets->SetBinError(jBin, performances[1]);

      TString sRun; sRun += presentRun_;
      TString hisID("NumberOfDigis_DetID"); hisID += *iDet; hisID += "_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[0]);
      int iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[1]);
    }
    else edm::LogError("WrongSize") << "sipixel performance summary size = "<< performances.size();
  }
  NumberOfDigisAllDets->LabelsDeflate("X");
}


void SiPixelHistoricInfoReader::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}


void SiPixelHistoricInfoReader::endJob() {
  for (std::vector<uint32_t>::const_iterator iDet=allDetIds.begin(); iDet!=allDetIds.end(); ++iDet) {
    TString hisID("NumberOfDigis_DetID"); hisID += *iDet; hisID += "_AllRuns";	     
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  }
  outputFile->Write();
  outputFile->Close();
}
