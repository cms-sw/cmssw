#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiPixelPerformanceSummaryRcd.h"

#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoReader.h"

#include <math.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <sys/time.h>


using namespace cms;
using namespace std;


SiPixelHistoricInfoReader::SiPixelHistoricInfoReader(const edm::ParameterSet& iConfig) 
			 : printDebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)), 
                           outputDir_(iConfig.getUntrackedParameter<std::string>("outputDir",".")), 
			   presentRun_(0) {}


SiPixelHistoricInfoReader::~SiPixelHistoricInfoReader() {}


void SiPixelHistoricInfoReader::beginJob(const edm::EventSetup& iSetup) {
  TString outputFilename(outputDir_); 
  outputFilename+="/SiPixelHistoricInfoReader.root"; 
  outputFile = new TFile(outputFilename, "RECREATE");

  allDetIds.clear(); // allDetIds.push_back(369345800);
  edm::ESHandle<SiPixelPerformanceSummary> pSummary;
  iSetup.get<SiPixelPerformanceSummaryRcd>().get(pSummary);
  pSummary->getAllDetIds(allDetIds);
    
  AllDetHistograms = new TObjArray();

  for (std::vector<uint32_t>::const_iterator iDet=allDetIds.begin(); iDet!=allDetIds.end(); ++iDet) {    
    hisID="NumberOfRawDataErrors_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    for (int pBin=0; pBin<14; pBin++) {
      hisID="RawDataErrorType"; hisID+=(pBin+25); hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
    } 
    for (int pBin=0; pBin<5; pBin++) {
      hisID="TBMType"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
    }
    for (int pBin=0; pBin<8; pBin++) {
      hisID="TBMMessage"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
    }
    for (int pBin=0; pBin<7; pBin++) {
      hisID="FEDfullType"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
    }
    for (int pBin=0; pBin<37; pBin++) {
      hisID="FEDtimeoutChannel"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
    } 
    hisID="SLinkErrSize_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="FEDmaxErrLink_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
    ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

    hisID="maxErr36ROC_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
    ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

    hisID="maxErrDCol_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
    ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

    hisID="maxErrPixelRow_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
    ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

    hisID="maxErr38ROC_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
    ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

    hisID="NumberOfDigis_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="ADC_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="DigimapHot_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="DigimapCold_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="NumberOfClusters_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="ClusterCharge_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="ClusterSizeX_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="ClusterSizeY_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="ClustermapHot_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="ClustermapCold_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="NumberOfRecHits_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="RecHitMatchedClusterSizeX_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="RecHitMatchedClusterSizeY_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="RecHitmapHot_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="RecHitmapCold_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="ResidualX_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    hisID="ResidualY_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

    cout << "booking histograms for DetID" << (*iDet) <<" finished" << endl; 
  }
  hisID="NumberOfRawDataErrors_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  for (int pBin=0; pBin<14; pBin++) {
    hisID="RawDataErrorType"; hisID+=(pBin+25); hisID+="_AllDets_AllRuns";	   
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
  } 
  for (int pBin=0; pBin<5; pBin++) {
    hisID="TBMType"; hisID+=pBin; hisID+="_AllDets_AllRuns";	   
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
  }
  for (int pBin=0; pBin<8; pBin++) {
    hisID="TBMMessage"; hisID+=pBin; hisID+="_AllDets_AllRuns";	   
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
  }
  for (int pBin=0; pBin<7; pBin++) {
    hisID="FEDfullType"; hisID+=pBin; hisID+="_AllDets_AllRuns";	   
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
  }
  for (int pBin=0; pBin<37; pBin++) {
    hisID="FEDtimeoutChannel"; hisID+=pBin; hisID+="_AllDets_AllRuns"; 	   
    AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
    ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
  } 
  hisID="SLinkErrSize_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="FEDmaxErrLink_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
  ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

  hisID="maxErr36ROC_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
  ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

  hisID="maxErrDCol_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
  ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

  hisID="maxErrPixelRow_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
  ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

  hisID="maxErr38ROC_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH2F(hisID, hisID, 1, 0, 1, 1, 0, 1));
  ((TH2F*) AllDetHistograms->FindObject(hisID))->SetBit(TH2::kCanRebin);

  hisID="NumberOfDigis_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="ADC_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="DigimapHot_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="DigimapCold_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="NumberOfClusters_AllDets_AllRuns";  	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="ClusterCharge_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="ClusterSizeX_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="ClusterSizeY_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="ClustermapHot_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="ClustermapCold_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="NumberOfRecHits_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="RecHitMatchedClusterSizeX_AllDets_AllRuns"; 	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="RecHitMatchedClusterSizeY_AllDets_AllRuns"; 	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="RecHitmapHot_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="RecHitmapCold_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="ResidualX_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  hisID="ResidualY_AllDets_AllRuns";	   
  AllDetHistograms->Add(new TH1F(hisID, hisID, 1, 0, 1));
  ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);

  cout << "booking all histograms finished" << endl; 
}


void SiPixelHistoricInfoReader::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  // it tries at every beginRun & gets a new SiPixelPerformanceSummary when the run number is right
  edm::ESHandle<SiPixelPerformanceSummary> pSummary;
  iSetup.get<SiPixelPerformanceSummaryRcd>().get(pSummary); 

  if (presentRun_!=pSummary->getRunNumber()) { 
    pSummary->print(); 
    presentRun_ = pSummary->getRunNumber();
    TString sRun; sRun+=presentRun_; int iBin=0; float exErr=0.0, error=0.0; 

    float numberOfEvents = float(pSummary->getNumberOfEvents());

    for (std::vector<uint32_t>::const_iterator iDet=allDetIds.begin(); iDet!=allDetIds.end(); ++iDet) {
      std::vector<float> performances; 
      performances.clear();
      pSummary->getDetSummary(*iDet, performances);
      
      hisID="NumberOfRawDataErrors_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[0]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[1]);

      for (int pBin=0; pBin<14; pBin++) {
        hisID="RawDataErrorType"; hisID+=(pBin+25); hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[2+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
        error = sqrt(performances[2+pBin]*(performances[2+pBin]+1)/numberOfEvents); 
	((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      for (int pBin=0; pBin<5; pBin++) {
        hisID="TBMType"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[16+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
        error = sqrt(performances[16+pBin]*(performances[16+pBin]+1)/numberOfEvents); 
	((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      for (int pBin=0; pBin<8; pBin++) {
        hisID="TBMMessage"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[21+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
	error = sqrt(performances[21+pBin]*(performances[21+pBin]+1)/numberOfEvents); 
        ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      for (int pBin=0; pBin<7; pBin++) {
        hisID="FEDfullType"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[29+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun); 
	error = sqrt(performances[29+pBin]*(performances[29+pBin]+1)/numberOfEvents); 
        ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      for (int pBin=0; pBin<37; pBin++) {
        hisID="FEDtimeoutChannel"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[36+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
        error = sqrt(performances[36+pBin]*(performances[36+pBin]+1)/numberOfEvents); 
	((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      hisID="SLinkErrSize_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[73]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[74]);
      
      hisID="FEDmaxErrLink_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[75]);
      
      hisID="maxErr36ROC_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[76]);
      
      hisID="maxErrDCol_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[77]);
      
      hisID="maxErrPixelRow_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[78]);
      
      hisID="maxErr38ROC_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[79]);
      
      hisID="NumberOfDigis_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[80]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[81]);
      
      hisID="ADC_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[82]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[83]);
      
      hisID="DigimapHot_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[84]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun); 
      error = sqrt(performances[84]*(performances[84]+1)/numberOfEvents); 
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      
      hisID="DigimapCold_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[85]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun); 
      error = sqrt(performances[85]*(performances[85]+1)/numberOfEvents); 
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);

      hisID="NumberOfClusters_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[86]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[87]);
      
      hisID="ClusterCharge_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[88]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[89]);
      
      hisID="ClusterSizeX_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[90]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[91]);
      
      hisID="ClusterSizeY_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[92]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[93]);
      
      hisID="ClustermapHot_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[94]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun); 
      error = sqrt(performances[94]*(performances[94]+1)/numberOfEvents); 
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      
      hisID="ClustermapCold_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[95]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun); 
      error = sqrt(performances[95]*(performances[95]+1)/numberOfEvents); 
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);

      hisID="NumberOfRecHits_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[96]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[97]);
      
      hisID="RecHitMatchedClusterSizeX_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[98]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[99]);
      
      hisID="RecHitMatchedClusterSizeY_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[100]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[101]);
      
      hisID="ClustermapHot_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[102]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun); 
      error = sqrt(performances[102]*(performances[102]+1)/numberOfEvents); 
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      
      hisID="ClustermapCold_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[103]); 
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      error = sqrt(performances[103]*(performances[103]+1)/numberOfEvents); 
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);

      hisID="ResidualX_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[104]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[105]);
      
      hisID="ResidualY_DetID"; hisID+=*iDet; hisID+="_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[106]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[107]);
      
      hisID="NumberOfRawDataErrors_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[0]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[1],2)));

      for (int pBin=0; pBin<14; pBin++) {
        hisID="RawDataErrorType"; hisID+=(pBin+25); hisID+="_AllDets_AllRuns";     
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[2+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
        exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
        error = sqrt(pow(exErr,2)+performances[2+pBin]*(performances[2+pBin]+1)/numberOfEvents);
	((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      for (int pBin=0; pBin<5; pBin++) {
        hisID="TBMType"; hisID+=pBin; hisID+="_AllDets_AllRuns";	     
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[16+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
        exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
        error = sqrt(pow(exErr,2)+performances[16+pBin]*(performances[16+pBin]+1)/numberOfEvents);
	((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      for (int pBin=0; pBin<8; pBin++) {
        hisID="TBMMessage"; hisID+=pBin; hisID+="_AllDets_AllRuns";	     
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[21+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
        exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
        error = sqrt(pow(exErr,2)+performances[21+pBin]*(performances[21+pBin]+1)/numberOfEvents);
	((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      for (int pBin=0; pBin<7; pBin++) {
        hisID="FEDfullType"; hisID+=pBin; hisID+="_AllDets_AllRuns";       
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[29+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
        exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
        error = sqrt(pow(exErr,2)+performances[29+pBin]*(performances[29+pBin]+1)/numberOfEvents);
	((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      for (int pBin=0; pBin<37; pBin++) {
        hisID="FEDtimeoutChannel"; hisID+=pBin; hisID+="_AllDets_AllRuns";	     
        ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[36+pBin]);
        iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
        exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
        error = sqrt(pow(exErr,2)+performances[36+pBin]*(performances[36+pBin]+1)/numberOfEvents);
	((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
      } 
      hisID="SLinkErrSize_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[73]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[74],2)));
    
      hisID="FEDmaxErrLink_AllDets_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[75]);
    
      hisID="maxErr36ROC_AllDets_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[76]);
    
      hisID="maxErrDCol_AllDets_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[77]);
    
      hisID="maxErrPixelRow_AllDets_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[78]);
    
      hisID="maxErr38ROC_AllDets_AllRuns";	     
      ((TH2F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[79]);

      hisID="NumberOfDigis_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[80]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[81],2)));
    
      hisID="ADC_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[82]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[83],2)));
    
      hisID="DigimapHot_AllDets_AllRuns";    
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[84]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      error = sqrt(pow(exErr,2)+performances[84]*(performances[84]+1)/numberOfEvents);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
    
      hisID="DigimapCold_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[85]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      error = sqrt(pow(exErr,2)+performances[85]*(performances[85]+1)/numberOfEvents);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);

      hisID="NumberOfClusters_AllDets_AllRuns";      
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[86]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[87],2)));
    
      hisID="ClusterCharge_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[88]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[89],2)));

      hisID="ClusterSizeX_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[90]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[91],2)));
    
      hisID="ClusterSizeY_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[92]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[93],2)));
    
      hisID="ClustermapHot_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[94]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      error = sqrt(pow(exErr,2)+performances[94]*(performances[94]+1)/numberOfEvents);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
    
      hisID="ClustermapCold_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[95]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      error = sqrt(pow(exErr,2)+performances[95]*(performances[95]+1)/numberOfEvents);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);

      hisID="NumberOfRecHits_AllDets_AllRuns";       
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[96]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[97],2)));
    
      hisID="RecHitMatchedClusterSizeX_AllDets_AllRuns";     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[98]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[99],2)));
    
      hisID="RecHitMatchedClusterSizeY_AllDets_AllRuns";     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[100]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[101],2)));
    
      hisID="ClustermapHot_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[102]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      error = sqrt(pow(exErr,2)+performances[102]*(performances[102]+1)/numberOfEvents);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);
    
      hisID="ClustermapCold_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[103]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      error = sqrt(pow(exErr,2)+performances[103]*(performances[103]+1)/numberOfEvents);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, error);

      hisID="ResidualX_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[104]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[105],2)));
    
      hisID="ResidualY_AllDets_AllRuns";	     
      ((TH1F*) AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[106]);
      iBin = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      exErr = ((TH1F*) AllDetHistograms->FindObject(hisID))->GetBinError(iBin);
      ((TH1F*) AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(performances[107],2)));    

      cout << "filling histograms for run number = "<< presentRun_<<"; detId = "<< *iDet << endl;  
    }    
  }
}


void SiPixelHistoricInfoReader::analyze(const edm::Event& event, const edm::EventSetup& iSetup) {}


void SiPixelHistoricInfoReader::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}


void SiPixelHistoricInfoReader::endJob() {
  for (std::vector<uint32_t>::const_iterator iDet=allDetIds.begin(); iDet!=allDetIds.end(); ++iDet) {
    hisID="NumberOfRawDataErrors_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

    for (int pBin=0; pBin<14; pBin++) {
      hisID="RawDataErrorType"; hisID+=(pBin+25); hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
      ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    } 
    for (int pBin=0; pBin<5; pBin++) {
      hisID="TBMType"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
      ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    } 
    for (int pBin=0; pBin<8; pBin++) {
      hisID="TBMMessage"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
      ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    } 
    for (int pBin=0; pBin<7; pBin++) {
      hisID="FEDfullType"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
      ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    } 
    for (int pBin=0; pBin<37; pBin++) {
      hisID="FEDtimeoutChannel"; hisID+=pBin; hisID+="_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
      ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    } 
    hisID="SLinkErrSize_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="FEDmaxErrLink_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="maxErr36ROC_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="maxErrDCol_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="maxErrPixelRow_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="maxErr38ROC_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="NumberOfDigis_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="ADC_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="DigimapHot_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="DigimapCold_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

    hisID="NumberOfClusters_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="ClusterCharge_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

    hisID="ClusterSizeX_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
 
    hisID="ClusterSizeY_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="ClustermapHot_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="ClustermapCold_DetID"; hisID+=*iDet; hisID+="_AllRuns"; 	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

    hisID="NumberOfRecHits_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="RecHitMatchedClusterSizeX_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="RecHitMatchedClusterSizeY_DetID"; hisID+=*iDet; hisID+="_AllRuns";	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="ClustermapHot_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="ClustermapCold_DetID"; hisID+=*iDet; hisID+="_AllRuns"; 	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

    hisID="ResidualX_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
    
    hisID="ResidualY_DetID"; hisID+=*iDet; hisID+="_AllRuns";  	   
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");    
  }
  hisID="NumberOfRawDataErrors_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

  for (int pBin=0; pBin<14; pBin++) {
    hisID="RawDataErrorType"; hisID+=(pBin+25); hisID+="_AllDets_AllRuns";		 
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  } 
  for (int pBin=0; pBin<5; pBin++) {
    hisID="TBMType"; hisID+=pBin; hisID+="_AllDets_AllRuns";  	 
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  } 
  for (int pBin=0; pBin<8; pBin++) {
    hisID="TBMMessage"; hisID+=pBin; hisID+="_AllDets_AllRuns";	 
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  } 
  for (int pBin=0; pBin<7; pBin++) {
    hisID="FEDfullType"; hisID+=pBin; hisID+="_AllDets_AllRuns";	 
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  } 
  for (int pBin=0; pBin<37; pBin++) {
    hisID="FEDtimeoutChannel"; hisID+=pBin; hisID+="_AllDets_AllRuns";	 
    ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  } 
  hisID="SLinkErrSize_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="FEDmaxErrLink_AllDets_AllRuns";	 
  ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="maxErr36ROC_AllDets_AllRuns";	 
  ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="maxErrDCol_AllDets_AllRuns";	 
  ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="maxErrPixelRow_AllDets_AllRuns";	 
  ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="maxErr38ROC_AllDets_AllRuns";	 
  ((TH2F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="NumberOfDigis_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="ADC_AllDets_AllRuns";  	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="DigimapHot_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="DigimapCold_AllDets_AllRuns";  	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

  hisID="NumberOfClusters_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="ClusterCharge_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

  hisID="ClusterSizeX_AllDets_AllRuns"; 	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
 
  hisID="ClusterSizeY_AllDets_AllRuns"; 	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="ClustermapHot_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="ClustermapCold_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

  hisID="NumberOfRecHits_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="RecHitMatchedClusterSizeX_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="RecHitMatchedClusterSizeY_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="ClustermapHot_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="ClustermapCold_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="ResidualX_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");
  
  hisID="ResidualY_AllDets_AllRuns";	 
  ((TH1F*) AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");

  cout << "label deflating finished" << endl; 
  
  outputFile->Write();
  outputFile->Close();
}
