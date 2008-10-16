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


SiPixelHistoricInfoReader::SiPixelHistoricInfoReader(const edm::ParameterSet& pSet) {
  parameterSet_ = pSet;  

  variables_ = parameterSet_.getUntrackedParameter<vstring>("variables");
  for (int i=0; i<10; i++) variable_[i] = false; 
  for (vector<string>::const_iterator variable = variables_.begin(); variable!=variables_.end(); ++variable) {
    if (variable->compare("errorType")==0) variable_[0] = true; 
    if (variable->compare("ndigis")==0)    variable_[1] = true; 
    if (variable->compare("adc")==0)       variable_[2] = true; 
    if (variable->compare("nclusters")==0) variable_[3] = true; 
    if (variable->compare("charge")==0)    variable_[4] = true; 
    if (variable->compare("sizeX")==0)     variable_[5] = true; 
    if (variable->compare("sizeY")==0)     variable_[6] = true; 
    if (variable->compare("nRecHits")==0)  variable_[7] = true; 
    if (variable->compare("residualX")==0) variable_[8] = true; 
    if (variable->compare("residualY")==0) variable_[9] = true; 
  }
  normEvents_ = parameterSet_.getUntrackedParameter<bool>("normEvents", false);  
  printDebug_ = parameterSet_.getUntrackedParameter<bool>("printDebug", false); 
  outputFile_ = parameterSet_.getUntrackedParameter<string>("outputFile", "testPixelHistoryReader.root");
  
  firstBeginRun_ = true; 
} 


SiPixelHistoricInfoReader::~SiPixelHistoricInfoReader() {}


void SiPixelHistoricInfoReader::beginJob(const edm::EventSetup& iSetup) {
  outputFile = new TFile(outputFile_.data(), "RECREATE");
} 


string SiPixelHistoricInfoReader::getMEregionString(uint32_t detID) const {
  uint32_t localMEdetID = detID; 
  TString regionStr; 
       if (localMEdetID>100000000) { regionStr = "det"; regionStr += localMEdetID; }
  else if (localMEdetID<40)        { regionStr = "FED"; regionStr += localMEdetID; }
  else if (localMEdetID>99 && localMEdetID<120) { 
    localMEdetID -= 100; 
    if (localMEdetID<12) { 
      regionStr = "Barrel/"; 
    	   if (localMEdetID<3) { regionStr += "Shell_mI/";	              }
      else if (localMEdetID<6) { regionStr += "Shell_mO/"; localMEdetID -= 3; }
      else if (localMEdetID<9) { regionStr += "Shell_pI/"; localMEdetID -= 6; } 
      else		       { regionStr += "Shell_pO/"; localMEdetID -= 9; }
      regionStr += "Layer_"; regionStr += (localMEdetID+1); 
    }
    else { 
      regionStr = "Endcap/"; localMEdetID -= 12; 
    	   if (localMEdetID<2) { regionStr += "HalfCylinder_mI/";		     }
      else if (localMEdetID<4) { regionStr += "HalfCylinder_mO/"; localMEdetID -= 2; }
      else if (localMEdetID<6) { regionStr += "HalfCylinder_pI/"; localMEdetID -= 4; } 
      else		       { regionStr += "HalfCylinder_pO/"; localMEdetID -= 6; }
      regionStr += "Disk_"; regionStr += (localMEdetID+1); 
    }
  } 
  else {
    localMEdetID -= 1000; 
    if (localMEdetID<192) { 
      regionStr = "Barrel/"; 
    	   if (localMEdetID<48)  { regionStr += "Shell_mI/";			  }
      else if (localMEdetID<96)  { regionStr += "Shell_mO/"; localMEdetID -= 48;  }
      else if (localMEdetID<144) { regionStr += "Shell_pI/"; localMEdetID -= 96;  } 
      else			 { regionStr += "Shell_pO/"; localMEdetID -= 144; }
           if (localMEdetID<10) { regionStr += "Layer_1/";		       }
      else if (localMEdetID<26) { regionStr += "Layer_2/"; localMEdetID -= 10; }
      else			{ regionStr += "Layer_3/"; localMEdetID -= 26; }
      regionStr += "Ladder_"; regionStr += (localMEdetID+1); 
    }
    else { 
      regionStr = "Endcap/"; localMEdetID -= 192; 
    	   if (localMEdetID<24) { regionStr += "HalfCylinder_mI/";		       }
      else if (localMEdetID<48) { regionStr += "HalfCylinder_mO/"; localMEdetID -= 24; }
      else if (localMEdetID<72) { regionStr += "HalfCylinder_pI/"; localMEdetID -= 48; } 
      else			{ regionStr += "HalfCylinder_pO/"; localMEdetID -= 72; }
      if (localMEdetID<12) { regionStr += "Disk_1/"; }
      else		   { regionStr += "Disk_2/"; localMEdetID -= 12; } 
      regionStr += "Blade_"; regionStr += (localMEdetID+1); 
    }
  }
  return regionStr.Data(); 
}


void SiPixelHistoricInfoReader::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  edm::ESHandle<SiPixelPerformanceSummary> pSummary;
  iSetup.get<SiPixelPerformanceSummaryRcd>().get(pSummary); 

  if (firstBeginRun_) {
    firstBeginRun_ = false;

    allDetIds.clear(); // allDetIds.push_back(369345800);
    pSummary->getAllDetIds(allDetIds);
      
    AllDetHistograms = new TObjArray();

    for (vector<uint32_t>::const_iterator iDet = allDetIds.begin(); iDet!=allDetIds.end(); ++iDet) {  
      string detRegion = getMEregionString(*iDet); 
      
      if (variable_[0] && *iDet<40) { 
        for (int pBin=0; pBin<15; pBin++) {
  	  hisID = "errorType"; hisID += (pBin+25); hisID += "_"; hisID += *iDet; 	     
          title = "errorType"; title += (pBin+25); title += " "; title += detRegion; 	        
  	  AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1));
  	  ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin); 
        } 
      }
      if (variable_[1] && *iDet>99) {
        hisID = "nDigis_"; hisID += *iDet; 	        
        title = "nDigis "; title += detRegion; 	        
        AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1)); 	        
        ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);  
      }
      if (variable_[2] && *iDet>99) {
        hisID = "adc_"; hisID += *iDet; 		        
        title = "adc "; title += detRegion; 	        
        AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1)); 	        
        ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);  
      }
      if (variable_[3] && *iDet>99) {
        hisID = "nClusters_"; hisID += *iDet;         
        title = "nClusters "; title += detRegion; 	        
        AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1)); 	        
        ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);  
      }
      if (variable_[4] && *iDet>99) {
        hisID = "charge_"; hisID += *iDet; 	        
        title = "charge "; title += detRegion; 	        
        AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1)); 	        
        ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);  
      }
      if (variable_[5] && *iDet>99) {
        hisID = "clusterSizeX_"; hisID += *iDet; 	        
        title = "clusterSizeX "; title += detRegion; 	        
        AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1)); 	        
        ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);  
      }
      if (variable_[6] && *iDet>99) {
        hisID = "clusterSizeY_"; hisID += *iDet; 	        
        title = "clusterSizeY "; title += detRegion; 	        
        AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1)); 	        
        ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);  
      }
      if (variable_[7] && *iDet>99) {
        hisID = "nRecHits_"; hisID += *iDet;          
        title = "nRecHits "; title += detRegion; 	        
        AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1)); 	        
        ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);  
      }
      if (variable_[8] && *iDet>99) {
        hisID = "residualX_"; hisID += *iDet; 	        
        title = "residualX "; title += detRegion; 	        
        AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1)); 	        
        ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);  
      }
      if (variable_[9] && *iDet>99) {
        hisID = "residualY_"; hisID += *iDet; 	        
        title = "residualY "; title += detRegion; 	        
        AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1)); 	        
        ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);  
      } 
    }
    if (printDebug_) {
      hisID = "allVars_allDets_cruDebugging";					
      title = "allVars_allDets_together for Crude Debugging";					
      AllDetHistograms->Add(new TH1F(hisID, title, 1, 0, 1));			
      ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBit(TH1::kCanRebin);	
    }
  }
  if (pSummary->getRunNumber()==run.run()) { // pSummary's run changes only when the table is newly retrieved 
    cout << "SiPixelPerformanceSummary table retrieved for run "<< run.run() << endl; 

    TString sRun; sRun += pSummary->getRunNumber();     
    float nEvents = pSummary->getNumberOfEvents(); 
    float SF = 1.0; if (normEvents_) SF = 100000.0/nEvents; 
    
    if (printDebug_) pSummary->printAll(); 
    for (vector<uint32_t>::const_iterator iDet = allDetIds.begin(); iDet!=allDetIds.end(); ++iDet) {
      vector<float> performances; 
      performances.clear();
      pSummary->getDetSummary(*iDet, performances); 
      
      if (variable_[0] && *iDet<40) {
        for (int pBin=0; pBin<15; pBin++) {
      	  hisID = "errorType"; hisID += (pBin+25); hisID += "_"; hisID += *iDet;      
      	  ((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[2+pBin]/nEvents);
	  int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
	  ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(performances[2+pBin])/nEvents);          
	  if (printDebug_) fillDebugHistogram(sRun, performances[2+pBin]/nEvents, sqrt(performances[2+pBin])/nEvents); 
      	} 
      } 
      if (variable_[1] && *iDet>99) {
      	hisID = "nDigis_"; hisID += *iDet; 	       
      	((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[80]*SF);
      	int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      	((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[81]*SF);
        if (printDebug_) fillDebugHistogram(sRun, performances[80]*SF, performances[81]*SF); 
      } 
      if (variable_[2] && *iDet>99) {      
      	hisID = "adc_"; hisID += *iDet; 	       
      	((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[82]*SF);
      	int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      	((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[83]*SF);
        if (printDebug_) fillDebugHistogram(sRun, performances[82]*SF, performances[83]*SF); 
      } 
      if (variable_[3] && *iDet>99) {      	
      	hisID = "nClusters_"; hisID += *iDet;        
      	((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[86]*SF);
      	int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      	((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[87]*SF);      
        if (printDebug_) fillDebugHistogram(sRun, performances[86]*SF, performances[87]*SF); 
      } 
      if (variable_[4] && *iDet>99) {      
      	hisID = "charge_"; hisID += *iDet; 	       
      	((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[88]*SF);
      	int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      	((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[89]*SF);      
        if (printDebug_) fillDebugHistogram(sRun, performances[88]*SF, performances[89]*SF); 
      } 
      if (variable_[5] && *iDet>99) {      
      	hisID = "clusterSizeX_"; hisID += *iDet; 	       
      	((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[90]*SF);
      	int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      	((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[91]*SF);
        if (printDebug_) fillDebugHistogram(sRun, performances[90]*SF, performances[91]*SF); 
      } 
      if (variable_[6] && *iDet>99) {      
      	hisID = "clusterSizeY_"; hisID += *iDet; 	       
      	((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[92]*SF);
      	int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      	((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[93]*SF);      
        if (printDebug_) fillDebugHistogram(sRun, performances[92]*SF, performances[93]*SF); 
      } 
      if (variable_[7] && *iDet>99) {      
      	hisID = "nRecHits_"; hisID += *iDet;         
      	((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[96]*SF);
      	int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      	((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[97]*SF);      
        if (printDebug_) fillDebugHistogram(sRun, performances[96]*SF, performances[97]*SF); 
      } 
      if (variable_[8] && *iDet>99) {      
      	hisID = "residualX_"; hisID += *iDet;       
      	((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[104]*SF);
      	int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      	((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[105]*SF);      
        if (printDebug_) fillDebugHistogram(sRun, performances[104]*SF, performances[105]*SF); 
      } 
      if (variable_[9] && *iDet>99) {      
      	hisID = "residualY_"; hisID += *iDet;       
      	((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, performances[106]*SF);
      	int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);
      	((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, performances[107]*SF);      
        if (printDebug_) fillDebugHistogram(sRun, performances[106]*SF, performances[107]*SF); 
      } 
    }    
  }
}


void SiPixelHistoricInfoReader::fillDebugHistogram(TString sRun, float pMean, float pRMS) {
  hisID = "allVars_allDets_cruDebugging"; 					 
  ((TH1F*)AllDetHistograms->FindObject(hisID))->Fill(sRun, pMean);  				 
  int iBin = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetXaxis()->FindBin(sRun);				 
  float exErr = ((TH1F*)AllDetHistograms->FindObject(hisID))->GetBinError(iBin);					 
  ((TH1F*)AllDetHistograms->FindObject(hisID))->SetBinError(iBin, sqrt(pow(exErr,2)+pow(pRMS,2)));   
}


void SiPixelHistoricInfoReader::analyze(const edm::Event& event, const edm::EventSetup& iSetup) {}


void SiPixelHistoricInfoReader::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}


void SiPixelHistoricInfoReader::endJob() {
  for (vector<uint32_t>::const_iterator iDet=allDetIds.begin(); iDet!=allDetIds.end(); ++iDet) {
    if (variable_[0] && *iDet<40) {
      for (int pBin=0; pBin<15; pBin++) {							       
        hisID = "errorType"; hisID += (pBin+25); hisID += "_"; hisID += *iDet;       
        ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			       
      } 											       
    } 
    if (variable_[1] && *iDet>99) {
      hisID = "nDigis_"; hisID += *iDet; 				       
      ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			       
    } 
    if (variable_[2] && *iDet>99) {
      hisID = "adc_"; hisID += *iDet; 					       
      ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			       
    } 
    if (variable_[3] && *iDet>99) {
      hisID = "nClusters_"; hisID += *iDet;   			       
      ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			       
    } 
    if (variable_[4] && *iDet>99) {
      hisID = "charge_"; hisID += *iDet; 				       
      ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			       
    } 
    if (variable_[5] && *iDet>99) {
      hisID = "clusterSizeX_"; hisID += *iDet; 				       
      ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			       
    } 
    if (variable_[6] && *iDet>99) {
      hisID = "clusterSizeY_"; hisID += *iDet; 				       
      ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			       
    } 
    if (variable_[7] && *iDet>99) {
      hisID = "nRecHits_"; hisID += *iDet; 				       
      ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			               											      
    } 
    if (variable_[8] && *iDet>99) {
      hisID = "residualX_"; hisID += *iDet;  				       
      ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			               											      
    } 
    if (variable_[9] && *iDet>99) {
      hisID = "residualY_"; hisID += *iDet;  				       
      ((TH1F*)AllDetHistograms->FindObject(hisID))->LabelsDeflate("X");			       
    }
  }
  if (printDebug_) ((TH1F*)AllDetHistograms->FindObject("allVars_allDets_cruDebugging"))->LabelsDeflate("X"); 
  
  outputFile->Write();
  outputFile->Close();
}
