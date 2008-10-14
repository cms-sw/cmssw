#include <memory>
#include <math.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <sys/time.h>

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoDQMClient.h"


using namespace edm;
using namespace std;


SiPixelHistoricInfoDQMClient::SiPixelHistoricInfoDQMClient(const ParameterSet& parameterSet) {   
  parameterSet_ = parameterSet;  

  useSummary_ = parameterSet_.getUntrackedParameter<bool>("useSummary", true);
  printDebug_ = parameterSet_.getUntrackedParameter<bool>("printDebug",false); 
  writeHisto_ = parameterSet_.getUntrackedParameter<bool>("writeHisto",false); 

  inputFiles_ = parameterSet_.getUntrackedParameter<vstring>("inputFiles");
  outputDir_  = parameterSet_.getUntrackedParameter<string>("outputDir",".");
}


SiPixelHistoricInfoDQMClient::~SiPixelHistoricInfoDQMClient() {}


void SiPixelHistoricInfoDQMClient::beginJob(const EventSetup& eventSetup) {
  dbe_ = Service<DQMStore>().operator->(); 
}


void SiPixelHistoricInfoDQMClient::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {}


void SiPixelHistoricInfoDQMClient::endRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  for (vector<string>::const_iterator iFile = inputFiles_.begin(); iFile!=inputFiles_.end(); ++iFile) {
    unsigned int runNumberfromFilename = atoi(iFile->substr(iFile->find("Run",0)+3,5).data()); 
    if (runNumberfromFilename==run.run()) {    
      dbe_->open(iFile->data(), true);
    
      performanceSummary = new SiPixelPerformanceSummary();   
      performanceSummary->setRunNumber(runNumberfromFilename); 

      retrieveMEs(); 
      fillPerformanceSummary();  
      writeDB(); 
  
      if (writeHisto_) {
        ostringstream endRunOutputFile; 
        endRunOutputFile << outputDir_<<"/SiPixelHistoricInfoDQMClient_"<< runNumberfromFilename <<".root"; 
        dbe_->save(endRunOutputFile.str()); 
      }
    }
  }
}


void SiPixelHistoricInfoDQMClient::analyze(const Event& event, const EventSetup& eventSetup) {}


void SiPixelHistoricInfoDQMClient::endJob() {
  if (writeHisto_) {
    ostringstream endJobOutputFile; 
    endJobOutputFile << outputDir_ << "/SiPixelHistoricInfoDQMClient_endJob.root"; 
    dbe_->save(endJobOutputFile.str());
  }
}


void SiPixelHistoricInfoDQMClient::retrieveMEs() {
  ClientPointersToModuleMEs.clear(); 

  vector<string> listOfMEsWithFullPath;
  dbe_->getContents(listOfMEsWithFullPath);
 
  bool noModules = true, noSummary = true; 
  for (vector<string>::const_iterator iME=listOfMEsWithFullPath.begin(); iME!=listOfMEsWithFullPath.end(); iME++) {
    if (iME->find("Pixel",0)!=string::npos) { 
      if (iME->find("Module_",0)!=string::npos) { noModules = false; break; } 
      else if (iME->find("Layer_",0)!=string::npos || 
               iME->find("Disk_",0)!=string::npos) { noSummary = false; break; }
    }
  }
  if ( useSummary_ && noSummary) cout << "use summary MEs but NO ladder/blade summary MEs in this input file! " << endl;
  if (!useSummary_ && noModules) cout << "use module/FED MEs but NO module/FED MEs in this input file! " << endl;

  for (vector<string>::const_iterator iME=listOfMEsWithFullPath.begin(); iME!=listOfMEsWithFullPath.end(); iME++) {
    uint32_t pathLength = (*iME).find(":",0);     
    string thePath = (*iME).substr(0, pathLength); 
    string allHists = (*iME).substr(pathLength+1); 
    
    if (thePath.find("Pixel",0)!=string::npos) { 
      if (thePath.find("FED_",0)!=string::npos) {
        uint histnameLength;
        do {
          histnameLength = allHists.find(",",0);
          string theHist;
          if (histnameLength!=string::npos) {
            theHist = allHists.substr(0, histnameLength);
            allHists.erase(0, histnameLength+1);
          }
          else theHist = allHists; 

          string fullPathHist = thePath + "/" + theHist;		
          MonitorElement* theMEpointer = dbe_->get(fullPathHist);	  
          if (theMEpointer) {
            uint32_t localMEdetID = hManager.getRawId(theMEpointer->getName());
            if (ClientPointersToModuleMEs.find(localMEdetID)==ClientPointersToModuleMEs.end()) {
              vector<MonitorElement*> newMEvector;
              newMEvector.push_back(theMEpointer);
              ClientPointersToModuleMEs.insert(make_pair(localMEdetID, newMEvector));
            }
            else (ClientPointersToModuleMEs.find(localMEdetID)->second).push_back(theMEpointer);
          } 
        } 
        while (histnameLength!=string::npos); 
      } 
      if (thePath.find("Module_",0)!=string::npos) {
	if (!useSummary_) {
  	  uint histnameLength;
  	  do {
  	    histnameLength = allHists.find(",",0);
  	    string theHist;
  	    if (histnameLength!=string::npos) {
  	      theHist = allHists.substr(0, histnameLength);
  	      allHists.erase(0, histnameLength+1);
  	    }
  	    else theHist = allHists; 

  	    string fullPathHist = thePath + "/" + theHist;		  
  	    MonitorElement* theMEpointer = dbe_->get(fullPathHist);  	    
  	    if (theMEpointer) {
	      uint32_t localMEdetID = hManager.getRawId(theMEpointer->getName());
	      if (ClientPointersToModuleMEs.find(localMEdetID)==ClientPointersToModuleMEs.end()) {
  	  	vector<MonitorElement*> newMEvector;
  	  	newMEvector.push_back(theMEpointer);
  	  	ClientPointersToModuleMEs.insert(make_pair(localMEdetID, newMEvector));
  	      }
  	      else (ClientPointersToModuleMEs.find(localMEdetID)->second).push_back(theMEpointer);
  	    } 
  	  } 
  	  while (histnameLength!=string::npos); 
	} 
      } 
      else if (thePath.find("Layer_",0)!=string::npos && thePath.find("Ladder_",0)==string::npos || 
               thePath.find("Disk_",0)!=string::npos && thePath.find("Blade_",0)==string::npos && 
	                                                thePath.find("Panel_",0)==string::npos) {
	if (useSummary_) {
   	  uint32_t localMEdetID = getLayerDiskID(thePath); 	  
	  uint histnameLength;
  	  do {
  	    histnameLength = allHists.find(",",0);
  	    string theHist;
  	    if (histnameLength!=string::npos) {
  	      theHist = allHists.substr(0, histnameLength);
  	      allHists.erase(0, histnameLength+1);
  	    }
  	    else theHist = allHists; 

  	    string fullPathHist = thePath + "/" + theHist;		  
  	    MonitorElement* theMEpointer = dbe_->get(fullPathHist);  	    
  	    if (theMEpointer) {
  	      if (ClientPointersToModuleMEs.find(localMEdetID)==ClientPointersToModuleMEs.end()) {
  	  	vector<MonitorElement*> newMEvector;
  	  	newMEvector.push_back(theMEpointer);
  	  	ClientPointersToModuleMEs.insert(make_pair(localMEdetID, newMEvector));
  	      }
  	      else (ClientPointersToModuleMEs.find(localMEdetID)->second).push_back(theMEpointer);
  	    } 
  	  } 
  	  while (histnameLength!=string::npos); 
	}   
      }
      if (thePath.find("EventInfo",0)!=string::npos) { 
      	uint histnameLength; 
	do {
      	  histnameLength = allHists.find(",",0);
      	  string theHist;
      	  if (histnameLength!=string::npos) {
      	    theHist = allHists.substr(0, histnameLength);
      	    allHists.erase(0, histnameLength+1);
      	  }
      	  else theHist = allHists; 
      	  
	  string fullPathHist = thePath + "/" + theHist;
      	  MonitorElement* theMEpointer = dbe_->get(fullPathHist); 	  
      	  if (theMEpointer) {
	    if (theHist.find("iRun",0)!=string::npos) performanceSummary->setRunNumber(theMEpointer->getIntValue()); 
	    if (theHist.find("iEvent",0)!=string::npos) performanceSummary->setNumberOfEvents(theMEpointer->getIntValue()); 
	    if (theHist.find("processTimeStamp",0)!=string::npos) performanceSummary->setTimeValue((unsigned 
	                                                                                            long 
												    long)theMEpointer->getFloatValue());
	  }
	} 
        while (histnameLength!=string::npos); 
      } 
    } 
  } 
} 


uint32_t SiPixelHistoricInfoDQMClient::getLayerDiskID(string thePath) const {
  uint32_t regionID = 100; 
         if (thePath.find("Barrel",0)!=string::npos) { 
    regionID += 0; 
    string shell = thePath.substr(thePath.find("Shell",0)+6,2);     
         if (shell.compare("mI")==0) regionID += 0; 
    else if (shell.compare("mO")==0) regionID += 3; 
    else if (shell.compare("pI")==0) regionID += 6; 
    else if (shell.compare("pO")==0) regionID += 9;     
    else cout << "Shell_" << shell.data() << "?!" << endl; 
    string layer = thePath.substr(thePath.find("Layer",0)+6,1); 
    regionID += (atoi(layer.data())-1); 
  }
  else if (thePath.find("Endcap",0)!=string::npos) { 
    regionID += 12; 
    string halfCylinder = thePath.substr(thePath.find("HalfCylinder",0)+13,2);     
         if (halfCylinder.compare("mI")==0) regionID += 0; 
    else if (halfCylinder.compare("mO")==0) regionID += 2; 
    else if (halfCylinder.compare("pI")==0) regionID += 4; 
    else if (halfCylinder.compare("pO")==0) regionID += 6; 
    else cout << "HalfCylinder_" << halfCylinder.data() << "?!" << endl; 
    string disk = thePath.substr(thePath.find("Disk",0)+5,1);     
    regionID += (atoi(disk.data())-1); 
  }
  return regionID; 
}


uint32_t SiPixelHistoricInfoDQMClient::getBLadID(string thePath) const {
  uint32_t regionID = 1000; 
         if (thePath.find("Barrel",0)!=string::npos) { 
    regionID += 0; 
    string shell = thePath.substr(thePath.find("Shell",0)+6,2);     
         if (shell.compare("mI")==0) regionID +=   0; 
    else if (shell.compare("mO")==0) regionID +=  48; 
    else if (shell.compare("pI")==0) regionID +=  96; 
    else if (shell.compare("pO")==0) regionID += 144;     
    else cout << "Shell_" << shell.data() << "?!" << endl; 
    string layer = thePath.substr(thePath.find("Layer",0)+6,1); 
         if (layer.compare("1")==0) regionID +=  0; 
    else if (layer.compare("2")==0) regionID += 10; 
    else if (layer.compare("3")==0) regionID += 26; 
    else cout << "Layer_" << layer.data() << "?!" << endl; 
    string ladder = thePath.substr(thePath.find("Ladder",0)+7,2); 
    regionID += (atoi(ladder.data())-1); 
  }
  else if (thePath.find("Endcap",0)!=string::npos) { 
    regionID += 192; 
    string halfCylinder = thePath.substr(thePath.find("HalfCylinder",0)+13,2);     
         if (halfCylinder.compare("mI")==0) regionID +=  0; 
    else if (halfCylinder.compare("mO")==0) regionID += 24; 
    else if (halfCylinder.compare("pI")==0) regionID += 48; 
    else if (halfCylinder.compare("pO")==0) regionID += 72; 
    else cout << "HalfCylinder_" << halfCylinder.data() << "?!" << endl; 
    string disk = thePath.substr(thePath.find("Disk",0)+5,1);     
    string blade = thePath.substr(thePath.find("Blade",0)+6,2);     
    regionID += (12*(atoi(disk.data())-1) + atoi(blade.data())-1); 
  }
  return regionID; 
}


void SiPixelHistoricInfoDQMClient::fillPerformanceSummary() const {
  unsigned int nEvents = performanceSummary->getNumberOfEvents(); 
  if (printDebug_) cout << "number of events in run "<< performanceSummary->getRunNumber() <<" = "<< nEvents << endl; 
  
  for (map< uint32_t, vector<MonitorElement*> >::const_iterator iMEvec=ClientPointersToModuleMEs.begin(); 
       iMEvec!=ClientPointersToModuleMEs.end(); iMEvec++) {
    uint32_t localMEdetID = iMEvec->first;
    vector<MonitorElement*> theMEvector = iMEvec->second;
    /*
    if (printDebug_) { 
      cout << localMEdetID << ":"; 
      for (vector<MonitorElement*>::const_iterator iMEpntr = theMEvector.begin(); 
           iMEpntr!=theMEvector.end(); iMEpntr++) cout << (*iMEpntr)->getName() << ","; 
      cout << endl; 
    } */
    for (vector<MonitorElement*>::const_iterator iMEpntr = theMEvector.begin(); 
         iMEpntr!=theMEvector.end(); iMEpntr++) {
      string theMEname = (*iMEpntr)->getName(); 

// from SiPixelMonitorRawData
      /*
      if (theMEname.find("NErrors")!=string::npos) { 
	performanceSummary->setNumberOfRawDataErrors(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      } */
      if (theMEname.find("errorType")!=string::npos) { 
	for (int v=25; v<40; v++) {
	  int b = (*iMEpntr)->getTH1()->GetXaxis()->FindBin(v); 
	  float percentage = (*iMEpntr)->getBinContent(b)/float(nEvents); 
	  performanceSummary->setRawDataErrorType(localMEdetID, v-25, percentage); 
	}
      } /*
      if (theMEname.find("TBMType")!=string::npos) { 
	for (int b=1; b<(*iMEpntr)->getNbinsX(); b++) {
	  if (b<5) {
	    float percentage = (*iMEpntr)->getBinContent(b+1)/float(nEvents); 
    	    performanceSummary->setTBMType(localMEdetID, b, percentage);
          } 
	  else if (localMEdetID<40) cout << "TBMType_"<< localMEdetID <<" in bin "<< b << endl;
	}
      } 
      if (theMEname.find("TBMMessage")!=string::npos) { 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {
	  if (b<8) { 
	    float percentage = (*iMEpntr)->getBinContent(b+1)/float(nEvents); 
    	    performanceSummary->setTBMMessage(localMEdetID, b, percentage);
          } 
	  else if (localMEdetID<40) cout << "TBMMessage_"<< localMEdetID <<" in bin "<< b << endl;
	}
      } 
      if (theMEname.find("fullType")!=string::npos) { 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {
	  if (b<7) {
	    float percentage = (*iMEpntr)->getBinContent(b+1)/float(nEvents); 
    	    performanceSummary->setFEDfullType(localMEdetID, b, percentage);
          } 
	  else if (localMEdetID<40) cout << "fullType_"<< localMEdetID <<" in bin "<< b << endl;
	}
      } 
      if (theMEname.find("chanNmbr")!=string::npos) { 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {
	  if (b<37) {
	    float percentage = (*iMEpntr)->getBinContent(b+1)/float(nEvents); 
    	    performanceSummary->setFEDtimeoutChannel(localMEdetID, b, percentage);
          } 
	  else if (localMEdetID<40) cout << "chanNmbr_"<< localMEdetID <<" in bin "<< b << endl;
	}
      }  
      if (theMEname.find("evtSize")!=string::npos) { 
        performanceSummary->setSLinkErrSize(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS()); 
      } 
      if (theMEname.find("linkId")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
	performanceSummary->setFEDmaxErrLink(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("ROCId")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErr36ROC(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("Type36Hitmap")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH2F()->ProjectionY()->GetMaximumBin(); 
        performanceSummary->setmaxErr36ROC(localMEdetID, (*iMEpntr)->getTH2F()->ProjectionY()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("DCOLId")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErrDCol(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("PXId")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErrPixelRow(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("ROCNmbr")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErr38ROC(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      } */
// from SiPixelMonitorDigi 
      if (theMEname.find("ndigis")!=string::npos && theMEname.find("FREQ")==string::npos) { 
	float avgMean=0.0, avgRMS=0.0; int nBins=0; 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {	  
	  float binMean = (*iMEpntr)->getBinContent(b+1), binRMS = (*iMEpntr)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { 
	  avgMean = (avgMean/float(nBins))*(10000.0/float(nEvents)); 
	  avgRMS = sqrt(avgRMS/float(nBins))*(10000.0/float(nEvents)); 
	}
	performanceSummary->setNumberOfDigis(localMEdetID, avgMean, avgRMS);
	if (printDebug_) cout << "ndigis_"<< localMEdetID <<" "<< avgMean << "+-" << avgRMS << endl;
      } 
      if (theMEname.find("adc")!=string::npos) { 
	float avgMean=0.0, avgRMS=0.0; int nBins=0; 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {	  
	  float binMean = (*iMEpntr)->getBinContent(b+1), binRMS = (*iMEpntr)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { 
	  avgMean = (avgMean/float(nBins))*(10000.0/float(nEvents)); 
	  avgRMS = sqrt(avgRMS/float(nBins))*(10000.0/float(nEvents)); 
	}
	performanceSummary->setADC(localMEdetID, avgMean, avgRMS);
      } /*
      if (theMEname.find("hitmap")!=string::npos) { 
    	int nEmptyCells=0, nHotCells=0; // 4 pixels per bin
	for (int xBin=0; xBin<(*iMEpntr)->getNbinsX(); xBin++) { 
	  for (int yBin=0; yBin<(*iMEpntr)->getNbinsY(); yBin++) { 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)>0.01*(*iMEpntr)->getEntries()) nHotCells++; 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	  } 
	} 
	performanceSummary->setDigimapHotCold(localMEdetID, nHotCells/float(nEvents), // the higher the worse
	                                                    nEmptyCells/float(nEvents)); // the lower the worse
      } */
// from SiPixelMonitorCluster
      if (theMEname.find("nclusters")!=string::npos) {
	float avgMean=0.0, avgRMS=0.0; int nBins=0; 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {	  
	  float binMean = (*iMEpntr)->getBinContent(b+1), binRMS = (*iMEpntr)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { 
	  avgMean = (avgMean/float(nBins))*(10000.0/float(nEvents)); 
	  avgRMS = sqrt(avgRMS/float(nBins))*(10000.0/float(nEvents)); 
	}
	performanceSummary->setNumberOfClusters(localMEdetID, avgMean, avgRMS);
      }
      if (theMEname.find("charge")!=string::npos) {
	float avgMean=0.0, avgRMS=0.0; int nBins=0; 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {	  
	  float binMean = (*iMEpntr)->getBinContent(b+1), binRMS = (*iMEpntr)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { 
	  avgMean = (avgMean/float(nBins))*(10000.0/float(nEvents)); 
	  avgRMS = sqrt(avgRMS/float(nBins))*(10000.0/float(nEvents)); 
	}
	performanceSummary->setClusterCharge(localMEdetID, avgMean, avgRMS);
      }
      if (theMEname.find("sizeX")!=string::npos) {
	float avgMean=0.0, avgRMS=0.0; int nBins=0; 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {	  
	  float binMean = (*iMEpntr)->getBinContent(b+1), binRMS = (*iMEpntr)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { 
	  avgMean = (avgMean/float(nBins))*(10000.0/float(nEvents)); 
	  avgRMS = sqrt(avgRMS/float(nBins))*(10000.0/float(nEvents)); 
	}
	performanceSummary->setClusterSizeX(localMEdetID, avgMean, avgRMS);
      }
      if (theMEname.find("sizeY")!=string::npos) {
	float avgMean=0.0, avgRMS=0.0; int nBins=0; 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {	  
	  float binMean = (*iMEpntr)->getBinContent(b+1), binRMS = (*iMEpntr)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { 
	  avgMean = (avgMean/float(nBins))*(10000.0/float(nEvents)); 
	  avgRMS = sqrt(avgRMS/float(nBins))*(10000.0/float(nEvents)); 
	}
	performanceSummary->setClusterSizeY(localMEdetID, avgMean, avgRMS);
      } /*
      if (theMEname.find("hitmap")!=string::npos) {
    	int nEmptyCells=0, nHotCells=0; // 4 pixels per bin
	for (int xBin=0; xBin<(*iMEpntr)->getNbinsX(); xBin++) { 
	  for (int yBin=0; yBin<(*iMEpntr)->getNbinsY(); yBin++) { 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)>0.01*(*iMEpntr)->getEntries()) nHotCells++; 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	  } 
	} 
	performanceSummary->setClustermapHotCold(localMEdetID, nHotCells/float(nEvents), // the higher the worse
	                                                       nEmptyCells/float(nEvents)); // the lower the worse
      } */
// from SiPixelMonitorRecHit
      if (theMEname.find("nRecHits")!=string::npos) {
	float avgMean=0.0, avgRMS=0.0; int nBins=0; 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {	  
	  float binMean = (*iMEpntr)->getBinContent(b+1), binRMS = (*iMEpntr)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { 
	  avgMean = (avgMean/float(nBins))*(10000.0/float(nEvents)); 
	  avgRMS = sqrt(avgRMS/float(nBins))*(10000.0/float(nEvents)); 
	}
	performanceSummary->setNumberOfRecHits(localMEdetID, avgMean, avgRMS);
      } /*
      if (theMEname.find("ClustX")!=string::npos) {
    	performanceSummary->setRecHitMatchedClusterSizeX(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("ClustY")!=string::npos) {
    	performanceSummary->setRecHitMatchedClusterSizeY(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("xypos")!=string::npos) {
    	int nEmptyCells=0, nHotCells=0; // 4 pixels per bin
	for (int xBin=0; xBin<(*iMEpntr)->getNbinsX(); xBin++) { 
	  for (int yBin=0; yBin<(*iMEpntr)->getNbinsY(); yBin++) { 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)>0.01*(*iMEpntr)->getEntries()) nHotCells++; 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	  } 
	} 
	performanceSummary->setRecHitmapHotCold(localMEdetID, nHotCells/float(nEvents), // the higher the worse
	                                                      nEmptyCells/float(nEvents)); // the lower the worse
      } */
// from SiPixelMonitorTrack
      if (theMEname.find("residualX")!=string::npos) {
	float avgMean=0.0, avgRMS=0.0; int nBins=0; 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {	  
	  float binMean = (*iMEpntr)->getBinContent(b+1), binRMS = (*iMEpntr)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { 
	  avgMean = (avgMean/float(nBins))*(10000.0/float(nEvents)); 
	  avgRMS = sqrt(avgRMS/float(nBins))*(10000.0/float(nEvents)); 
	}
	performanceSummary->setResidualX(localMEdetID, avgMean, avgRMS);
      }
      if (theMEname.find("residualY")!=string::npos) {
	float avgMean=0.0, avgRMS=0.0; int nBins=0; 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {	  
	  float binMean = (*iMEpntr)->getBinContent(b+1), binRMS = (*iMEpntr)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { 
	  avgMean = (avgMean/float(nBins))*(10000.0/float(nEvents)); 
	  avgRMS = sqrt(avgRMS/float(nBins))*(10000.0/float(nEvents)); 
	}
	performanceSummary->setResidualY(localMEdetID, avgMean, avgRMS);
      } 
    }
  }
}


void SiPixelHistoricInfoDQMClient::writeDB() const {
  cout << "SiPixelHistoricInfoDQMClient::writeDB() for run "<< performanceSummary->getRunNumber() << endl; 
  
  Service<cond::service::PoolDBOutputService> mydbservice; 
  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiPixelPerformanceSummaryRcd")) {
      mydbservice->createNewIOV<SiPixelPerformanceSummary>(performanceSummary, 
                                                           mydbservice->beginOfTime(), 
							   mydbservice->endOfTime(), 
							  "SiPixelPerformanceSummaryRcd"); 
    } 
    else {
      mydbservice->appendSinceTime<SiPixelPerformanceSummary>(performanceSummary, 
                                                              mydbservice->currentTime(), 
							     "SiPixelPerformanceSummaryRcd"); 
    }
  }
  else  LogError("writetoDB") << "service unavailable" << endl; 
}
