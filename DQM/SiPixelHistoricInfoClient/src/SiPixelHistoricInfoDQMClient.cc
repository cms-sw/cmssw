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
  mapOfdetIDtoMEs.clear(); 

  vector<string> listOfMEswithFullPath;
  dbe_->getContents(listOfMEswithFullPath);
 
  bool noModules = true, noSummary = true; 
  for (vector<string>::const_iterator iMEstr = listOfMEswithFullPath.begin(); 
       iMEstr!=listOfMEswithFullPath.end(); iMEstr++) {
    if (iMEstr->find("Pixel",0)!=string::npos) { 
      if (iMEstr->find("Module_",0)!=string::npos) { noModules = false; break; } 
      else if (iMEstr->find("Layer_",0)!=string::npos || 
               iMEstr->find("Disk_",0)!=string::npos) { noSummary = false; break; }
    }
  }
  if ( useSummary_ && noSummary) cout << "use summary MEs but NO ladder/blade summary MEs in this input file! " << endl;
  if (!useSummary_ && noModules) cout << "use module/FED MEs but NO module/FED MEs in this input file! " << endl;

  for (vector<string>::const_iterator iMEstr = listOfMEswithFullPath.begin(); 
       iMEstr!=listOfMEswithFullPath.end(); iMEstr++) {
    uint32_t pathLength = iMEstr->find(":",0);     
    string thePath = iMEstr->substr(0, pathLength); 
    string allHists = iMEstr->substr(pathLength+1); 
    
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
          MonitorElement* newME = dbe_->get(fullPathHist);	  
          if (newME) {
            uint32_t newMEdetID = histogramManager.getRawId(newME->getName());

            if (mapOfdetIDtoMEs.find(newMEdetID)==mapOfdetIDtoMEs.end()) {
              vector<MonitorElement*> newMEvector;
              newMEvector.push_back(newME);
              mapOfdetIDtoMEs.insert(make_pair(newMEdetID, newMEvector));
            }
            else (mapOfdetIDtoMEs.find(newMEdetID)->second).push_back(newME);
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
  	    MonitorElement* newME = dbe_->get(fullPathHist);  	    
  	    if (newME) {
	      uint32_t newMEdetID = histogramManager.getRawId(newME->getName());

	      if (mapOfdetIDtoMEs.find(newMEdetID)==mapOfdetIDtoMEs.end()) {
  	  	vector<MonitorElement*> newMEvector;
  	  	newMEvector.push_back(newME);
  	  	mapOfdetIDtoMEs.insert(make_pair(newMEdetID, newMEvector));
  	      }
  	      else (mapOfdetIDtoMEs.find(newMEdetID)->second).push_back(newME);
  	    } 
  	  } 
  	  while (histnameLength!=string::npos); 
	} 
      } 
      else if (thePath.find("Layer_",0)!=string::npos && thePath.find("Ladder_",0)==string::npos || 
               thePath.find("Disk_",0)!=string::npos && thePath.find("Blade_",0)==string::npos && 
	                                                thePath.find("Panel_",0)==string::npos) {
	if (useSummary_) {
   	  uint32_t newMEdetID = getLayerDiskID(thePath); 	  

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
  	    MonitorElement* newME = dbe_->get(fullPathHist);  	    
  	    if (newME) {
  	      if (mapOfdetIDtoMEs.find(newMEdetID)==mapOfdetIDtoMEs.end()) {
  	  	vector<MonitorElement*> newMEvector;
  	  	newMEvector.push_back(newME);
  	  	mapOfdetIDtoMEs.insert(make_pair(newMEdetID, newMEvector));
  	      }
  	      else (mapOfdetIDtoMEs.find(newMEdetID)->second).push_back(newME);
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
      	  
	  if (theHist.find("iRun",0)!=string::npos) { 
	    string fullPathHist = thePath + "/" + theHist;
      	    MonitorElement* strME = dbe_->get(fullPathHist); 	  
      	    if (strME) performanceSummary->setRunNumber(strME->getIntValue()); 
	  }
	  if (theHist.find("iEvent",0)!=string::npos) { 
	    string fullPathHist = thePath + "/" + theHist;
      	    MonitorElement* strME = dbe_->get(fullPathHist); 	  
	    if (strME) performanceSummary->setNumberOfEvents(strME->getIntValue()); 
	  } 
	  if (theHist.find("processTimeStamp",0)!=string::npos) { 
	    string fullPathHist = thePath + "/" + theHist;
      	    MonitorElement* strME = dbe_->get(fullPathHist); 	  
	    if (strME) performanceSummary->setTimeValue((unsigned long long)strME->getFloatValue());
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


uint32_t SiPixelHistoricInfoDQMClient::getLadderBladeID(string thePath) const {
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
  if (useSummary_) fillPerformanceSummaryWithSummaryMEs(); 
  else fillPerformanceSummaryWithModuleMEs(); 
}

void SiPixelHistoricInfoDQMClient::fillPerformanceSummaryWithSummaryMEs() const {
  for (map< uint32_t, vector<MonitorElement*> >::const_iterator iMEvec = mapOfdetIDtoMEs.begin(); 
       iMEvec!=mapOfdetIDtoMEs.end(); iMEvec++) {
    uint32_t theMEdetID = iMEvec->first;
    vector<MonitorElement*> theMEvector = iMEvec->second;

    if (printDebug_) { 
      cout << theMEdetID << ":"; 
      for (vector<MonitorElement*>::const_iterator iME = theMEvector.begin(); 
           iME!=theMEvector.end(); iME++) cout << (*iME)->getName() << ","; 
      cout << endl; 
    } 
    for (vector<MonitorElement*>::const_iterator iME = theMEvector.begin(); iME!=theMEvector.end(); iME++) {
      string theMEname = (*iME)->getName(); 
      
      // from SiPixelMonitorRawData
      if (theMEname.find("errorType")!=string::npos) { 
	for (int v=25; v<40; v++) {
	  int b = (*iME)->getTH1()->GetXaxis()->FindBin(v); 
	  performanceSummary->setRawDataErrorType(theMEdetID, v-25, (*iME)->getBinContent(b)); 
	}
      } 
      // from SiPixelMonitorDigi 
      if (theMEname.find("ndigis")!=string::npos && theMEname.find("FREQ")==string::npos) { 
	int nBins=0; float avgMean=0.0, avgRMS=0.0; 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {	  
	  float binMean = (*iME)->getBinContent(b+1), binRMS = (*iME)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { avgMean = avgMean/float(nBins); avgRMS = sqrt(avgRMS/float(nBins)); }
	performanceSummary->setNumberOfDigis(theMEdetID, avgMean, avgRMS);
      } 
      if (theMEname.find("adc")!=string::npos) { 
	int nBins=0; float avgMean=0.0, avgRMS=0.0; 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {	  
	  float binMean = (*iME)->getBinContent(b+1), binRMS = (*iME)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { avgMean = avgMean/float(nBins); avgRMS = sqrt(avgRMS/float(nBins)); }
	performanceSummary->setADC(theMEdetID, avgMean, avgRMS);
      } 
      // from SiPixelMonitorCluster
      if (theMEname.find("nclusters")!=string::npos) {
	int nBins=0; float avgMean=0.0, avgRMS=0.0; 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {	  
	  float binMean = (*iME)->getBinContent(b+1), binRMS = (*iME)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { avgMean = avgMean/float(nBins); avgRMS = sqrt(avgRMS/float(nBins)); }
	performanceSummary->setNumberOfClusters(theMEdetID, avgMean, avgRMS);
      }
      if (theMEname.find("charge")!=string::npos) {
	int nBins=0; float avgMean=0.0, avgRMS=0.0; 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {	  
	  float binMean = (*iME)->getBinContent(b+1), binRMS = (*iME)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { avgMean = avgMean/float(nBins); avgRMS = sqrt(avgRMS/float(nBins)); }
	performanceSummary->setClusterCharge(theMEdetID, avgMean, avgRMS);
      }
      if (theMEname.find("sizeX")!=string::npos) {
	int nBins=0; float avgMean=0.0, avgRMS=0.0; 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {	  
	  float binMean = (*iME)->getBinContent(b+1), binRMS = (*iME)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { avgMean = avgMean/float(nBins); avgRMS = sqrt(avgRMS/float(nBins)); }
	performanceSummary->setClusterSizeX(theMEdetID, avgMean, avgRMS);
      }
      if (theMEname.find("sizeY")!=string::npos) {
	int nBins=0; float avgMean=0.0, avgRMS=0.0; 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {	  
	  float binMean = (*iME)->getBinContent(b+1), binRMS = (*iME)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { avgMean = avgMean/float(nBins); avgRMS = sqrt(avgRMS/float(nBins)); }
	performanceSummary->setClusterSizeY(theMEdetID, avgMean, avgRMS);
      } 
      // from SiPixelMonitorRecHit
      if (theMEname.find("nRecHits")!=string::npos) {
	int nBins=0; float avgMean=0.0, avgRMS=0.0; 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {	  
	  float binMean = (*iME)->getBinContent(b+1), binRMS = (*iME)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { avgMean = avgMean/float(nBins); avgRMS = sqrt(avgRMS/float(nBins)); }
	performanceSummary->setNumberOfRecHits(theMEdetID, avgMean, avgRMS);
      } 
      // from SiPixelMonitorTrack
      if (theMEname.find("residualX")!=string::npos) {
	int nBins=0; float avgMean=0.0, avgRMS=0.0; 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {	  
	  float binMean = (*iME)->getBinContent(b+1), binRMS = (*iME)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { avgMean = avgMean/float(nBins); avgRMS = sqrt(avgRMS/float(nBins)); }
	performanceSummary->setResidualX(theMEdetID, avgMean, avgRMS);
      }
      if (theMEname.find("residualY")!=string::npos) {
	int nBins=0; float avgMean=0.0, avgRMS=0.0; 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {	  
	  float binMean = (*iME)->getBinContent(b+1), binRMS = (*iME)->getBinError(b+1); 
	  if (binMean!=0 && binRMS!=0) { nBins++; avgMean += binMean; avgRMS += pow(binRMS,2); }
	} 
	if (nBins>0) { avgMean = avgMean/float(nBins); avgRMS = sqrt(avgRMS/float(nBins)); }
	performanceSummary->setResidualY(theMEdetID, avgMean, avgRMS);
      } 
    }
  }
}


void SiPixelHistoricInfoDQMClient::fillPerformanceSummaryWithModuleMEs() const {
  for (map< uint32_t, vector<MonitorElement*> >::const_iterator iMEvec = mapOfdetIDtoMEs.begin(); 
       iMEvec!=mapOfdetIDtoMEs.end(); iMEvec++) {
    uint32_t theMEdetID = iMEvec->first;
    vector<MonitorElement*> theMEvector = iMEvec->second;
    
    if (printDebug_) { 
      cout << theMEdetID << ":"; 
      for (vector<MonitorElement*>::const_iterator iME = theMEvector.begin(); 
           iME!=theMEvector.end(); iME++) cout << (*iME)->getName() << ","; 
      cout << endl; 
    } 
    for (vector<MonitorElement*>::const_iterator iME = theMEvector.begin(); iME!=theMEvector.end(); iME++) {
      string theMEname = (*iME)->getName(); 

      // from SiPixelMonitorRawData
      if (theMEname.find("errorType")!=string::npos) { 
	for (int v=25; v<40; v++) {
	  int b = (*iME)->getTH1()->GetXaxis()->FindBin(v); 
	  performanceSummary->setRawDataErrorType(theMEdetID, v-25, (*iME)->getBinContent(b)); 
	}
      } 
      // from SiPixelMonitorDigi 
      if (theMEname.find("ndigis")!=string::npos && theMEname.find("FREQ")==string::npos) { 
    	performanceSummary->setNumberOfDigis(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      } 
      if (theMEname.find("adc")!=string::npos) { 
    	performanceSummary->setADC(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      // from SiPixelMonitorCluster
      if (theMEname.find("nclusters")!=string::npos) {
	performanceSummary->setNumberOfClusters(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("charge")!=string::npos) {
	performanceSummary->setClusterCharge(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("sizeX")!=string::npos) {
	performanceSummary->setClusterSizeX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("sizeY")!=string::npos) {
	performanceSummary->setClusterSizeY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      } 
      // from SiPixelMonitorRecHit
      if (theMEname.find("nRecHits")!=string::npos) {
	performanceSummary->setNumberOfRecHits(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      } 
      // from SiPixelMonitorTrack
      if (theMEname.find("residualX")!=string::npos) {
	performanceSummary->setResidualX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("residualY")!=string::npos) {
	performanceSummary->setResidualY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      } 
    }
  }
}


void SiPixelHistoricInfoDQMClient::writeDB() const {
  cout << "SiPixelHistoricInfoDQMClient::writeDB() for run "<< performanceSummary->getRunNumber() 
       <<" with "<< performanceSummary->getNumberOfEvents() <<" events" << endl; 
  
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
  else LogError("writeDB") << "service unavailable" << endl; 
}
