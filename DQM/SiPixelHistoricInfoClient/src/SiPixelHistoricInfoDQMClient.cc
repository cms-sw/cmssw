#include <memory>
#include <iostream>
#include <string>
#include <stdio.h>
#include <sys/time.h>

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoDQMClient.h"


using namespace edm;
using namespace std;


SiPixelHistoricInfoDQMClient::SiPixelHistoricInfoDQMClient(const ParameterSet& parameterSet) {   
  parameterSet_ = parameterSet;  
  printDebug_ = parameterSet_.getUntrackedParameter<bool>("printDebug", false); 
  writeHisto_ = parameterSet_.getUntrackedParameter<bool>("writeHisto", false); 
  outputDir_ = parameterSet_.getUntrackedParameter<string>("outputDir", ".");
}


SiPixelHistoricInfoDQMClient::~SiPixelHistoricInfoDQMClient() {}


void SiPixelHistoricInfoDQMClient::beginJob(const EventSetup& eventSetup) {
  dbe_ = Service<DQMStore>().operator->(); 
  dbe_->setVerbose(0); 
}


void SiPixelHistoricInfoDQMClient::analyze(const Event& event, const EventSetup& eventSetup) {}


void SiPixelHistoricInfoDQMClient::endJob() {
  vector<string> inputFiles = parameterSet_.getUntrackedParameter<vstring>("inputFiles");
  for (vector<string>::const_iterator iFile = inputFiles.begin(); 
       iFile!=inputFiles.end(); ++iFile) {
    dbe_->open(iFile->data());
    
    performanceSummary = new SiPixelPerformanceSummary();
    performanceSummary->clear(); 
    
    int runNmfrmFilename = atoi(iFile->substr(iFile->find("Run",0)+3,5).data()); 
    performanceSummary->setRunNumber(runNmfrmFilename); 

    retrieveMEs(); 
    fillPerformanceSummary();

    performanceSummary->print();
    writeDB(iFile->data()); 
  
    if (writeHisto_) {
      ostringstream endRunOutputFile; 
      endRunOutputFile << outputDir_ << "/SiPixelHistoricInfoDQMClient_" << runNmfrmFilename <<".root"; 
      dbe_->save(endRunOutputFile.str()); 
    } 
  }
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
  
  for (vector<string>::const_iterator ime=listOfMEsWithFullPath.begin(); 
       ime!=listOfMEsWithFullPath.end(); ime++) {
    uint32_t pathLength = (*ime).find(":",0);     
    string thePath = (*ime).substr(0, pathLength); 
    string allHists = (*ime).substr(pathLength+1); 
    
    if (thePath.find("Pixel",0)!=string::npos) { 
      if (thePath.find("Module_",0)!=string::npos || thePath.find("FED_",0)!=string::npos || 
          thePath.find("EventInfo",0)!=string::npos) {
      	uint histnameLength; 
      	do {
      	  histnameLength = allHists.find(",",0);
      	  string theHist;
      	  if (histnameLength!=string::npos) {
      	    theHist = allHists.substr(0, histnameLength);
      	    allHists.erase(0, histnameLength+1);
      	  }
      	  else theHist = allHists; 

      	  if (thePath.find("Module_",0)!=string::npos || thePath.find("FED_",0)!=string::npos) {
      	    string fullPathHist = thePath + "/" + theHist;      	  
	    MonitorElement* theMEpointer = dbe_->get(fullPathHist);
	  
	    SiPixelHistogramId hisIDmanager;
      	    string hisID;
      	    uint32_t localMEdetID;
      	    if (theMEpointer) {
      	      hisID = theMEpointer->getName();
      	      localMEdetID = hisIDmanager.getRawId(hisID);

      	      if (ClientPointersToModuleMEs.find(localMEdetID)==ClientPointersToModuleMEs.end()) {
      	    	vector<MonitorElement*> newMEvector;
      	    	newMEvector.push_back(theMEpointer);
      	    	ClientPointersToModuleMEs.insert(make_pair(localMEdetID, newMEvector));
      	      }
      	      else ((ClientPointersToModuleMEs.find(localMEdetID))->second).push_back(theMEpointer);
      	    }
      	  } 
	  else if (theHist.find("iEvent",0)!=string::npos || theHist.find("iRun",0)!=string::npos || 
	           theHist.find("processTimeStamp",0)!=string::npos) { 
      	    string fullPathHist = thePath + "/" + theHist;
      	    MonitorElement* theMEpointer = dbe_->get(fullPathHist); 
	    
	    if (theHist.find("iEvent",0)!=string::npos) {
              performanceSummary->setNumberOfEvents(theMEpointer->getIntValue()); 
	    }
	    if (theHist.find("iRun",0)!=string::npos) { 
              performanceSummary->setRunNumber(theMEpointer->getIntValue()); 
      	    }
	    if (theHist.find("processTimeStamp",0)!=string::npos) {
	      performanceSummary->setTimeValue((unsigned long long)theMEpointer->getFloatValue()); 
      	    } 
	  }
	} 
        while (histnameLength!=string::npos); 
      } 
    } 
  } 
} 


void SiPixelHistoricInfoDQMClient::fillPerformanceSummary() const {
  for (map< uint32_t, vector<MonitorElement*> >::const_iterator iMEvec=ClientPointersToModuleMEs.begin(); 
       iMEvec!=ClientPointersToModuleMEs.end(); iMEvec++) {
    uint32_t localMEdetID = iMEvec->first;
    vector<MonitorElement*> theMEvector = iMEvec->second;
    
    for (vector<MonitorElement*>::const_iterator iMEpntr = theMEvector.begin(); 
         iMEpntr!=theMEvector.end(); iMEpntr++) {
      string theMEname = (*iMEpntr)->getName();

// from SiPixelMonitorRawData
      if (theMEname.find("NErrors_siPixelDigis")!=string::npos) { 
	performanceSummary->setNumberOfRawDataErrors(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      } 
      if (theMEname.find("errorType_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {
	  float percentage = (*iMEpntr)->getBinContent(b+1)/float(nEventsInRun); 
	  performanceSummary->setRawDataErrorType(localMEdetID, b, percentage); 
	}
      } 
      if (theMEname.find("TBMType_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {
	  float percentage = (*iMEpntr)->getBinContent(b+1)/float(nEventsInRun); 
    	  performanceSummary->setTBMType(localMEdetID, b, percentage);
        }
      } 
      if (theMEname.find("TBMMessage_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {
	  float percentage = (*iMEpntr)->getBinContent(b+1)/float(nEventsInRun); 
    	  performanceSummary->setTBMMessage(localMEdetID, b, percentage);
        }
      } 
      if (theMEname.find("fullType_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {
	  float percentage = (*iMEpntr)->getBinContent(b+1)/float(nEventsInRun); 
    	  performanceSummary->setFEDfullType(localMEdetID, b, percentage);
        }
      } 
      if (theMEname.find("chanNmbr_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iMEpntr)->getNbinsX(); b++) {
	  float percentage = (*iMEpntr)->getBinContent(b+1)/float(nEventsInRun); 
    	  performanceSummary->setFEDtimeoutChannel(localMEdetID, b, percentage);
        }
      }  
      if (theMEname.find("evtSize_siPixelDigis")!=string::npos) { 
        performanceSummary->setSLinkErrSize(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS()); 
      } 
      if (theMEname.find("linkId_siPixelDigis")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
	performanceSummary->setFEDmaxErrLink(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("ROCId_siPixelDigis")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErr36ROC(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("Type36Hitmap_siPixelDigis")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH2F()->ProjectionY()->GetMaximumBin(); 
        performanceSummary->setmaxErr36ROC(localMEdetID, (*iMEpntr)->getTH2F()->ProjectionY()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("DCOLId_siPixelDigis")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErrDCol(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("PXId_siPixelDigis")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErrPixelRow(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("ROCNmbr_siPixelDigis")!=string::npos) { 
	int maxBin = (*iMEpntr)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErr38ROC(localMEdetID, (*iMEpntr)->getTH1F()->GetBinCenter(maxBin)); 
      }
// from SiPixelMonitorDigi 
      if (theMEname.find("ndigis_siPixelDigis")!=string::npos) { 
    	performanceSummary->setNumberOfDigis(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
	// to debug 
        if (printDebug_) { 
          if (localMEdetID==302121992 || 
	      localMEdetID==302056720 || 
	      localMEdetID==352390664) cout << "module "<< localMEdetID <<" has nDigis "
	                                    << (*iMEpntr)->getMean() <<" +- "<< (*iMEpntr)->getRMS() << endl; 
        }
      } 
      if (theMEname.find("adc_siPixelDigis")!=string::npos) { 
    	performanceSummary->setADC(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      } 
      if (theMEname.find("hitmap_siPixelDigis")!=string::npos) { 
    	int nEmptyCells=0, nHotCells=0; // 4 pixels per bin
	for (int xBin=0; xBin<(*iMEpntr)->getNbinsX(); xBin++) { 
	  for (int yBin=0; yBin<(*iMEpntr)->getNbinsY(); yBin++) { 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)>0.01*(*iMEpntr)->getEntries()) nHotCells++; 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	  } 
	} 
	performanceSummary->setDigimapHotCold(localMEdetID, nHotCells/float(nEventsInRun), // the higher the worse
	                                                    nEmptyCells/float(nEventsInRun)); // the lower the worse
      } 
// from SiPixelMonitorCluster
      if (theMEname.find("nclusters_siPixelClusters")!=string::npos) {
    	performanceSummary->setNumberOfClusters(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("charge_siPixelClusters")!=string::npos) {
    	performanceSummary->setClusterCharge(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("sizeX_siPixelClusters")!=string::npos) {
    	performanceSummary->setClusterSizeX(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("sizeY_siPixelClusters")!=string::npos) {
    	performanceSummary->setClusterSizeY(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("hitmap_siPixelClusters")!=string::npos) {
    	int nEmptyCells=0, nHotCells=0; // 4 pixels per bin
	for (int xBin=0; xBin<(*iMEpntr)->getNbinsX(); xBin++) { 
	  for (int yBin=0; yBin<(*iMEpntr)->getNbinsY(); yBin++) { 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)>0.01*(*iMEpntr)->getEntries()) nHotCells++; 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	  } 
	} 
	performanceSummary->setClustermapHotCold(localMEdetID, nHotCells/float(nEventsInRun), // the higher the worse
	                                                       nEmptyCells/float(nEventsInRun)); // the lower the worse
      }
// from SiPixelMonitorRecHit
      if (theMEname.find("nRecHits_siPixelRecHits")!=string::npos) {
    	performanceSummary->setNumberOfRecHits(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("ClustX_siPixelRecHits")!=string::npos) {
    	performanceSummary->setRecHitMatchedClusterSizeX(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("ClustY_siPixelRecHits")!=string::npos) {
    	performanceSummary->setRecHitMatchedClusterSizeY(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("xypos_siPixelRecHits")!=string::npos) {
    	int nEmptyCells=0, nHotCells=0; // 4 pixels per bin
	for (int xBin=0; xBin<(*iMEpntr)->getNbinsX(); xBin++) { 
	  for (int yBin=0; yBin<(*iMEpntr)->getNbinsY(); yBin++) { 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)>0.01*(*iMEpntr)->getEntries()) nHotCells++; 
	    if ((*iMEpntr)->getBinContent(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	  } 
	} 
	performanceSummary->setRecHitmapHotCold(localMEdetID, nHotCells/float(nEventsInRun), // the higher the worse
	                                                      nEmptyCells/float(nEventsInRun)); // the lower the worse
      }
// from SiPixelMonitorTrack
      if (theMEname.find("residualX_siPixelTracks")!=string::npos) {
    	performanceSummary->setResidualX(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      }
      if (theMEname.find("residualY_siPixelTracks")!=string::npos) {
    	performanceSummary->setResidualY(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      } 
    }
  }
}


void SiPixelHistoricInfoDQMClient::writeDB(string filename) const {
  cout << endl << "SiPixelHistoricInfoDQMClient::writeDB() for File "<< filename << endl; 
  
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

  cout << "SiPixelHistoricInfoDQMClient::writeDB() finished" << endl; 
}


void SiPixelHistoricInfoDQMClient::saveFile(string filename) const {
  dbe_->save(filename); 
}
