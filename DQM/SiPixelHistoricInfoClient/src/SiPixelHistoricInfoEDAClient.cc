#include <memory>
#include <iostream>
#include <string>
#include <stdio.h>
#include <sys/time.h>

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoEDAClient.h"


using namespace std;
using namespace edm;
// using namespace cgicc;
// using namespace xcept;


SiPixelHistoricInfoEDAClient::SiPixelHistoricInfoEDAClient(const ParameterSet& parameterSet) { 
  dbe_ = Service<DQMStore>().operator->(); 
  dbe_->setVerbose(0); 
  
  parameterSet_ = parameterSet;  
  printDebug_ = parameterSet_.getUntrackedParameter<bool>("printDebug", false); 
  writeHisto_ = parameterSet_.getUntrackedParameter<bool>("writeHisto", true); 
  outputDir_ = parameterSet_.getUntrackedParameter<string>("outputDir", ".");
}


/*// mui_ = new DQMOldReceiver();
  // // sipixelWebInterface_ = new SiPixelWebInterface("dummy","dummy",&mui_); 
  // webInterface_ = new SiPixelHistoricInfoWebInterface(getContextURL(), getApplicationURL(), &mui_);
  // defaultWebPageCreated_ = false; 

void SiPixelHistoricInfoEDAClient::defaultWebPage(xgi::Input* in, xgi::Output* out) {
  if (!defaultPageCreated_) {
    static const int BUF_SIZE = 256; 
    ifstream fin("loader.html", ios::in);
    if (!fin) {
      cerr << "loader.html could not be opened!" << endl;
      return;
    }
    char buf[BUF_SIZE];
    ostringstream html_dump;
    while (fin.getline(buf, BUF_SIZE,'\n')) html_dump << buf << endl; 
    fin.close();
   *out << html_dump.str() << endl;
   
    defaultWebPageCreated_ = true;
  }  
  webInterface_->handleEDARequest(in, out);
} */


SiPixelHistoricInfoEDAClient::~SiPixelHistoricInfoEDAClient() {}


void SiPixelHistoricInfoEDAClient::beginJob(const EventSetup& eventSetup) {}


void SiPixelHistoricInfoEDAClient::beginRun(const Run& run, const EventSetup& eventSetup) {
  performanceSummary = new SiPixelPerformanceSummary();
  performanceSummary->clear(); 
  performanceSummary->setRunNumber(run.run());

  nEventsInRun = 0; 
  firstEventInRun = true; 
}


void SiPixelHistoricInfoEDAClient::analyze(const Event& event, const EventSetup& eventSetup) {
  nEventsInRun++; 
  if (firstEventInRun) {
    firstEventInRun = false; 

    performanceSummary->setTimeValue(event.time().value());
  } 
}


void SiPixelHistoricInfoEDAClient::endRun(const Run& run, const EventSetup& eventSetup) {
  performanceSummary->setNumberOfEvents(nEventsInRun);

  retrievePointersToModuleMEs();
  fillSummaryObjects();

  performanceSummary->print();

  cout << "SiPixelHistoricInfoEDAClient::writetoDB() run = "<< run.run() << endl; 
  writetoDB(); 
  
  if (writeHisto_) {
    ostringstream endRunOutputFile; 
    endRunOutputFile << outputDir_ << "/SiPixelHistoricInfoEDAClient_" << run.run() <<".root"; 
    dbe_->save(endRunOutputFile.str()); 
  }
}


void SiPixelHistoricInfoEDAClient::endJob() {
  if (writeHisto_) {
    ostringstream endJobOutputFile; 
    endJobOutputFile << outputDir_ << "/SiPixelHistoricInfoEDAClient_endJob.root"; 
    dbe_->save(endJobOutputFile.str());
  }
}


void SiPixelHistoricInfoEDAClient::retrievePointersToModuleMEs() {
  ClientPointersToModuleMEs.clear(); 
  
  vector<string> listOfMEsWithFullPath;
  dbe_->getContents(listOfMEsWithFullPath);
    
  for (vector<string>::const_iterator iME=listOfMEsWithFullPath.begin(); 
       iME!=listOfMEsWithFullPath.end(); iME++) {
    uint32_t pathLength = (*iME).find(":",0);     
    string thePath = (*iME).substr(0, pathLength); 
    string allHists = (*iME).substr(pathLength+1); 
    
    if (thePath.find("Pixel",0)!=string::npos) { 
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
      while (histnameLength!=string::npos); 
    } 
  } 
}


void SiPixelHistoricInfoEDAClient::fillSummaryObjects() const {
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


void SiPixelHistoricInfoEDAClient::writetoDB() const {  
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
  else LogError("writetoDB") << "service unavailable" << endl; 

  cout << "SiPixelHistoricInfoEDAClient::writetoDB() finished" << endl; 
}


void SiPixelHistoricInfoEDAClient::savetoFile(string filename) const {
  dbe_->save(filename); 
}
