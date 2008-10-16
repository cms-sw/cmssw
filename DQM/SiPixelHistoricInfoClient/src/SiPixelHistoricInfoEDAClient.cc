#include <memory>
#include <iostream>
#include <string>
#include <stdio.h>
#include <sys/time.h>

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
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


/* mui_ = new DQMOldReceiver();
   webInterface_ = new SiPixelHistoricInfoWebInterface(getContextURL(), getApplicationURL(), &mui_);
   defaultWebPageCreated_ = false; 

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
    while (fin.getline(buf, BUF_SIZE, '\n')) html_dump << buf << endl; 
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

  retrieveMEs();
  fillPerformanceSummary();
  writeDB(); 
  
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


void SiPixelHistoricInfoEDAClient::retrieveMEs() {
  mapOfdetIDtoMEs.clear(); 
  
  vector<string> listOfMEswithFullPath;
  dbe_->getContents(listOfMEswithFullPath);
    
  for (vector<string>::const_iterator iMEstr = listOfMEswithFullPath.begin(); 
       iMEstr!=listOfMEswithFullPath.end(); iMEstr++) {
    uint32_t pathLength = iMEstr->find(":",0);     
    string thePath = iMEstr->substr(0, pathLength); 
    string allHists = iMEstr->substr(pathLength+1); 
        
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
        MonitorElement* newME = dbe_->get(fullPathHist);
        if (newME) {
          uint32_t newMEdetID = histogramManager.getRawId(newME->getName());

          if (mapOfdetIDtoMEs.find(newMEdetID)==mapOfdetIDtoMEs.end()) {
            vector<MonitorElement*> newMEvector;
            newMEvector.push_back(newME);
            mapOfdetIDtoMEs.insert(make_pair(newMEdetID, newMEvector));
          }
          else ((mapOfdetIDtoMEs.find(newMEdetID))->second).push_back(newME);
        }
      } 
      while (histnameLength!=string::npos); 
    } 
  } 
}


void SiPixelHistoricInfoEDAClient::fillPerformanceSummary() const {
  for (map< uint32_t, vector<MonitorElement*> >::const_iterator iMEvec=mapOfdetIDtoMEs.begin(); 
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
      if (theMEname.find("NErrors_siPixelDigis")!=string::npos) { 
	performanceSummary->setNumberOfRawDataErrors(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      } 
      if (theMEname.find("errorType_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {
	  performanceSummary->setRawDataErrorType(theMEdetID, b, (*iME)->getBinContent(b+1)); 
	}
      } 
      if (theMEname.find("TBMType_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {
    	  performanceSummary->setTBMType(theMEdetID, b, (*iME)->getBinContent(b+1));
        }
      } 
      if (theMEname.find("TBMMessage_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {
    	  performanceSummary->setTBMMessage(theMEdetID, b, (*iME)->getBinContent(b+1));
        }
      } 
      if (theMEname.find("fullType_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {
    	  performanceSummary->setFEDfullType(theMEdetID, b, (*iME)->getBinContent(b+1));
        }
      } 
      if (theMEname.find("chanNmbr_siPixelDigis")!=string::npos) { 
	for (int b=0; b<(*iME)->getNbinsX(); b++) {
    	  performanceSummary->setFEDtimeoutChannel(theMEdetID, b, (*iME)->getBinContent(b+1));
        }
      }  
      if (theMEname.find("evtSize_siPixelDigis")!=string::npos) { 
        performanceSummary->setSLinkErrSize(theMEdetID, (*iME)->getMean(), (*iME)->getRMS()); 
      } 
      if (theMEname.find("linkId_siPixelDigis")!=string::npos) { 
	int maxBin = (*iME)->getTH1F()->GetMaximumBin(); 
	performanceSummary->setFEDmaxErrLink(theMEdetID, (*iME)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("ROCId_siPixelDigis")!=string::npos) { 
	int maxBin = (*iME)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErr36ROC(theMEdetID, (*iME)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("Type36Hitmap_siPixelDigis")!=string::npos) { 
	int maxBin = (*iME)->getTH2F()->ProjectionY()->GetMaximumBin(); 
        performanceSummary->setmaxErr36ROC(theMEdetID, (*iME)->getTH2F()->ProjectionY()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("DCOLId_siPixelDigis")!=string::npos) { 
	int maxBin = (*iME)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErrDCol(theMEdetID, (*iME)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("PXId_siPixelDigis")!=string::npos) { 
	int maxBin = (*iME)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErrPixelRow(theMEdetID, (*iME)->getTH1F()->GetBinCenter(maxBin)); 
      }
      if (theMEname.find("ROCNmbr_siPixelDigis")!=string::npos) { 
	int maxBin = (*iME)->getTH1F()->GetMaximumBin(); 
        performanceSummary->setmaxErr38ROC(theMEdetID, (*iME)->getTH1F()->GetBinCenter(maxBin)); 
      }
// from SiPixelMonitorDigi 
      if (theMEname.find("ndigis_siPixelDigis")!=string::npos) { 
    	performanceSummary->setNumberOfDigis(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      } 
      if (theMEname.find("adc_siPixelDigis")!=string::npos) { 
    	performanceSummary->setADC(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      } 
      if (theMEname.find("hitmap_siPixelDigis")!=string::npos) { 
    	int nEmptyCells=0, nHotCells=0; // 4 pixels per bin
	for (int xBin=0; xBin<(*iME)->getNbinsX(); xBin++) { 
	  for (int yBin=0; yBin<(*iME)->getNbinsY(); yBin++) { 
	    if ((*iME)->getBinContent(xBin+1, yBin+1)>0.01*(*iME)->getEntries()) nHotCells++; 
	    if ((*iME)->getBinContent(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	  } 
	} 
	performanceSummary->setDigimapHotCold(theMEdetID, nHotCells, nEmptyCells); 
      } 
// from SiPixelMonitorCluster
      if (theMEname.find("nclusters_siPixelClusters")!=string::npos) {
    	performanceSummary->setNumberOfClusters(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("charge_siPixelClusters")!=string::npos) {
    	performanceSummary->setClusterCharge(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("sizeX_siPixelClusters")!=string::npos) {
    	performanceSummary->setClusterSizeX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("sizeY_siPixelClusters")!=string::npos) {
    	performanceSummary->setClusterSizeY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("hitmap_siPixelClusters")!=string::npos) {
    	int nEmptyCells=0, nHotCells=0; // 4 pixels per bin
	for (int xBin=0; xBin<(*iME)->getNbinsX(); xBin++) { 
	  for (int yBin=0; yBin<(*iME)->getNbinsY(); yBin++) { 
	    if ((*iME)->getBinContent(xBin+1, yBin+1)>0.01*(*iME)->getEntries()) nHotCells++; 
	    if ((*iME)->getBinContent(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	  } 
	} 
	performanceSummary->setClustermapHotCold(theMEdetID, nHotCells, nEmptyCells); 
      }
// from SiPixelMonitorRecHit
      if (theMEname.find("nRecHits_siPixelRecHits")!=string::npos) {
    	performanceSummary->setNumberOfRecHits(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("ClustX_siPixelRecHits")!=string::npos) {
    	performanceSummary->setRecHitMatchedClusterSizeX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("ClustY_siPixelRecHits")!=string::npos) {
    	performanceSummary->setRecHitMatchedClusterSizeY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("xypos_siPixelRecHits")!=string::npos) {
    	int nEmptyCells=0, nHotCells=0; // 4 pixels per bin
	for (int xBin=0; xBin<(*iME)->getNbinsX(); xBin++) { 
	  for (int yBin=0; yBin<(*iME)->getNbinsY(); yBin++) { 
	    if ((*iME)->getBinContent(xBin+1, yBin+1)>0.01*(*iME)->getEntries()) nHotCells++; 
	    if ((*iME)->getBinContent(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	  } 
	} 
	performanceSummary->setRecHitmapHotCold(theMEdetID, nHotCells, nEmptyCells); 
      }
// from SiPixelMonitorTrack
      if (theMEname.find("residualX_siPixelTracks")!=string::npos) {
    	performanceSummary->setResidualX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      }
      if (theMEname.find("residualY_siPixelTracks")!=string::npos) {
    	performanceSummary->setResidualY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS());
      } 
    }
  }
}


void SiPixelHistoricInfoEDAClient::writeDB() const {  
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
