#include <memory>
#include <math.h>
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
  
  parameterSet_ = parameterSet;  
  
  printDebug_ = parameterSet_.getUntrackedParameter<bool>("printDebug", false); 
  writeHisto_ = parameterSet_.getUntrackedParameter<bool>("writeHisto", false); 
  
  outputDir_ = parameterSet_.getUntrackedParameter<string>("outputDir", ".");
}


SiPixelHistoricInfoEDAClient::~SiPixelHistoricInfoEDAClient() {}


void SiPixelHistoricInfoEDAClient::beginJob(const EventSetup& eventSetup) {}


void SiPixelHistoricInfoEDAClient::beginRun(const Run& run, const EventSetup& eventSetup) {
  performanceSummary = new SiPixelPerformanceSummary(); 
  performanceSummary->setRunNumber(run.run());

  nEventsInRun = 0; 
  firstEventInRun = true; 
}


void SiPixelHistoricInfoEDAClient::beginLuminosityBlock(const edm::LuminosityBlock& luminosityBlock, 
                                                        const edm::EventSetup& eventSetup) {
  performanceSummary->setLuminosityBlock(luminosityBlock.luminosityBlock());
}


void SiPixelHistoricInfoEDAClient::analyze(const Event& event, const EventSetup& eventSetup) {
  nEventsInRun++; 
  if (firstEventInRun) { 
    firstEventInRun = false; 

    performanceSummary->setTimeStamp(event.time().value()); 
  }
}


void SiPixelHistoricInfoEDAClient::endLuminosityBlock(const edm::LuminosityBlock& luminosityBlock, 
                                                      const edm::EventSetup& eventSetup) {}


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
    if (printDebug_) cout << iMEstr->data() << endl; 
      
    size_t pathLength = iMEstr->find(":",0);     
    string thePath = iMEstr->substr(0, pathLength); 
    string allHists = iMEstr->substr(pathLength+1); 
        
    if (thePath.find("Pixel",0)!=string::npos) { 
      if (thePath.find("Track",0)!=string::npos) { // for Pixel/Tracks, Pixel/Clusters/On,OffTrack
   	                  		uint32_t newMEdetID = 77; 
	if (thePath.find("On", 0)!=string::npos) newMEdetID = 78; 
	if (thePath.find("Off",0)!=string::npos) newMEdetID = 79; 
	
	size_t histnameLength;
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
      else {
        size_t histnameLength; 
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
}


void SiPixelHistoricInfoEDAClient::fillPerformanceSummary() const {
  for (map< uint32_t, vector<MonitorElement*> >::const_iterator iMEvec=mapOfdetIDtoMEs.begin(); 
       iMEvec!=mapOfdetIDtoMEs.end(); iMEvec++) {
    uint32_t theMEdetID = iMEvec->first;
    vector<MonitorElement*> theMEvector = iMEvec->second;
    
    if (printDebug_) { 
      cout << theMEdetID << ":"; for (vector<MonitorElement*>::const_iterator iME = theMEvector.begin(); 
      				      iME!=theMEvector.end(); iME++) cout << (*iME)->getName() << ","; cout << endl; 
    }     
    for (vector<MonitorElement*>::const_iterator iME = theMEvector.begin(); iME!=theMEvector.end(); iME++) {
      string theMEname = (*iME)->getName();

      
      if (theMEdetID<50) {
        // from SiPixelMonitorRawData
      	if (theMEname.find("errorType_siPixelDigis")!=string::npos) { 
	  for (int v=25; v<40; v++) {
	    int b = (*iME)->getTH1()->GetXaxis()->FindBin(v); 
	    performanceSummary->setRawDataErrorType(theMEdetID, v-25, (*iME)->getBinContent(b)); 
	  }
      	} 
      } 
      // from SiPixelMonitorTrack
      else if (theMEdetID==77) {
      	if (theMEname.find("ntracks")!=string::npos && theMEname.find("InPixVol")==string::npos) { 
      	  float trkMean=0.0, trkRMS=0.0; 
	  if ((*iME)->getBinContent(1)>0.0) {
      	    trkMean = float((*iME)->getBinContent(3))/(*iME)->getBinContent(1); // Barrel regionID: 80
	    trkRMS = sqrt(trkMean*(trkMean+1.0)/(*iME)->getBinContent(1)); 
      	    performanceSummary->setFractionOfTracks(80, trkMean, trkRMS); 

      	    trkMean = float((*iME)->getBinContent(4))/(*iME)->getBinContent(1); // Endcap regionID: 81
	    trkRMS = sqrt(trkMean*(trkMean+1.0)/(*iME)->getBinContent(1)); 
      	    performanceSummary->setFractionOfTracks(81, trkMean, trkRMS); 
      	  }
	}
      } 
      else if (theMEdetID==78) { // OnTrack
        if (theMEname.find("nclusters_siPixelClusters")!=string::npos) { 
          performanceSummary->setNumberOfOnTrackClusters(80, (*iME)->getBinContent(2)); 
          performanceSummary->setNumberOfOnTrackClusters(81, (*iME)->getBinContent(3)); 
        }
        if (theMEname.find("charge_siPixelClusters_Barrel")!=string::npos) {
          performanceSummary->setClusterChargeOnTrack(80, (*iME)->getMean(), (*iME)->getRMS()); 
        }
        if (theMEname.find("charge_siPixelClusters_Endcap")!=string::npos) {
          performanceSummary->setClusterChargeOnTrack(81, (*iME)->getMean(), (*iME)->getRMS()); 
        }
        if (theMEname.find("size_siPixelClusters_Barrel")!=string::npos) {
          performanceSummary->setClusterSizeOnTrack(80, (*iME)->getMean(), (*iME)->getRMS()); 
        }
        if (theMEname.find("size_siPixelClusters_Endcap")!=string::npos) {
          performanceSummary->setClusterSizeOnTrack(81, (*iME)->getMean(), (*iME)->getRMS()); 
        }
      }
      else if (theMEdetID==79) { // OffTrack
        if (theMEname.find("nclusters_siPixelClusters")!=string::npos) { 
          performanceSummary->setNumberOfOffTrackClusters(80, (*iME)->getBinContent(2)); 
          performanceSummary->setNumberOfOffTrackClusters(81, (*iME)->getBinContent(3)); 
        }
        if (theMEname.find("charge_siPixelClusters_Barrel")!=string::npos) {
          performanceSummary->setClusterChargeOffTrack(80, (*iME)->getMean(), (*iME)->getRMS()); 
        }
        if (theMEname.find("charge_siPixelClusters_Endcap")!=string::npos) {
          performanceSummary->setClusterChargeOffTrack(81, (*iME)->getMean(), (*iME)->getRMS()); 
        }
        if (theMEname.find("size_siPixelClusters_Barrel")!=string::npos) {
          performanceSummary->setClusterSizeOffTrack(80, (*iME)->getMean(), (*iME)->getRMS()); 
        }
        if (theMEname.find("size_siPixelClusters_Endcap")!=string::npos) {
          performanceSummary->setClusterSizeOffTrack(81, (*iME)->getMean(), (*iME)->getRMS()); 
        }
      }
      else {
      	// from SiPixelMonitorDigi 
      	if (theMEname.find("ndigis_siPixelDigis")!=string::npos) { 
    	  performanceSummary->setNumberOfDigis(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	} 
      	if (theMEname.find("adc_siPixelDigis")!=string::npos) { 
    	  performanceSummary->setADC(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	} 
      	// from SiPixelMonitorCluster
      	if (theMEname.find("nclusters_siPixelClusters")!=string::npos) {
    	  performanceSummary->setNumberOfClusters(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	}
      	if (theMEname.find("charge_siPixelClusters")!=string::npos) {
    	  performanceSummary->setClusterCharge(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	} 
      	if (theMEname.find("size_siPixelClusters")!=string::npos) {
    	  performanceSummary->setClusterSize(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	} 
      	if (theMEname.find("sizeX_siPixelClusters")!=string::npos) {
    	  performanceSummary->setClusterSizeX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	}
      	if (theMEname.find("sizeY_siPixelClusters")!=string::npos) {
    	  performanceSummary->setClusterSizeY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	}
      	// from SiPixelMonitorRecHit
      	if (theMEname.find("nRecHits_siPixelRecHits")!=string::npos) {
    	  performanceSummary->setNumberOfRecHits(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	}
      	// from SiPixelMonitorTrack
      	if (theMEname.find("residualX_siPixelTrackResiduals")!=string::npos) {
    	  performanceSummary->setResidualX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	}
      	if (theMEname.find("residualY_siPixelTrackResiduals")!=string::npos) {
    	  performanceSummary->setResidualY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), (*iME)->getEntries()==0.0);
      	} 
      	// temporary solutions
      	if (theMEname.find("hitmap_siPixelClusters")!=std::string::npos || 
      	    theMEname.find("hitmap_siPixelDigis")!=std::string::npos) { // if digi map exists, use it; else use cluster map
	  int nNoisyCells=0, nEmptyCells=0;				//		       not use xypos_siPixelRecHits
	  for (int xBin=0; xBin<(*iME)->getNbinsX(); xBin++) {  	// 4-1 pixel per bin
	    for (int yBin=0; yBin<(*iME)->getNbinsY(); yBin++) { 
	      if ((*iME)->getBinContent(xBin+1, yBin+1)>0.01*(*iME)->getEntries()) nNoisyCells++; 
	      if ((*iME)->getBinContent(xBin+1, yBin+1)==.0 && (*iME)->getBinError(xBin+1, yBin+1)==0.0) nEmptyCells++; 
	    } 
	  } 
      	  performanceSummary->setNumberOfNoisCells(theMEdetID, float(nNoisyCells)); 
      	  performanceSummary->setNumberOfDeadCells(theMEdetID, float(nEmptyCells)); 
      	} 
      	// performanceSummary->setNumberOfPixelHitsInTrackFit(theMEdetId, float(nPixelHits)); 
      }
    }
  }
}


void SiPixelHistoricInfoEDAClient::writeDB() const {  
  if (printDebug_) performanceSummary->printAll(); 
  else performanceSummary->print(); 
  cout << "SiPixelHistoricInfoEDAClient::writeDB()" << endl; 

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
