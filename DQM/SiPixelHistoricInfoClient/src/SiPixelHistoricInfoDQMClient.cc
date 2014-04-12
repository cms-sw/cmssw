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


void SiPixelHistoricInfoDQMClient::beginJob() {
  dbe_ = Service<DQMStore>().operator->(); 
}


void SiPixelHistoricInfoDQMClient::endRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  for (vector<string>::const_iterator iFile = inputFiles_.begin(); iFile!=inputFiles_.end(); ++iFile) {
    unsigned int runNumber=0; 
    if (iFile->find("R0000",0)!=string::npos) runNumber = atoi(iFile->substr(iFile->find("R0000",0)+5,5).data()); 
    else if (iFile->find("Run",0)!=string::npos) runNumber = atoi(iFile->substr(iFile->find("Run",0)+3,5).data()); 
    else cout << "run number cannot be extracted from file name" << endl; 
    
    if (runNumber==run.run()) {    
      dbe_->open(iFile->data(), true);
    
      performanceSummary = new SiPixelPerformanceSummary();   
      performanceSummary->setRunNumber(runNumber); 

      retrieveMEs(); 
      fillPerformanceSummary(); 
      writeDB(); 
  
      if (writeHisto_) {
        ostringstream endRunOutputFile; 
        endRunOutputFile << outputDir_<<"/SiPixelHistoricInfoDQMClient_"<< runNumber <<".root"; 
        dbe_->save(endRunOutputFile.str()); 
      }
    }
  }
}


void SiPixelHistoricInfoDQMClient::endJob() {
  if (writeHisto_) {
    ostringstream endJobOutputFile; 
    endJobOutputFile << outputDir_ << "/SiPixelHistoricInfoDQMClient_endJob.root"; 
    dbe_->save(endJobOutputFile.str());
  }
}


void SiPixelHistoricInfoDQMClient::retrieveMEs() {
  mapOfdetIDtoMEs.clear(); 
  bool noModules = true, noSummary = true; 

  vector<string> listOfMEswithFullPath;
  dbe_->getContents(listOfMEswithFullPath);

  for (vector<string>::const_iterator iMEstr = listOfMEswithFullPath.begin(); 
       iMEstr!=listOfMEswithFullPath.end(); iMEstr++) {
    if (printDebug_) cout << iMEstr->data() << endl; 
      
    size_t pathLength = iMEstr->find(":",0);     
    string thePath = iMEstr->substr(0, pathLength); 
    string allHists = iMEstr->substr(pathLength+1); 

    if (thePath.find("Pixel",0)!=string::npos) { 
      if (thePath.find("FED_",0)!=string::npos) {
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
            else (mapOfdetIDtoMEs.find(newMEdetID)->second).push_back(newME);
          } 
        } 
        while (histnameLength!=string::npos); 
      } 
      if (thePath.find("Module_",0)!=string::npos) {
	if (!useSummary_) {
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
  	      else (mapOfdetIDtoMEs.find(newMEdetID)->second).push_back(newME);
  	    } 
  	  } 
  	  while(histnameLength!=string::npos); 
	}
	if (noModules) noModules = false; 
      } 
      else if( (thePath.find("Layer_",0)!=string::npos || 
                (thePath.find("Disk_",0)!=string::npos && thePath.find("Panel_",0)==string::npos)) ) {
	if (useSummary_) {
   	  uint32_t newMEdetID = getSummaryRegionID(thePath); 
	  if (printDebug_) cout << thePath.data() <<" in region "<< newMEdetID << endl; 	  
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
	if (noSummary) noSummary = false; 
      }
      if (thePath.find("Track",0)!=string::npos) { // for Pixel/Tracks, Pixel/Clusters/OnTrack, Pixel/Clusters/OffTrack
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
      if (thePath.find("EventInfo",0)!=string::npos) { 
      	size_t histnameLength; 
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
	  if (theHist.find("iLumiSection",0)!=string::npos) { 
	    string fullPathHist = thePath + "/" + theHist;
      	    MonitorElement* strME = dbe_->get(fullPathHist); 	  
	    if (strME) performanceSummary->setLuminosityBlock(strME->getIntValue());
	  }
	  if (theHist.find("processedEvents",0)!=string::npos) { 
	    string fullPathHist = thePath + "/" + theHist;
      	    MonitorElement* strME = dbe_->get(fullPathHist); 	  
	    if (strME) performanceSummary->setNumberOfEvents(strME->getIntValue()); 
	  } 
	  if (theHist.find("eventTimeStamp",0)!=string::npos) { 
	    string fullPathHist = thePath + "/" + theHist;
      	    MonitorElement* strME = dbe_->get(fullPathHist); 	  
	    if (strME) performanceSummary->setTimeStamp((unsigned long long)strME->getFloatValue());
	  }
	} 
        while (histnameLength!=string::npos); 
      } 
    }     
  } 
  if ( useSummary_ && noSummary) cout << endl << "use "<<"summary MEs but NO "<<"summary MEs in the input file" << endl << endl;
  if (!useSummary_ && noModules) cout << endl << "use "<< "module MEs but NO "<< "module MEs in the input file" << endl << endl;
} 


uint32_t SiPixelHistoricInfoDQMClient::getSummaryRegionID(string thePath) const {
  uint32_t regionID = 666; 
       if (thePath.find("Ladder",0)==string::npos && thePath.find("Blade",0)==string::npos) { 
    regionID = 100; 
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
  } 
  else if (thePath.find("Ladder",0)!=string::npos || thePath.find("Blade",0)!=string::npos) {
    regionID = 1000; 
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
  }
  return regionID; 
}


void SiPixelHistoricInfoDQMClient::getSummaryMEmeanRMSnBins(vector<MonitorElement*>::const_iterator iME, 
                                                            float& mean, float& RMS, float& emPtn) const {
  int nBins=0; for (int b=0; b<(*iME)->getNbinsX(); b++) { 
    float binMean = (*iME)->getBinContent(b+1); 
    float binRMS  = (*iME)->getBinError(b+1); 
    if (binMean!=0.0 || binRMS!=0.0) { nBins++; mean += binMean; RMS += pow(binRMS,2); } 
  } 
  if (nBins>0) { 
    mean = mean/float(nBins); 
    RMS = sqrt(RMS/float(nBins)); // emPtn = proportion of empty modules in a summary ME
  } 
  if ((*iME)->getNbinsX()>0) emPtn = 1.0 - float(nBins)/float((*iME)->getNbinsX()); 
}


void SiPixelHistoricInfoDQMClient::fillPerformanceSummary() const {
  for (map< uint32_t, vector<MonitorElement*> >::const_iterator iMEvec = mapOfdetIDtoMEs.begin(); 
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
      	if (useSummary_) { 
      	  if (theMEname.find("SUM")!=string::npos) {
	    // from SiPixelMonitorDigi 
      	    if (theMEname.find("ndigis")!=string::npos && theMEname.find("FREQ")==string::npos) { 
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      getSummaryMEmeanRMSnBins(iME, avgMean, avgRMS, emPtn); 
	      performanceSummary->setNumberOfDigis(theMEdetID, avgMean, avgRMS, emPtn);       
      	    } 
      	    if (theMEname.find("adc")!=string::npos) { 
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      getSummaryMEmeanRMSnBins(iME, avgMean, avgRMS, emPtn); 
	      performanceSummary->setADC(theMEdetID, avgMean, avgRMS, emPtn); 
      	    } 
      	    // from SiPixelMonitorCluster
      	    if (theMEname.find("nclusters")!=string::npos) {
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      getSummaryMEmeanRMSnBins(iME, avgMean, avgRMS, emPtn); 
	      performanceSummary->setNumberOfClusters(theMEdetID, avgMean, avgRMS, emPtn);
      	    }
      	    if (theMEname.find("charge")!=string::npos) {
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      getSummaryMEmeanRMSnBins(iME, avgMean, avgRMS, emPtn); 
	      performanceSummary->setClusterCharge(theMEdetID, avgMean, avgRMS, emPtn);
      	    } 
      	    if (theMEname.find("size")!=string::npos) {
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      getSummaryMEmeanRMSnBins(iME, avgMean, avgRMS, emPtn); 
	      performanceSummary->setClusterSize(theMEdetID, avgMean, avgRMS, emPtn);
      	    } 
      	    if (theMEname.find("sizeX")!=string::npos) {
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      getSummaryMEmeanRMSnBins(iME, avgMean, avgRMS, emPtn); 
	      performanceSummary->setClusterSizeX(theMEdetID, avgMean, avgRMS, emPtn);
      	    }
      	    if (theMEname.find("sizeY")!=string::npos) {
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      getSummaryMEmeanRMSnBins(iME, avgMean, avgRMS, emPtn); 
	      performanceSummary->setClusterSizeY(theMEdetID, avgMean, avgRMS, emPtn);
      	    } 
      	    // from SiPixelMonitorRecHit
      	    if (theMEname.find("nRecHits")!=string::npos) {
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      getSummaryMEmeanRMSnBins(iME, avgMean, avgRMS, emPtn); 
	      performanceSummary->setNumberOfRecHits(theMEdetID, avgMean, avgRMS, emPtn);
      	    } 
      	    // from SiPixelMonitorTrack
      	    if (theMEname.find("residualX")!=string::npos) {
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      getSummaryMEmeanRMSnBins(iME, avgMean, avgRMS, emPtn); 
	      performanceSummary->setResidualX(theMEdetID, avgMean, avgRMS, emPtn);
      	    }
      	    if (theMEname.find("residualY")!=string::npos) {
	      float avgMean=0.0, avgRMS=0.0, emPtn=0.0; 
	      performanceSummary->setResidualY(theMEdetID, avgMean, avgRMS, emPtn);
      	    } 
      	    // temporary solutions
      	    if (theMEname.find("OccupancyMap")!=std::string::npos) { // entire barrel and entire endcap only
	      int nNoisyCells=0, nEmptyCells=0; 
	      for (int xBin=0; xBin<(*iME)->getNbinsX(); xBin++) {   // 1 pixel per bin
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
	  else {
      	    // from SiPixelMonitorDigi 
      	    if (theMEname.find("ndigis_siPixelDigis")!=string::npos && theMEname.find("FREQ")==string::npos) { 
	      performanceSummary->setNumberOfDigis(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    } 
      	    if (theMEname.find("adc_siPixelDigis")!=string::npos) { 
    	      performanceSummary->setADC(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    }
      	    // from SiPixelMonitorCluster
      	    if (theMEname.find("nclusters_siPixelClusters")!=string::npos) {
	      performanceSummary->setNumberOfClusters(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    }
      	    if (theMEname.find("charge_siPixelClusters")!=string::npos) {
	      performanceSummary->setClusterCharge(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    } 
      	    if (theMEname.find("size_siPixelClusters")!=string::npos) {
	      performanceSummary->setClusterSize(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    } 
      	    if (theMEname.find("sizeX_siPixelClusters")!=string::npos) {
	      performanceSummary->setClusterSizeX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    }
      	    if (theMEname.find("sizeY_siPixelClusters")!=string::npos) {
	      performanceSummary->setClusterSizeY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    } 
      	    // from SiPixelMonitorRecHit
      	    if (theMEname.find("nRecHits_siPixelRecHits")!=string::npos) {
	      performanceSummary->setNumberOfRecHits(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    } 
      	    // from SiPixelMonitorTrack
      	    if (theMEname.find("residualX_siPixelTrackResiduals")!=string::npos) {
	      performanceSummary->setResidualX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    }
      	    if (theMEname.find("residualY_siPixelTrackResiduals")!=string::npos) {
	      performanceSummary->setResidualY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:-99.9));
      	    } 
	  }
      	} 
      	else {
      	  // from SiPixelMonitorDigi 
      	  if (theMEname.find("ndigis_siPixelDigis")!=string::npos && theMEname.find("FREQ")==string::npos) { 
	    performanceSummary->setNumberOfDigis(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  } 
      	  if (theMEname.find("adc_siPixelDigis")!=string::npos) { 
    	    performanceSummary->setADC(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  }
      	  // from SiPixelMonitorCluster
      	  if (theMEname.find("nclusters_siPixelClusters")!=string::npos) {
	    performanceSummary->setNumberOfClusters(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  }
      	  if (theMEname.find("charge_siPixelClusters")!=string::npos) {
	    performanceSummary->setClusterCharge(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  } 
      	  if (theMEname.find("size_siPixelClusters")!=string::npos) {
	    performanceSummary->setClusterSize(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  } 
      	  if (theMEname.find("sizeX_siPixelClusters")!=string::npos) {
	    performanceSummary->setClusterSizeX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  }
      	  if (theMEname.find("sizeY_siPixelClusters")!=string::npos) {
	    performanceSummary->setClusterSizeY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  } 
      	  // from SiPixelMonitorRecHit
      	  if (theMEname.find("nRecHits_siPixelRecHits")!=string::npos) {
	    performanceSummary->setNumberOfRecHits(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  } 
      	  // from SiPixelMonitorTrack
      	  if (theMEname.find("residualX_siPixelTrackResiduals")!=string::npos) {
	    performanceSummary->setResidualX(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  }
      	  if (theMEname.find("residualY_siPixelTrackResiduals")!=string::npos) {
	    performanceSummary->setResidualY(theMEdetID, (*iME)->getMean(), (*iME)->getRMS(), ((*iME)->getEntries()==0.0 ? 1.0:0.0));
      	  } 
      	  // temporary solutions
      	  if (theMEname.find("hitmap_siPixelClusters")!=std::string::npos || 
      	      theMEname.find("hitmap_siPixelDigis")!=std::string::npos) { // if digi map exists, use it; else use cluster map
	    int nNoisyCells=0, nEmptyCells=0;				  //			 not use xypos_siPixelRecHits
	    for (int xBin=0; xBin<(*iME)->getNbinsX(); xBin++) {	  // 4-1 pixel per bin
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
}


void SiPixelHistoricInfoDQMClient::writeDB() const {
  if (printDebug_) performanceSummary->printAll(); 
  else performanceSummary->print(); 
  cout << "SiPixelHistoricInfoDQMClient::writeDB()" << endl; 

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
