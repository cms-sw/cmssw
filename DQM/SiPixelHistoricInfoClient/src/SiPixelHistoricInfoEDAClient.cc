#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <sys/time.h>

#include "TNamed.h"
#include "TH1F.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoEDAClient.h"


using namespace std;
using namespace cgicc;
using namespace xcept;


SiPixelHistoricInfoEDAClient::SiPixelHistoricInfoEDAClient(const edm::ParameterSet& parameterSet) { 
  parameterSet_ = parameterSet;  
  dbe_ = edm::Service<DQMStore>().operator->(); 
  dbe_->setVerbose(0); 
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
    while (fin.getline(buf, BUF_SIZE,'\n')) html_dump << buf << std::endl; 
    fin.close();
   *out << html_dump.str() << std::endl;
   
    defaultWebPageCreated_ = true;
  }  
  webInterface_->handleEDARequest(in, out);
} */


SiPixelHistoricInfoEDAClient::~SiPixelHistoricInfoEDAClient() {}


void SiPixelHistoricInfoEDAClient::beginJob(const edm::EventSetup& eventSetup) {
  nEvents = 0; 
}


void SiPixelHistoricInfoEDAClient::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  performanceSummary_ = new SiPixelPerformanceSummary();
  performanceSummary_->clear(); 
  performanceSummary_->setRunNumber(run.run());
  firstEventInRun = true;
}


void SiPixelHistoricInfoEDAClient::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  if (firstEventInRun) {
    firstEventInRun = false;
    performanceSummary_->setTimeValue(event.time().value());
  }
  nEvents++;
}


void SiPixelHistoricInfoEDAClient::endRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  firstEventInRun = false;
  std::cout << "SiPixelHistoricInfoEDAClient::endRun() number of events = "<< nEvents << std::endl;

  retrievePointersToModuleMEs(eventSetup);
  fillSummaryObjects(run);
  performanceSummary_->print();
  writetoDB(run); 
  
  if (parameterSet_.getUntrackedParameter<bool>("writeHisto", true)) {
    std::string outputDir = parameterSet_.getUntrackedParameter<std::string>("outputDir",".");
    std::ostringstream endRunOutputFile; 
    endRunOutputFile << outputDir << "/SiPixelHistoricInfoEDAClient_" << run.run() <<".root"; 
    dbe_->save(endRunOutputFile.str()); 
  }
}


void SiPixelHistoricInfoEDAClient::endJob() {
  if (parameterSet_.getUntrackedParameter<bool>("writeHisto", true)) {
    std::string outputDir = parameterSet_.getUntrackedParameter<std::string>("outputDir",".");
    std::ostringstream endJobOutputFile; 
    endJobOutputFile << outputDir << "/SiPixelHistoricInfoEDAClient_endJob.root"; 
    dbe_->save(endJobOutputFile.str());
  }
}


void SiPixelHistoricInfoEDAClient::retrievePointersToModuleMEs(const edm::EventSetup& eventSetup) {
  std::vector<string> listOfMEsWithFullPath;
  dbe_->getContents(listOfMEsWithFullPath);
    
  for (std::vector<string>::const_iterator ime=listOfMEsWithFullPath.begin(); 
       ime!=listOfMEsWithFullPath.end(); ime++) {
    uint32_t pathLength = (*ime).find(":",0);     
    string thePath = (*ime).substr(0, pathLength); 
    string allHists = (*ime).substr(pathLength+1); 
    
    if (thePath.find("Pixel",0)!=string::npos && thePath.find("Module",0)!=string::npos) {
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
            std::vector<MonitorElement*> newMEvector;
            newMEvector.push_back(theMEpointer);
            ClientPointersToModuleMEs.insert(std::make_pair(localMEdetID, newMEvector));
          }
          else ((ClientPointersToModuleMEs.find(localMEdetID))->second).push_back(theMEpointer);
        }
      } 
      while (histnameLength!=string::npos); 
    } 
  } 
}


void SiPixelHistoricInfoEDAClient::fillSummaryObjects(const edm::Run& run) const {
  for (std::map< uint32_t, std::vector<MonitorElement*> >::const_iterator iMEvec=ClientPointersToModuleMEs.begin(); 
       iMEvec!=ClientPointersToModuleMEs.end(); iMEvec++) {
    uint32_t localMEdetID = iMEvec->first;
    std::vector<MonitorElement*> theMEvector = iMEvec->second;
    
    for (std::vector<MonitorElement*>::const_iterator iMEpntr = theMEvector.begin(); 
         iMEpntr!=theMEvector.end(); iMEpntr++) {
      std::string theMEname = (*iMEpntr)->getName();
      // if (parameterSet_.getUntrackedParameter<bool>("debug", true)) std::cout << theMEname << std::endl; 
      
      if (theMEname.find("ndigis_siPixelDigis")!=std::string::npos) { 
    	performanceSummary_->setNumberOfDigis(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
        if (parameterSet_.getUntrackedParameter<bool>("debug", true)) { 
	  std::cout << "sipixel performance summary table set with the number of digi's "<< (*iMEpntr)->getMean() 
	                                                                        << " +- "<< (*iMEpntr)->getRMS() 
		    <<" for det ID "<< localMEdetID << std::endl; 
        }
      } 
      /* if (theMEname.find("size_siPixelClusters")!=std::string::npos) {
    	performanceSummary_->setClusterSize(localMEdetID, (*iMEpntr)->getMean(), (*iMEpntr)->getRMS());
      } */
    }
  }
}


void SiPixelHistoricInfoEDAClient::writetoDB(const edm::Run& run) const {
  unsigned int thisRun = run.run();
  std::cout << std::endl << "SiPixelHistoricInfoEDAClient::writetoDB() run = "<< thisRun << std::endl; 

  edm::Service<cond::service::PoolDBOutputService> mydbservice; 
  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiPixelPerformanceSummaryRcd")) {
      mydbservice->createNewIOV<SiPixelPerformanceSummary>(performanceSummary_, 
                                                           mydbservice->beginOfTime(), 
							   mydbservice->endOfTime(), 
							  "SiPixelPerformanceSummaryRcd"); 
    } 
    else {
      mydbservice->appendSinceTime<SiPixelPerformanceSummary>(performanceSummary_, 
                                                              mydbservice->currentTime(), 
							     "SiPixelPerformanceSummaryRcd"); 
    }
  }
  else  edm::LogError("writetoDB") << "service unavailable" << std::endl; 

  std::cout << "SiPixelHistoricInfoEDAClient::writetoDB() finished. "<< std::endl; 
}


void SiPixelHistoricInfoEDAClient::writetoDB(edm::EventID eventID, edm::Timestamp EventTime) const {
  unsigned int thisRun        = eventID.run();
  unsigned int thisEvent      = eventID.event();
  unsigned long long thisTime = EventTime.value();
  std::cout << std::endl << "SiPixelHistoricInfoEDAClient::writetoDB() run = "<< thisRun 
                                                                 <<" event = "<< thisEvent 
								  <<" time = "<< thisTime << std::endl;

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiPixelPerformanceSummaryRcd")) {
      mydbservice->createNewIOV<SiPixelPerformanceSummary>(performanceSummary_, 
                                                           mydbservice->beginOfTime(), 
							   mydbservice->endOfTime(), 
							  "SiPixelPerformanceSummaryRcd");     
    } 
    else {
      mydbservice->appendSinceTime<SiPixelPerformanceSummary>(performanceSummary_, 
                                                              mydbservice->currentTime(), 
							     "SiPixelPerformanceSummaryRcd");
    }
  } 
  else edm::LogError("writetoDB") << "service unavailable" << std::endl;
}


void SiPixelHistoricInfoEDAClient::savetoFile(std::string filename) const {
  dbe_->save(filename); 
}


float SiPixelHistoricInfoEDAClient::calculatePercentOver(MonitorElement* me) const {
  TH1F* hist = me->getTH1F();
  unsigned int nBins = hist->GetNbinsX();
  unsigned int upperBin = hist->FindBin(hist->GetMean()+3*hist->GetRMS()); 
  float percentage=0.0;
  if ((upperBin-nBins)<1) {
    percentage = hist->Integral(upperBin,nBins)/hist->Integral(); 
    return percentage;
  }
  return -99.9; 
}

/*
  edm::ESHandle<TrackerGeometry> pDD;
  eventSetup.get<TrackerDigiGeometryRecord>().get(pDD);

  ClientPointersToModuleMEs.clear(); 
  for (TrackerGeometry::DetContainer::const_iterator it=pDD->dets().begin(); 
       it!=pDD->dets().end(); it++) {
    if (dynamic_cast<PixelGeomDetUnit*>((*it))!=0) {
      DetId detId = (*it)->geographicalId();

      if (detId.subdetId()==static_cast<int>(PixelSubdetector::PixelBarrel)) {
	std::vector<MonitorElement*> local_mes = dbe_->get(detId()); 
	ClientPointersToModuleMEs.insert(std::make_pair(detId(), local_mes));
      } 
      else if (detId.subdetId()==static_cast<int>(PixelSubdetector::PixelEndcap)) {
	std::vector<MonitorElement*> local_mes = dbe_->get(detId()); 
	ClientPointersToModuleMEs.insert(std::make_pair(detId(), local_mes));
      }
    }
  }
  std::cout << "SiPixelHistoricInfoEDAClient::retrievePointersToModuleMEs() ClientPointersToModuleMEs.size() = "
	    <<  ClientPointersToModuleMEs.size() << std::endl;


  float percentage = calculatePercentOver(*iMEpntr);
  if (percentage>-99.9) performanceSummary_->setNoisePercentage(localMEdetID, calculatePercentOver(*iMEpntr)); 


void SiPixelHistoricInfoEDAClient::printMEs() const {
  std::cout << "SiPixelHistoricInfoEDAClient::printMEs ClientPointersToModuleMEs.size() = "
            << ClientPointersToModuleMEs.size() << std::endl;

  for (std::map< uint32_t, std::vector<MonitorElement*> >::iterator imapmes=ClientPointersToModuleMEs.begin();
       imapmes!=ClientPointersToModuleMEs.end(); imapmes++) {
    std::cout << "Det ID = "<< imapmes->first << std::endl;

    std::vector<MonitorElement*> locvec = imapmes->second;
    for (std::vector<MonitorElement*>::const_iterator imep=locvec.begin();
         imep!=locvec.end(); imep++) {
      std::cout << (*imep)->getName() << " entries/mean/rms: "
           	<< (*imep)->getEntries() <<" / "   
           	<< (*imep)->getMean() <<" / "	   
           	<< (*imep)->getRMS() << std::endl; 
    }
    std::vector<MonitorElement*> tagged_mes = dbe_->get(imapmes->first);
    for (std::vector<MonitorElement*>::const_iterator imep=tagged_mes.begin();
         imep!=tagged_mes.end(); imep++) {
      std::cout << "(tagged ME) "<< (*imep)->getName() <<" entries/mean/rms: "
                            	 << (*imep)->getEntries() <<" / "
                            	 << (*imep)->getMean() <<" / "
                            	 << (*imep)->getRMS() << std::endl;
    }
  }
}
*/

