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


#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoEDAClient.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


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

  std::ostringstream endRunoutputfile; 
  endRunoutputfile << "SiPixelHistoricInfoEDAClient_"<< run.run() <<".root"; 
  savetoFile(endRunoutputfile.str()); 
}


void SiPixelHistoricInfoEDAClient::endJob() {
  if (parameterSet_.getUntrackedParameter<bool>("writeHisto", true)) {
    std::string outputfile = parameterSet_.getUntrackedParameter<std::string>("outputFile","SiPixelHistoricInfoEDAClient.root");
    std::cout << "SiPixelHistoricInfoEDAClient::endJob() outputFile = "<< outputfile << std::endl;
    dbe_->save(outputfile);
  }
}


void SiPixelHistoricInfoEDAClient::retrievePointersToModuleMEs(const edm::EventSetup& eventSetup) {
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
*/
  std::vector<string> listOfMEsWithFullPath;
  dbe_->getContents(listOfMEsWithFullPath);
  std::cout << "SiPixelHistoricInfoEDAClient::retrievePointersToModuleMEs: listOfMEsWithFullPath.size() = "
       << listOfMEsWithFullPath.size() << std::endl;

  for (std::vector<string>::const_iterator ime=listOfMEsWithFullPath.begin();
       ime!=listOfMEsWithFullPath.end(); ime++) {
    uint32_t length_path = (*ime).find(":",0); // divide path and histogram names
    if (length_path==string::npos) continue; // no ":" found, skip this and continue with next iterator step

    string thepath = (*ime).substr(0, length_path); // path part of the string is ended with ":"
    string allhistonames = (*ime).substr(length_path+1); // the rest  are histogram names
    uint while_counter=0;
    while (true) { // go out of the implicit loop when no more ',' is found
      while_counter++;
      uint thehistonamelength = allhistonames.find(",",0);
      string thehistoname;
      if (thehistonamelength!=string::npos) {
        thehistoname = allhistonames.substr(0, thehistonamelength);
        allhistonames.erase(0, thehistonamelength+1);
      }
      else thehistoname = allhistonames; // take all if no more ','

      string fullhistopath = thepath + "/" + thehistoname;
      MonitorElement* theMEPointer = dbe_->get(fullhistopath);

      SiPixelHistogramId hIdManager;
      string histoId = "#";
      uint32_t theMEDetId=0;
      if (theMEPointer) {
        histoId = theMEPointer->getName();
        theMEDetId = hIdManager.getRawId(histoId);

        std::map< uint32_t, std::vector<MonitorElement*> >::iterator is_me_in_map = ClientPointersToModuleMEs.find(theMEDetId);
        if (is_me_in_map==ClientPointersToModuleMEs.end()) {
          // if the key is not in map, create a new pair and insert it into the map
          std::vector<MonitorElement*> newvec;
          newvec.push_back(theMEPointer);
          ClientPointersToModuleMEs.insert(std::make_pair(theMEDetId, newvec));
        }
        else {
          // if the key is already in map, add the ME pointer to its std::vector
          (is_me_in_map->second).push_back(theMEPointer);
        }
      }
      if (thehistonamelength==string::npos) break; // leave the loop if no further ','

      if (while_counter>15) {
        std::cout << "leaving the loop while_counter = "<< while_counter << std::endl;
        break;
      }
    }
  }
}


void SiPixelHistoricInfoEDAClient::fillSummaryObjects(const edm::Run& run) const {
  for (std::map< uint32_t, std::vector<MonitorElement*> >::const_iterator imapmes=ClientPointersToModuleMEs.begin(); 
       imapmes!=ClientPointersToModuleMEs.end(); imapmes++) {
    uint32_t local_detid = imapmes->first;
    std::vector<MonitorElement*> locvec = imapmes->second;
    for (std::vector<MonitorElement*>::const_iterator imep = locvec.begin(); imep!=locvec.end(); imep++) {
      std::string MEName = (*imep)->getName();
      if (MEName.find("ndigis")!=std::string::npos) {
    	performanceSummary_->setNumberOfDigis(local_detid, (*imep)->getMean(), (*imep)->getRMS());

    	float percover = calculatePercentOver(*imep);
    	if (percover>-99.9) performanceSummary_->setNoisePercentage(local_detid, calculatePercentOver(*imep)); 
      }
    }
  }
}


float SiPixelHistoricInfoEDAClient::calculatePercentOver(MonitorElement* me) const {
  TH1F* root_ob = me->getTH1F();
  float percsum=0.0;
  TAxis* ta = root_ob->GetXaxis();
  unsigned int maxbins = ta->GetNbins();
  unsigned int upperbin = root_ob->FindBin(root_ob->GetMean() + 3.0*root_ob->GetRMS()); 
  if (upperbin<=maxbins) {
    percsum = root_ob->Integral(upperbin, maxbins)/root_ob->Integral();
    return percsum;
  }
  return -99.9; 
}


void SiPixelHistoricInfoEDAClient::writetoDB(const edm::Run& run) const {
  unsigned int thisRun = run.run();
  std::cout << "SiPixelHistoricInfoEDAClient::writetoDB() run = "<< thisRun << std::endl;

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


void SiPixelHistoricInfoEDAClient::writetoDB(edm::EventID eventID, edm::Timestamp EventTime) const {
  unsigned int thisRun        = eventID.run();
  unsigned int thisEvent      = eventID.event();
  unsigned long long thisTime = EventTime.value();
  std::cout << "SiPixelHistoricInfoEDAClient::writetoDB() run = "<< thisRun <<" event = "<< thisEvent 
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

/*
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
