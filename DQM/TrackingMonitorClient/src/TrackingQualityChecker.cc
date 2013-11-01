#include "DQM/TrackingMonitorClient/interface/TrackingQualityChecker.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/QReport.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DQM/TrackingMonitorClient/interface/TrackingUtility.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <iomanip>
//
// -- Constructor
// 
TrackingQualityChecker::TrackingQualityChecker(edm::ParameterSet const& ps) :
  pSet_(ps)
{
  edm::LogInfo("TrackingQualityChecker") << " Creating TrackingQualityChecker " << "\n" ;

  bookedTrackingGlobalStatus_ = false;
  bookedTrackingLSStatus_     = false;

  if(!edm::Service<TkDetMap>().isAvailable()){
    edm::LogError("TkHistoMap") <<
      "\n------------------------------------------"
      "\nUnAvailable Service TkHistoMap: please insert in the configuration file an instance like"
      "\n\tprocess.TkDetMap = cms.Service(\"TkDetMap\")"
      "\n------------------------------------------";
  }

  TopFolderName_ = pSet_.getUntrackedParameter<std::string>("TopFolderName","Tracking");

  TrackingMEs tracking_mes;
  std::vector<edm::ParameterSet> TrackingGlobalQualityMEs = pSet_.getParameter< std::vector<edm::ParameterSet> >("TrackingGlobalQualityPSets" );
  for ( auto meQTset : TrackingGlobalQualityMEs ) {

    std::string QTname           = meQTset.getParameter<std::string>("QT");
    tracking_mes.HistoDir        = meQTset.getParameter<std::string>("dir");
    tracking_mes.HistoName       = meQTset.getParameter<std::string>("name");
    //    std::cout << "[TrackingQualityChecker::TrackingQualityChecker] inserting " << QTname << " in TrackingMEsMap" << std::endl;
    TrackingMEsMap.insert(std::pair<std::string, TrackingMEs>(QTname, tracking_mes));
  }
  //  std::cout << "[TrackingQualityChecker::TrackingQualityChecker] created TrackingMEsMap" << std::endl;

  TrackingLSMEs tracking_ls_mes;
  std::vector<edm::ParameterSet> TrackingLSQualityMEs = pSet_.getParameter< std::vector<edm::ParameterSet> >("TrackingLSQualityPSets" );
  for ( auto meQTset : TrackingLSQualityMEs ) {

    std::string QTname           = meQTset.getParameter<std::string>("QT");
    tracking_ls_mes.HistoLSDir      = meQTset.exists("LSdir")      ? meQTset.getParameter<std::string>("LSdir")  : "";
    tracking_ls_mes.HistoLSName     = meQTset.exists("LSname")     ? meQTset.getParameter<std::string>("LSname") : "";
    tracking_ls_mes.HistoLSLowerCut = meQTset.exists("LSlowerCut") ? meQTset.getParameter<double>("LSlowerCut")  : -1.;
    tracking_ls_mes.HistoLSUpperCut = meQTset.exists("LSupperCut") ? meQTset.getParameter<double>("LSupperCut")  : -1.;
    tracking_ls_mes.TrackingFlag = 0;

    //    std::cout << "[TrackingQualityChecker::TrackingQualityChecker] inserting " << QTname << " in TrackingMEsMap" << std::endl;
    TrackingLSMEsMap.insert(std::pair<std::string, TrackingLSMEs>(QTname, tracking_ls_mes));
  }
  //  std::cout << "[TrackingQualityChecker::TrackingQualityChecker] created TrackingLSMEsMap" << std::endl;

}
//
// --  Destructor
// 
TrackingQualityChecker::~TrackingQualityChecker() {
  edm::LogInfo("TrackingQualityChecker") << " Deleting TrackingQualityChecker " << "\n" ;
}
//
// -- create reportSummary MEs
//
void TrackingQualityChecker::bookGlobalStatus(DQMStore* dqm_store) {
  
  //  std::cout << "[TrackingQualityChecker::bookGlobalStatus] already booked ? " << (bookedTrackingGlobalStatus_ ? "yes" : "nope") << std::endl;

  if (!bookedTrackingGlobalStatus_) {
    dqm_store->cd();     
    edm::LogInfo("TrackingQualityChecker") << " booking TrackingQualityStatus" << "\n";

    std::string tracking_dir = "";
    TrackingUtility::getTopFolderPath(dqm_store, TopFolderName_, tracking_dir);
    dqm_store->setCurrentFolder(TopFolderName_+"/EventInfo"); 
    
    TrackGlobalSummaryReportGlobal = dqm_store->bookFloat("reportSummary");
    
    std::string hname, htitle;
    hname  = "reportSummaryMap";
    htitle = "Tracking Report Summary Map";
    
    size_t nQT = TrackingMEsMap.size();
    //    std::cout << "[TrackingQualityChecker::bookGlobalStatus] nQT: " << nQT << std::endl;
    TrackGlobalSummaryReportMap    = dqm_store->book2D(hname, htitle, nQT,0.5,float(nQT)+0.5,1,0.5,1.5);
    TrackGlobalSummaryReportMap->setAxisTitle("Track Quality Type", 1);
    TrackGlobalSummaryReportMap->setAxisTitle("QTest Flag", 2);
    size_t ibin =0;
    for ( auto meQTset : TrackingMEsMap ) {
      TrackGlobalSummaryReportMap->setBinLabel(ibin+1,meQTset.first);
      ibin++;
    }

    dqm_store->setCurrentFolder(TopFolderName_+"/EventInfo/reportSummaryContents");  

    for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin();
         it != TrackingMEsMap.end(); it++) {
      std::string meQTname = it->first;
      it->second.TrackingFlag = dqm_store->bookFloat("Track"+meQTname);
      //      std::cout << "[TrackingQualityChecker::bookGlobalStatus] " << it->first << " exists ? " << it->second.TrackingFlag << std::endl;      
      //      std::cout << "[TrackingQualityChecker::bookGlobalStatus] DONE w/ TrackingMEsMap" << std::endl;
    }

    bookedTrackingGlobalStatus_ = true;
    dqm_store->cd();
  }
}

void TrackingQualityChecker::bookLSStatus(DQMStore* dqm_store) {
  
  //  std::cout << "[TrackingQualityChecker::bookLSStatus] already booked ? " << (bookedTrackingLSStatus_ ? "yes" : "nope") << std::endl;

  if (!bookedTrackingLSStatus_) {
    dqm_store->cd();     
    edm::LogInfo("TrackingQualityChecker") << " booking TrackingQualityStatus" << "\n";

    std::string tracking_dir = "";
    TrackingUtility::getTopFolderPath(dqm_store, TopFolderName_, tracking_dir);
    dqm_store->setCurrentFolder(TopFolderName_+"/EventInfo"); 
    
    TrackLSSummaryReportGlobal = dqm_store->bookFloat("reportSummary");
    
    std::string hname, htitle;
    hname  = "reportSummaryMap";
    htitle = "Tracking Report Summary Map";
    
    //    size_t nQT = TrackingLSMEsMap.size();
    //    std::cout << "[TrackingQualityChecker::bookLSStatus] nQT: " << nQT << std::endl;

    dqm_store->setCurrentFolder(TopFolderName_+"/EventInfo/reportSummaryContents");  
    for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin();
         it != TrackingLSMEsMap.end(); it++) {
      std::string meQTname = it->first;
      it->second.TrackingFlag = dqm_store->bookFloat("Track"+meQTname);
      //      std::cout << "[TrackingQualityChecker::bookLSStatus] " << it->first << " exists ? " << it->second.TrackingFlag << std::endl;      
      //      std::cout << "[TrackingQualityChecker::bookLSStatus] DONE w/ TrackingLSMEsMap" << std::endl;
    }

    bookedTrackingLSStatus_ = true;
    dqm_store->cd();
  }
}

//
// -- Fill Dummy  Status
//
void TrackingQualityChecker::fillDummyGlobalStatus(){
  //  std::cout << "[TrackingQualityChecker::fillDummyGlobalStatus] starting ..." << std::endl;

  resetGlobalStatus();
  //  std::cout << "[TrackingQualityChecker::fillDummyGlobalStatus] already booked ? " << (bookedTrackingGlobalStatus_ ? "yes" : "nope") << std::endl;
  if (bookedTrackingGlobalStatus_) {  

    TrackGlobalSummaryReportGlobal->Fill(-1.0);

    for (int ibin = 1; ibin < TrackGlobalSummaryReportMap->getNbinsX()+1; ibin++) {
      fillStatusHistogram(TrackGlobalSummaryReportMap, ibin, 1, -1.0);
    }

    for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin();
         it != TrackingMEsMap.end(); it++)
      it->second.TrackingFlag->Fill(-1.0);
    //    std::cout << "[TrackingQualityChecker::fillDummyGlobalStatus] DONE w/ TrackingMEsMap" << std::endl;

  }
}
void TrackingQualityChecker::fillDummyLSStatus(){
  //  std::cout << "[TrackingQualityChecker::fillDummyLSStatus] starting ..." << std::endl;

  resetLSStatus();
  //  std::cout << "[TrackingQualityChecker::fillDummyLSStatus] already booked ? " << (bookedTrackingLSStatus_ ? "yes" : "nope") << std::endl;
  if (bookedTrackingLSStatus_) {  

    TrackLSSummaryReportGlobal->Fill(-1.0);
    for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin();
         it != TrackingLSMEsMap.end(); it++)
      it->second.TrackingFlag->Fill(-1.0);
    //    std::cout << "[TrackingQualityChecker::fillDummyLSStatus] DONE w/ TrackingLSMEsMap" << std::endl;

  }
}

//
// -- Reset Status
//
void TrackingQualityChecker::resetGlobalStatus() {

  //  std::cout << "[TrackingQualityChecker::resetGlobalStatus] already booked ? " << (bookedTrackingGlobalStatus_ ? "yes" : "nope") << std::endl;
  if (bookedTrackingGlobalStatus_) {  

    TrackGlobalSummaryReportGlobal -> Reset();
    TrackGlobalSummaryReportMap    -> Reset();

    for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin();
         it != TrackingMEsMap.end(); it++) {
      MonitorElement* me = it->second.TrackingFlag;
      //      std::cout << "[TrackingQualityChecker::resetGlobalStatus] " << it->second.HistoName << " exist ? " << ( it->second.TrackingFlag == NULL ? "nope" : "yes" ) << " ---> " << me << std::endl;      
      me->Reset();
    }
    //    std::cout << "[TrackingQualityChecker::resetGlobalStatus] DONE w/ TrackingMEsMap" << std::endl;

  }
}
void TrackingQualityChecker::resetLSStatus() {

  //  std::cout << "[TrackingQualityChecker::resetLSStatus] already booked ? " << (bookedTrackingLSStatus_ ? "yes" : "nope") << std::endl;
  if (bookedTrackingLSStatus_) {  

    TrackLSSummaryReportGlobal -> Reset();
    for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin();
         it != TrackingLSMEsMap.end(); it++) {
      MonitorElement* me = it->second.TrackingFlag;
      //      std::cout << "[TrackingQualityChecker::resetLSStatus] " << it->second.HistoLSName << " exist ? " << ( it->second.TrackingFlag == NULL ? "nope" : "yes" ) << " ---> " << me << std::endl;      
      me->Reset();
    }
    //    std::cout << "[TrackingQualityChecker::resetLSStatus] DONE w/ TrackingLSMEsMap" << std::endl;

  }
}

//
// -- Fill Status
//
void TrackingQualityChecker::fillGlobalStatus(DQMStore* dqm_store) {

  //  std::cout << "[TrackingQualityChecker::fillGlobalStatus] already booked ? " << (bookedTrackingGlobalStatus_ ? "yes" : "nope") << std::endl;
  if (!bookedTrackingGlobalStatus_) bookGlobalStatus(dqm_store);

  fillDummyGlobalStatus();
  fillTrackingStatus(dqm_store);
  //  std::cout << "[TrackingQualityChecker::fillGlobalStatus] DONE" << std::endl;
  dqm_store->cd();
}

void TrackingQualityChecker::fillLSStatus(DQMStore* dqm_store) {

  //  std::cout << "[TrackingQualityChecker::fillLSStatus] already booked ? " << (bookedTrackingLSStatus_ ? "yes" : "nope") << std::endl;
  if (!bookedTrackingLSStatus_) bookLSStatus(dqm_store);

  fillDummyLSStatus();
  fillTrackingStatusAtLumi(dqm_store);
  //  std::cout << "[TrackingQualityChecker::fillLSStatus] DONE" << std::endl;
  dqm_store->cd();
}

//
// -- Fill Tracking Status
//
void TrackingQualityChecker::fillTrackingStatus(DQMStore* dqm_store) {

  float gstatus = 0.0;

  dqm_store->cd();
  if (!TrackingUtility::goToDir(dqm_store, TopFolderName_)) return;
  
    
  int ibin = 0;
  for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin();
       it != TrackingMEsMap.end(); it++) {

    //    std::cout << "[TrackingQualityChecker::fillTrackingStatus] ME: " << it->first << " [" << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue() << std::endl;

    ibin++;
    
    std::string localMEdirpath = it->second.HistoDir;
    std::string MEname         = it->second.HistoName;

    std::vector<MonitorElement*> tmpMEvec = dqm_store->getContents(dqm_store->pwd()+"/"+localMEdirpath);
    MonitorElement* me = NULL;

    size_t nMEs = 0;
    for ( auto ime : tmpMEvec ) {
      std::string name = ime->getName();
      if ( name.find(MEname) != std::string::npos) {
	me = ime;
	nMEs++;
      }
    }
    // only one ME found
    if (nMEs == 1) {
      float status = 0.;
      for ( auto ime : tmpMEvec ) {
	std::string name = ime->getName();
	if ( name.find(MEname) != std::string::npos) {
	  me = ime;
	}
      }
      if (!me) continue;
      //      std::cout << "[TrackingQualityChecker::fillTrackingStatus] status: " << status << std::endl;
      std::vector<QReport *> qt_reports = me->getQReports();          
      size_t nQTme = qt_reports.size();
      if (nQTme != 0) {
	//	std::cout << "[TrackingQualityChecker::fillTrackingStatus] qt_reports: " << qt_reports.size() << std::endl;
	// loop on possible QTs
	for ( auto iQT : qt_reports ) {
	  status += iQT->getQTresult();
	  //	  std::cout << "[TrackingQualityChecker::fillTrackingStatus] iQT: " << iQT->getQRName() << std::endl;
	  //	  std::cout << "[TrackingQualityChecker::fillTrackingStatus] MEname: " << MEname << " status: " << iQT->getQTresult() << " exists ? " << (it->second.TrackingFlag ? "yes " : "no ") << it->second.TrackingFlag << std::endl;
	}
	status = status/float(nQTme);
	//	std::cout << "[TrackingQualityChecker::fillTrackingStatus] MEname: " << MEname << " status: " << status << std::endl;
	it->second.TrackingFlag->Fill(status);
	//	std::cout << "[TrackingQualityChecker::fillTrackingStatus] TrackGlobalSummaryReportMap: " << TrackGlobalSummaryReportMap << std::endl;
	fillStatusHistogram(TrackGlobalSummaryReportMap, ibin, 1, status);
      }
      
      //      std::cout << "[TrackingQualityChecker::fillTrackingStatus] gstatus: " << gstatus << " x status: " << status << std::endl;
      if ( status < 0. ) gstatus = -1.;
      else gstatus += status; 
      //      std::cout << "[TrackingQualityChecker::fillTrackingStatus] ===> gstatus: " << gstatus << std::endl;
      //      std::cout << "[TrackingQualityChecker::fillTrackingStatus] ME: " << it->first << " [" << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue() << std::endl;

    } else { // more than 1 ME w/ the same root => they need to be considered together
      float status = 1.;
      for ( auto ime : tmpMEvec ) {
	float tmp_status = 1.;
	std::string name = ime->getName();
	if ( name.find(MEname) != std::string::npos) {
	  me = ime;

	  //	  std::cout << "[TrackingQualityChecker::fillTrackingStatus] status: " << status << std::endl;
	  std::vector<QReport *> qt_reports = me->getQReports();          
	  size_t nQTme = qt_reports.size();
	  if (nQTme != 0) {
	    //	    std::cout << "[TrackingQualityChecker::fillTrackingStatus] qt_reports: " << qt_reports.size() << std::endl;
	    // loop on possible QTs
	    for ( auto iQT : qt_reports ) {
	      tmp_status += iQT->getQTresult();
	      //	      std::cout << "[TrackingQualityChecker::fillTrackingStatus] iQT: " << iQT->getQRName() << std::endl;
	      //	      std::cout << "[TrackingQualityChecker::fillTrackingStatus] MEname: " << MEname << " status: " << iQT->getQTresult() << " exists ? " << (it->second.TrackingFlag ? "yes " : "no ") << it->second.TrackingFlag << std::endl;
	    }
	    tmp_status = tmp_status/float(nQTme);
	  }
	}
	status = fminf(tmp_status,status);
      }
      if ( status < 0. ) gstatus = -1.;
      else gstatus += status;
      //      std::cout << "[TrackingQualityChecker::fillTrackingStatus] MEname: " << MEname << " status: " << status << std::endl;
      it->second.TrackingFlag->Fill(status);
      //      std::cout << "[TrackingQualityChecker::fillTrackingStatus] TrackGlobalSummaryReportMap: " << TrackGlobalSummaryReportMap << std::endl;

      fillStatusHistogram(TrackGlobalSummaryReportMap, ibin, 1, status);
    }
  }

  //  std::cout << "[TrackingQualityChecker::fillTrackingStatus] gstatus: " << gstatus << std::endl;  
  size_t nQT = TrackingMEsMap.size();
  if (gstatus < 1.) gstatus = -1.;
  else gstatus = gstatus/float(nQT);

  //  std::cout << "[TrackingQualityChecker::fillTrackingStatus] ===> gstatus: " << gstatus << std::endl;
  TrackGlobalSummaryReportGlobal->Fill(gstatus);
  dqm_store->cd();

  //  std::cout << "[TrackingQualityChecker::fillTrackingStatus] DONE" << std::endl;

}

//
// -- Fill Report Summary Map
//
 void TrackingQualityChecker::fillStatusHistogram(MonitorElement* me, int xbin, int ybin, float val){
   if (me &&  me->kind() == MonitorElement::DQM_KIND_TH2F) {
     TH2F*  th2d = me->getTH2F();
     th2d->SetBinContent(xbin, ybin, val);
   }
 }

// Fill Tracking Status MEs at the Lumi block
// 
void TrackingQualityChecker::fillTrackingStatusAtLumi(DQMStore* dqm_store){

  //  std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] starting .. " << std::endl;
  float gstatus = 1.0;

  dqm_store->cd();
  if (!TrackingUtility::goToDir(dqm_store, TopFolderName_)) return;


  int ibin = 0;
  for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin();
       it != TrackingLSMEsMap.end(); it++) {
    
    //    std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ME: " << it->first << " [" << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue() << std::endl;

    ibin++;
  
    std::string localMEdirpath = it->second.HistoLSDir;
    std::string MEname         = it->second.HistoLSName;
    float lower_cut            = it->second.HistoLSLowerCut; 
    float upper_cut            = it->second.HistoLSUpperCut; 

    float status = 1.0; 

    std::vector<MonitorElement*> tmpMEvec = dqm_store->getContents(dqm_store->pwd()+"/"+localMEdirpath);
    //    std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] tmpMEvec: " << tmpMEvec.size() << std::endl;

    MonitorElement* me = NULL;

    size_t nMEs = 0;
    for ( auto ime : tmpMEvec ) {
      std::string name = ime->getName();
      if ( name.find(MEname) != std::string::npos) {
	me = ime;
	nMEs++;
      }
    }
    // only one ME found
    if (nMEs == 1) {
      for ( auto ime : tmpMEvec ) {
	std::string name = ime->getName();
	if ( name.find(MEname) != std::string::npos) {
	  me = ime;
	}
      }
      if (!me) continue;
      
      if (me->kind() == MonitorElement::DQM_KIND_TH1F) {
	float x_mean = me->getMean();
	//	std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] MEname: " << MEname << " x_mean: " << x_mean << std::endl;
	if (x_mean <= lower_cut || x_mean > upper_cut) status = 0.0;
	else status = 1.0; 
      }
    } else { // more than 1 ME w/ the same root => they need to be considered together
      for ( auto ime : tmpMEvec ) {
	float tmp_status = 1.;
	std::string name = ime->getName();
	if ( name.find(MEname) != std::string::npos) {
	  me = ime;
	  if (!me) continue;
	  
	  if (me->kind() == MonitorElement::DQM_KIND_TH1F) {
	    float x_mean = me->getMean();
	    //	    std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] MEname: " << MEname << "[" << me->getName() << "]  x_mean: " << x_mean << std::endl;
	    if (x_mean <= lower_cut || x_mean > upper_cut) tmp_status = 0.0;
	    else tmp_status = 1.0; 
	    //	    std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] tmp_status: " << tmp_status << std::endl;
	  }
	}
	status = fminf(tmp_status,status);
	//	std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ==> status: " << status << std::endl;
      } // loop on tmpMEvec
    }
    it->second.TrackingFlag->Fill(status);
    //    std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ===> status: " << status << " [" << gstatus << "]" << std::endl;
    if (status == 0.0) gstatus = -1.0;
    else gstatus = gstatus * status; 
    //    std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ===> gstatus: " << gstatus << std::endl;
    //    std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ME: " << it->first << " [" << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue() << std::endl;
  }
  TrackLSSummaryReportGlobal->Fill(gstatus);
  dqm_store->cd();
  
  //  std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] DONE" << std::endl;
}
