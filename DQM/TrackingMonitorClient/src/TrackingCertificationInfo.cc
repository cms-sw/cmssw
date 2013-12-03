#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/TrackingMonitorClient/interface/TrackingUtility.h"
#include "DQM/TrackingMonitorClient/interface/TrackingCertificationInfo.h"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//Run Info
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
//
// -- Contructor
//
TrackingCertificationInfo::TrackingCertificationInfo(edm::ParameterSet const& pSet)
  : pSet_(pSet)
  , trackingCertificationBooked_(false)
  , trackingLSCertificationBooked_(false)
  , nFEDConnected_(0)
  , allPixelFEDConnected_(true)
{
  // Create MessageSender
  edm::LogInfo( "TrackingCertificationInfo") << "TrackingCertificationInfo::Deleting TrackingCertificationInfo ";

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();

  TopFolderName_ = pSet_.getUntrackedParameter<std::string>("TopFolderName","Tracking");
  //  std::cout << "[TrackingCertificationInfo::TrackingCertificationInfo] TopFolderName_: " << TopFolderName_ << std::endl;

  TrackingMEs tracking_mes;
  // load variables for Global certification

  checkPixelFEDs_ = pSet_.getParameter<bool>("checkPixelFEDs");
  if ( checkPixelFEDs_ ) {
    std::string QTname        = "pixel";
    tracking_mes.TrackingFlag = 0;
    TrackingMEsMap.insert(std::pair<std::string, TrackingMEs>(QTname, tracking_mes));
  }

  std::vector<edm::ParameterSet> TrackingGlobalQualityMEs = pSet_.getParameter< std::vector<edm::ParameterSet> >("TrackingGlobalQualityPSets" );
  for ( auto meQTset : TrackingGlobalQualityMEs ) {

    std::string QTname        = meQTset.getParameter<std::string>("QT");
    tracking_mes.TrackingFlag = 0;

    //    std::cout << "[TrackingQualityChecker::TrackingCertificationInfo] inserting " << QTname << " in TrackingMEsMap" << std::endl;
    TrackingMEsMap.insert(std::pair<std::string, TrackingMEs>(QTname, tracking_mes));
  }

  TrackingLSMEs tracking_ls_mes;
  // load variables for LS certification 
  std::vector<edm::ParameterSet> TrackingLSQualityMEs = pSet_.getParameter< std::vector<edm::ParameterSet> >("TrackingLSQualityMEs" );
  for ( auto meQTset : TrackingLSQualityMEs ) {

    std::string QTname        = meQTset.getParameter<std::string>("QT");
    tracking_ls_mes.TrackingFlag = 0;

    //    std::cout << "[TrackingQualityChecker::TrackingCertificationInfo] inserting " << QTname << " in TrackingMEsMap" << std::endl;
    TrackingLSMEsMap.insert(std::pair<std::string, TrackingLSMEs>(QTname, tracking_ls_mes));
  }


  // define sub-detectors which affect the quality
  SubDetFolder.push_back("SiStrip");
  SubDetFolder.push_back("Pixel");

}

TrackingCertificationInfo::~TrackingCertificationInfo() {
  edm::LogInfo("TrackingCertificationInfo") << "TrackingCertificationInfo::Deleting TrackingCertificationInfo ";

}
//
// -- Begin Job
//
void TrackingCertificationInfo::beginJob() {

}
//
// -- Begin Run
//
void TrackingCertificationInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {

  //  std::cout << "[TrackingCertificationInfo::beginRun] starting .." << std::endl;

  edm::LogInfo ("TrackingCertificationInfo") <<"TrackingCertificationInfo:: Begining of Run";
  unsigned long long cacheID = eSetup.get<SiStripDetCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       
  }
  eSetup.get<SiStripDetCablingRcd>().get(detCabling_);

  nFEDConnected_ = 0;
  int nPixelFEDConnected_ = 0;
  const int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID; 
  const int siPixelFedIdMin = FEDNumbering::MINSiPixelFEDID;
  const int siPixelFedIdMax = FEDNumbering::MAXSiPixelFEDID;
  const int siPixelFeds = (siPixelFedIdMax-siPixelFedIdMin+1);

  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
  if( eSetup.find( recordKey ) != 0) {

    edm::ESHandle<RunInfo> sumFED;
    eSetup.get<RunInfoRcd>().get(sumFED);    
    
    if ( sumFED.isValid() ) {

      std::vector<int> FedsInIds= sumFED->m_fed_in;   
      for ( auto fedID : FedsInIds ) {
	if (  fedID >= siPixelFedIdMin && fedID <= siPixelFedIdMax ) {
	  ++nFEDConnected_;
	  ++nPixelFEDConnected_;
	}
	else if ( fedID >= siStripFedIdMin && fedID <= siStripFedIdMax )
	  ++nFEDConnected_;
      }
      LogDebug ("TrackingDcsInfo") << " TrackingDcsInfo :: Connected FEDs " << nFEDConnected_;
    }
  }

  allPixelFEDConnected_ = ( nPixelFEDConnected_ == siPixelFeds );
 
  bookTrackingCertificationMEs();
  fillDummyTrackingCertification();

  bookTrackingCertificationMEsAtLumi();
  fillDummyTrackingCertificationAtLumi();
  
}

//
// -- Book MEs for Tracking Certification fractions  
//
void TrackingCertificationInfo::bookTrackingCertificationMEs() {

  //  std::cout << "[TrackingCertificationInfo::bookTrackingCertificationMEs] starting .. trackingCertificationBooked_: " << trackingCertificationBooked_ << std::endl;

  if ( !trackingCertificationBooked_ ) {

    dqmStore_->cd();
    std::string tracking_dir = "";
    TrackingUtility::getTopFolderPath(dqmStore_, TopFolderName_, tracking_dir);

    if (tracking_dir.size() > 0 ) dqmStore_->setCurrentFolder(tracking_dir+"/EventInfo");
    else dqmStore_->setCurrentFolder(TopFolderName_+"/EventInfo");

    TrackingCertification = dqmStore_->bookFloat("CertificationSummary");  
    
    std::string hname, htitle;
    hname  = "CertificationReportMap";
    htitle = "Tracking Certification Summary Map";
    size_t nQT = TrackingMEsMap.size();
    TrackingCertificationSummaryMap = dqmStore_->book2D(hname, htitle, nQT,0.5,float(nQT)+0.5,1,0.5,1.5);
    TrackingCertificationSummaryMap->setAxisTitle("Track Quality Type", 1);
    TrackingCertificationSummaryMap->setAxisTitle("QTest Flag", 2);
    size_t ibin =0;
    for ( auto meQTset : TrackingMEsMap ) {
      TrackingCertificationSummaryMap->setBinLabel(ibin+1,meQTset.first);
      ibin++;
    }


    if (tracking_dir.size() > 0 ) dqmStore_->setCurrentFolder(TopFolderName_+"/EventInfo/CertificationContents");
    else dqmStore_->setCurrentFolder(TopFolderName_+"/EventInfo/CertificationContents");

    for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin();
	 it != TrackingMEsMap.end(); it++) {
      std::string meQTname = it->first;
      //      std::cout << "[TrackingCertificationInfo::bookStatus] meQTname: " << meQTname << std::endl;
      it->second.TrackingFlag = dqmStore_->bookFloat("Track"+meQTname);
      //      std::cout << "[TrackingCertificationInfo::bookStatus] " << it->first << " exists ? " << it->second.TrackingFlag << std::endl;      
    }

    trackingCertificationBooked_ = true;
    dqmStore_->cd();
  }

  //  std::cout << "[TrackingCertificationInfo::bookStatus] trackingCertificationBooked_: " << trackingCertificationBooked_ << std::endl;
}

//
// -- Book MEs for Tracking Certification per LS
//
void TrackingCertificationInfo::bookTrackingCertificationMEsAtLumi() {

  //  std::cout << "[TrackingCertificationInfo::bookTrackingCertificationMEs] starting .. trackingCertificationBooked_: " << trackingCertificationBooked_ << std::endl;

  if ( !trackingLSCertificationBooked_ ) {

    dqmStore_->cd();
    std::string tracking_dir = "";
    TrackingUtility::getTopFolderPath(dqmStore_, TopFolderName_, tracking_dir);

    if (tracking_dir.size() > 0 ) dqmStore_->setCurrentFolder(tracking_dir+"/EventInfo");
    else dqmStore_->setCurrentFolder(TopFolderName_+"/EventInfo");

    TrackingLSCertification = dqmStore_->bookFloat("CertificationSummary");  
    
    if (tracking_dir.size() > 0 ) dqmStore_->setCurrentFolder(TopFolderName_+"/EventInfo/CertificationContents");
    else dqmStore_->setCurrentFolder(TopFolderName_+"/EventInfo/CertificationContents");

    for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin();
	 it != TrackingLSMEsMap.end(); it++) {
      std::string meQTname = it->first;
      //      std::cout << "[TrackingCertificationInfo::bookStatus] meQTname: " << meQTname << std::endl;
      it->second.TrackingFlag = dqmStore_->bookFloat("Track"+meQTname);
      //      std::cout << "[TrackingCertificationInfo::bookStatus] " << it->first << " exists ? " << it->second.TrackingFlag << std::endl;      
    }

    trackingLSCertificationBooked_ = true;
    dqmStore_->cd();
  }

  //  std::cout << "[TrackingCertificationInfo::bookStatus] trackingCertificationBooked_: " << trackingCertificationBooked_ << std::endl;
}
//
// -- Analyze
//
void TrackingCertificationInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) {
}
//
// -- End Luminosity Block
//
void TrackingCertificationInfo::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo( "TrackingDaqInfo") << "TrackingDaqInfo::endLuminosityBlock";

  if ( nFEDConnected_ > 0 ) fillTrackingCertificationMEsAtLumi();
  else fillDummyTrackingCertificationAtLumi();
}

//
// -- End of Run
//
void TrackingCertificationInfo::endRun(edm::Run const& run, edm::EventSetup const& eSetup){

  //  std::cout << "[TrackingCertificationInfo::endRun]" << std::endl;

  edm::LogInfo ("TrackingCertificationInfo") <<"TrackingCertificationInfo:: End Run";

  if ( nFEDConnected_ > 0 ) fillTrackingCertificationMEs(eSetup);
  else fillDummyTrackingCertification();

  //  std::cout << "[TrackingCertificationInfo::endRun] DONE" << std::endl;

}
//
// --Fill Tracking Certification 
//
void TrackingCertificationInfo::fillTrackingCertificationMEs(edm::EventSetup const& eSetup) {
  if ( !trackingCertificationBooked_ ) {
    //    edm::LogError("TrackingCertificationInfo") << " TrackingCertificationInfo::fillTrackingCertificationMEs : MEs missing ";
    return;
  }

  dqmStore_->cd();
  std::string tracking_dir = "";
  TrackingUtility::getTopFolderPath(dqmStore_, TopFolderName_, tracking_dir);
  //  std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEs] tracking_dir: " << tracking_dir << std::endl;
  std::vector<MonitorElement*> all_mes = dqmStore_->getContents(tracking_dir+"/EventInfo/reportSummaryContents");
  float fval = 1.0;

  //  std::cout << "all_mes: " << all_mes.size() << std::endl;

  if ( checkPixelFEDs_ ) {
    float val = 1.;
    if ( allPixelFEDConnected_ ) val = 0.;
    int xbin = 0;
    for (std::map<std::string, TrackingMEs>::const_iterator it = TrackingMEsMap.begin();
	 it != TrackingMEsMap.end(); it++) {
      std::string type = it->first;
      if ( type == "pixel" ) {
	it->second.TrackingFlag->Fill(val);
	TH2F*  th2d = TrackingCertificationSummaryMap->getTH2F();
	std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEs] xbin: " << xbin << " val: " << val << std::endl;
	th2d->SetBinContent(xbin+1,1,val);
      }
      xbin++;
    }
    fval = fminf(fval,val);
  }

  int xbin = ( checkPixelFEDs_ ? 1 : 0);
  for (std::vector<MonitorElement *>::const_iterator ime = all_mes.begin();
      ime!= all_mes.end(); ime++) {
    MonitorElement * me = (*ime);
    if (!me) continue;
    //    std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEs] me: " << me->getName() << std::endl;
    if (me->kind() == MonitorElement::DQM_KIND_REAL) {
      std::string name = me->getName();
      float val   = me->getFloatValue();

      for (std::map<std::string, TrackingMEs>::const_iterator it = TrackingMEsMap.begin();
	   it != TrackingMEsMap.end(); it++) {
	
	//	std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEs] ME: " << it->first << " [" << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue() << std::endl;
	
	std::string type = it->first;
	if (name.find(type) != std::string::npos) {
	  //	  std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEs] type: " << type << " <---> name: " << name << std::endl;
	  it->second.TrackingFlag->Fill(val);
	  std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEs] xbin: " << xbin << " val: " << val << std::endl;
	  TH2F*  th2d = TrackingCertificationSummaryMap->getTH2F();
	  th2d->SetBinContent(xbin+1,1,val);
	  xbin++;
	  break;
        }
	//	std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEs] ME: " << it->first << " [" << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue() << std::endl;

      }
      fval = fminf(fval,val);
    }
  }  
  //  std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEs] TrackingCertification: " << fval << std::endl;
  TrackingCertification->Fill(fval);  
}

//
// --Reset Tracking Certification 
//
void TrackingCertificationInfo::resetTrackingCertificationMEs() {
  if ( !trackingCertificationBooked_ ) bookTrackingCertificationMEs();
  TrackingCertification->Reset();
  for (std::map<std::string, TrackingMEs>::const_iterator it = TrackingMEsMap.begin();
       it != TrackingMEsMap.end(); it++) {
    it->second.TrackingFlag->Reset();
  }
}

//
// --Reset Tracking Certification per LS
//
void TrackingCertificationInfo::resetTrackingCertificationMEsAtLumi() {
  if ( !trackingLSCertificationBooked_ ) bookTrackingCertificationMEsAtLumi();
  TrackingLSCertification->Reset();
  for (std::map<std::string, TrackingLSMEs>::const_iterator it = TrackingLSMEsMap.begin();
       it != TrackingLSMEsMap.end(); it++) {
    it->second.TrackingFlag->Reset();
  }
}

//
// -- Fill Dummy Tracking Certification 
//
void TrackingCertificationInfo::fillDummyTrackingCertification() {
  resetTrackingCertificationMEs();
  if (trackingCertificationBooked_) {
    TrackingCertification->Fill(-1.0);
    for (std::map<std::string, TrackingMEs>::const_iterator it = TrackingMEsMap.begin();
	 it != TrackingMEsMap.end(); it++) {
      it->second.TrackingFlag->Fill(-1.0);
    }

    for (int xbin = 1; xbin < TrackingCertificationSummaryMap->getNbinsX()+1; xbin++ )
      for (int ybin = 1; ybin < TrackingCertificationSummaryMap->getNbinsY()+1; ybin++ )
	TrackingCertificationSummaryMap->Fill(xbin,ybin,-1);

    
  }
}

//
// -- Fill Dummy Tracking Certification per LS
//
void TrackingCertificationInfo::fillDummyTrackingCertificationAtLumi() {
  resetTrackingCertificationMEsAtLumi();
  if (trackingLSCertificationBooked_) {
    TrackingLSCertification->Fill(-1.0);
    for (std::map<std::string, TrackingLSMEs>::const_iterator it = TrackingLSMEsMap.begin();
	 it != TrackingLSMEsMap.end(); it++) {
      it->second.TrackingFlag->Fill(-1.0);
    }

  }
}

//
// --Fill Tracking Certification per LS
//
void TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi() {
  //  std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] starting .." << std::endl;
  if ( !trackingLSCertificationBooked_ ) {
    return;
  }
  resetTrackingCertificationMEsAtLumi();

  dqmStore_->cd();
  std::string tracking_dir = "";
  TrackingUtility::getTopFolderPath(dqmStore_, TopFolderName_, tracking_dir);
  //  std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] tracking_dir: " << tracking_dir << std::endl;


  //  std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] tracking_dir: " << tracking_dir << std::endl;
  std::vector<MonitorElement*> all_mes = dqmStore_->getContents(tracking_dir+"/EventInfo/reportSummaryContents");

  //  std::cout << "all_mes: " << all_mes.size() << std::endl;

  for (std::vector<MonitorElement *>::const_iterator ime = all_mes.begin();
      ime!= all_mes.end(); ime++) {
    MonitorElement * me = (*ime);
    if (!me) continue;
    //    std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] me: " << me->getName() << std::endl;
    if (me->kind() == MonitorElement::DQM_KIND_REAL) {
      std::string name = me->getName();
      float val   = me->getFloatValue();
      //      std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] val:  " << val << std::endl;

      for (std::map<std::string, TrackingLSMEs>::const_iterator it = TrackingLSMEsMap.begin();
	   it != TrackingLSMEsMap.end(); it++) {
	
	//	std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] ME: " << it->first << " [" << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue() << std::endl;
	
	std::string type = it->first;
	//	std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] type: " << type << std::endl;
	if (name.find(type) != std::string::npos) {
	  //	  std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] type: " << type << " <---> name: " << name << std::endl;
	  it->second.TrackingFlag->Fill(val);
	  break;
	}
	//	std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] ME: " << it->first << " [" << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue() << std::endl;
      }
    }
  }

  float global_dqm_flag = 1.0;
  std::string full_path = tracking_dir + "/EventInfo/reportSummary";
  MonitorElement* me_dqm = dqmStore_->get(full_path);
  if (me_dqm && me_dqm->kind() == MonitorElement::DQM_KIND_REAL) global_dqm_flag = me_dqm->getFloatValue();
  //  std::cout << "[TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi] global_dqm_flag: " << global_dqm_flag << std::endl; 

  TrackingLSCertification->Reset();
  TrackingLSCertification->Fill(global_dqm_flag);
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingCertificationInfo);
