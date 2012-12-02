#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

//
// -- Constructor
// 
SiStripTrackerMapCreator::SiStripTrackerMapCreator() {
  trackerMap_ = 0;
  if(!edm::Service<TkDetMap>().isAvailable()){
    edm::LogError("TkHistoMap") <<
      "\n------------------------------------------"
      "\nUnAvailable Service TkHistoMap: please insert in the configuration file an instance like"
      "\n\tprocess.TkDetMap = cms.Service(\"TkDetMap\")"
      "\n------------------------------------------";
  }
  tkDetMap_=edm::Service<TkDetMap>().operator->();
}
//
// -- Destructor
//
SiStripTrackerMapCreator::~SiStripTrackerMapCreator() {
  if (trackerMap_) delete trackerMap_;
}
//
// -- Create Geometric and Fed Tracker Map
//
void SiStripTrackerMapCreator::create(const edm::ParameterSet & tkmapPset, 
     const edm::ESHandle<SiStripFedCabling>& fedcabling, DQMStore* dqm_store, std::string& map_type) {

  if (trackerMap_) delete trackerMap_;
  trackerMap_ = new TrackerMap(tkmapPset, fedcabling);
  std::string tmap_title = " Tracker Map from  " + map_type;
  trackerMap_->setTitle(tmap_title);
 
  nDet     = 0;
  tkMapLog = false;
  tkMapMax = 0.0; 
  tkMapMin = 0.0; 

  if (map_type == "QTestAlarm") {
    trackerMap_->fillc_all_blank();
    const std::vector<uint16_t>& feds = fedcabling->feds(); 
    uint32_t detId_save = 0;
    for(std::vector<unsigned short>::const_iterator ifed = feds.begin(); 
	ifed < feds.end(); ifed++){
      const std::vector<FedChannelConnection> fedChannels = fedcabling->connections( *ifed );
      for(std::vector<FedChannelConnection>::const_iterator iconn = fedChannels.begin(); iconn < fedChannels.end(); iconn++){
	
	uint32_t detId = iconn->detId();
	if (detId == 0 || detId == 0xFFFFFFFF)  continue;
	if (detId_save != detId) {
	  detId_save = detId;
          paintTkMapFromAlarm(detId, dqm_store);
	}
      }
    } 
  } else {
    trackerMap_->fill_all_blank();
    setTkMapFromHistogram(dqm_store, map_type);
    setTkMapRange(map_type);
  }
  trackerMap_->printonline();
  delete trackerMap_;
  trackerMap_ = 0;
}
//
// -- Create Tracker Map for Offline process
//
void SiStripTrackerMapCreator::createForOffline(const edm::ParameterSet & tkmapPset, 
						const edm::ESHandle<SiStripFedCabling>& fedcabling,
						DQMStore* dqm_store, std::string& map_type){
  if (trackerMap_) delete trackerMap_;
  trackerMap_ = new TrackerMap(tkmapPset,fedcabling);

  tkMapLog = tkmapPset.getUntrackedParameter<bool>("logScale",false);
  bool tkMapPSU = tkmapPset.getUntrackedParameter<bool>("psuMap",false);
 
  std::string tmap_title = " Tracker Map from  " + map_type;
  if(tkMapLog) tmap_title += ": Log10 scale";
  trackerMap_->setTitle(tmap_title);

  setTkMapFromHistogram(dqm_store, map_type);
  // if not overwitten by manual configuration min=0 and max= mean value * 2.5
  setTkMapRangeOffline();

  // check manual setting
  
  if(tkmapPset.exists("mapMax")) tkMapMax = tkmapPset.getUntrackedParameter<double>("mapMax");
  if(tkmapPset.exists("mapMin")) tkMapMin = tkmapPset.getUntrackedParameter<double>("mapMin");
  
  std::cout << "Ready to save TkMap " << map_type << " with range set to " << tkMapMin << " - " << tkMapMax << std::endl;
  
  trackerMap_->save(true, tkMapMin,tkMapMax, map_type+".svg");  
  trackerMap_->save(true, tkMapMin,tkMapMax, map_type+".png",4500,2400);

  if(tkMapPSU) {

    std::cout << "Ready to save PSU TkMap " << map_type << " with range set to " << tkMapMin << " - " << tkMapMax << std::endl;
    trackerMap_->save_as_psutrackermap(true, tkMapMin,tkMapMax, map_type+"_psu.svg");
    trackerMap_->save_as_psutrackermap(true, tkMapMin,tkMapMax, map_type+"_psu.png",6000,3200);

  }

  delete trackerMap_;
  trackerMap_ = 0;
}
//
// -- Paint Tracker Map with QTest Alarms 
//
void SiStripTrackerMapCreator::paintTkMapFromAlarm(uint32_t det_id, DQMStore* dqm_store) {
  
  std::ostringstream comment;
  uint16_t flag = 0; 
  flag = getDetectorFlagAndComment(dqm_store, det_id, comment);

  int rval, gval, bval;
  SiStripUtility::getDetectorStatusColor(flag, rval, gval, bval);
  trackerMap_->setText(det_id, comment.str());
  trackerMap_->fillc(det_id, rval, gval, bval);
}

//
// --  Paint Tracker Map from TkHistoMap Histograms
void SiStripTrackerMapCreator::setTkMapFromHistogram(DQMStore* dqm_store, std::string& htype) {
  dqm_store->cd();

  std::string mdir = "MechanicalView";
  if (!SiStripUtility::goToDir(dqm_store, mdir)) return;
  std::string mechanicalview_dir = dqm_store->pwd();

  std::vector<std::string> subdet_folder;
  subdet_folder.push_back("TIB");
  subdet_folder.push_back("TOB");
  subdet_folder.push_back("TEC/side_1");
  subdet_folder.push_back("TEC/side_2");
  subdet_folder.push_back("TID/side_1");
  subdet_folder.push_back("TID/side_2");

  nDet     = 0;
  tkMapMax = 0.0; 
  tkMapMin = 0.0; 

  for (std::vector<std::string>::const_iterator it = subdet_folder.begin(); it != subdet_folder.end(); it++) {
    std::string dname = mechanicalview_dir + "/" + (*it);
    if (!dqm_store->dirExists(dname)) continue;
    dqm_store->cd(dname);  
    std::vector<std::string> layerVec = dqm_store->getSubdirs();
    for (std::vector<std::string>::const_iterator iLayer = layerVec.begin(); iLayer != layerVec.end(); iLayer++) { 
      if ((*iLayer).find("BadModuleList") !=std::string::npos) continue;
      std::vector<MonitorElement*> meVec = dqm_store->getContents((*iLayer));
      MonitorElement* tkhmap_me = 0;
      std::string name;
      for (std::vector<MonitorElement*>::const_iterator itkh = meVec.begin();  itkh != meVec.end(); itkh++) {
	name = (*itkh)->getName();
	if (name.find("TkHMap") == std::string::npos) continue;
	if (htype == "QTestAlarm" ){
	  tkhmap_me = (*itkh);
	  break;
	} else if (name.find(htype) != std::string::npos) {
	  tkhmap_me = (*itkh);
	  break; 
	} 
      }
      if (tkhmap_me != 0) {
        paintTkMapFromHistogram(dqm_store,tkhmap_me, htype);
      } 
    }
    dqm_store->cd(mechanicalview_dir);
  }
  dqm_store->cd();
}
void SiStripTrackerMapCreator::paintTkMapFromHistogram(DQMStore* dqm_store, MonitorElement* me, std::string& htype) {
  std::string name  = me->getName();
  std::string lname = name.substr(name.find("TkHMap_")+7);  
  lname = lname.substr(lname.find("_T")+1);
  std::vector<uint32_t> layer_detids;
  tkDetMap_->getDetsForLayer(tkDetMap_->getLayerNum(lname), layer_detids);
  for (std::vector<uint32_t>::const_iterator idet = layer_detids.begin(); idet != layer_detids.end(); idet++) {
    uint32_t det_id= (*idet);
    if (det_id <= 0) continue;
    nDet++;
    std::ostringstream comment;
    uint32_t flag = 0;
    flag = getDetectorFlagAndComment(dqm_store, det_id, comment);
    trackerMap_->setText(det_id, comment.str());
    const TkLayerMap::XYbin& xyval = tkDetMap_->getXY(det_id);
    float fval = 0.0;
    if ( (name.find("NumberOfOfffTrackCluster") != std::string::npos) || 
         (name.find("NumberOfOnTrackCluster") != std::string::npos) ) {
      if (me->kind() == MonitorElement::DQM_KIND_TPROFILE2D) {   
	TProfile2D* tp = me->getTProfile2D() ;
	fval =  tp->GetBinEntries(tp->GetBin(xyval.ix, xyval.iy)) * tp->GetBinContent(xyval.ix, xyval.iy);
      }
    } else  fval = me->getBinContent(xyval.ix, xyval.iy);
    if (htype == "QTestAlarm") {
      int rval, gval, bval;
      SiStripUtility::getDetectorStatusColor(flag, rval, gval, bval);
      trackerMap_->fillc(det_id, rval, gval, bval);
    } else {
      if (fval == 0.0) trackerMap_->fillc(det_id,255, 255, 255);  
      else {
        if(tkMapLog) fval = log(fval)/log(10);
 	trackerMap_->fill_current_val(det_id, fval);
      }
      tkMapMax += fval;
    }
  }
} 
//
// -- Get Tracker Map Fill Range
//
void SiStripTrackerMapCreator::setTkMapRange(std::string& map_type) {
  tkMapMin = 0.0;
  if (tkMapMax == 0.0) { 
    if (map_type.find("FractionOfBadChannels") != std::string::npos)        tkMapMax = 1.0;
    else if (map_type.find("NumberOfCluster") != std::string::npos)         tkMapMax = 0.01;
    else if (map_type.find("NumberOfDigi") != std::string::npos)            tkMapMax = 0.6;
    else if (map_type.find("NumberOfOffTrackCluster") != std::string::npos) tkMapMax = 100.0;
    else if (map_type.find("NumberOfOnTrackCluster") != std::string::npos)  tkMapMax = 50.0;
    else if (map_type.find("StoNCorrOnTrack") != std::string::npos)         tkMapMax = 200.0;
  } else {
    tkMapMax = tkMapMax/nDet*1.0;
    tkMapMax = tkMapMax * 2.5;
 }
  trackerMap_->setRange(tkMapMin, tkMapMax);
}
void SiStripTrackerMapCreator::setTkMapRangeOffline() {
  tkMapMin = 0.0;
  if (tkMapMax != 0.0) { 
    tkMapMax = tkMapMax/nDet*1.0;
    tkMapMax = tkMapMax * 2.5;
 }
  trackerMap_->setRange(tkMapMin, tkMapMax);
}
//
// -- Get Flag and status Comment
//
uint16_t SiStripTrackerMapCreator::getDetectorFlagAndComment(DQMStore* dqm_store, uint32_t det_id, std::ostringstream& comment) {
  comment << " DetId " << det_id << " : ";
  uint16_t flag = 0;

  SiStripFolderOrganizer folder_organizer;
  std::string subdet_folder, badmodule_folder;

  folder_organizer.getSubDetFolder(det_id, subdet_folder);
  if (dqm_store->dirExists(subdet_folder)){ 
    badmodule_folder = subdet_folder + "/BadModuleList";
  } else {
    badmodule_folder = dqm_store->pwd() + "/BadModuleList"; 
  }
  if (!dqm_store->dirExists(badmodule_folder)) return flag;

  std::ostringstream badmodule_path;
  badmodule_path << badmodule_folder << "/" << det_id;

  MonitorElement* bad_module_me = dqm_store->get(badmodule_path.str());
  if (bad_module_me && bad_module_me->kind() == MonitorElement::DQM_KIND_INT) {
    flag = bad_module_me->getIntValue();
    std::string message;
    SiStripUtility::getBadModuleStatus(flag, message);
    comment << message.c_str();
  }
  return flag;
}
