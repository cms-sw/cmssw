#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"

#include <iostream>

//
// -- Constructor
// 
/*
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
*/
SiStripTrackerMapCreator::SiStripTrackerMapCreator(const edm::EventSetup& eSetup): meanToMaxFactor_(2.5),eSetup_(eSetup)
						  //, psumap_() 
{
  cached_detid=0;
  cached_layer=0;
  trackerMap_ = 0;
  stripTopLevelDir_="";
  eSetup_.get<SiStripDetCablingRcd>().get(detcabling_);
  //  psumap_.BuildMap("CalibTracker/SiStripDCS/data/StripPSUDetIDMap_FromJan132010.dat",false);
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
				      DQMStore* dqm_store, std::string& map_type,
                                      const edm::EventSetup& eSetup) {

  const SiStripFedCabling* fedcabling = detcabling_->fedCabling();

  if(!fedcabling) return;

  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  if (trackerMap_) delete trackerMap_;
  trackerMap_ = new TrackerMap(tkmapPset, fedcabling,tTopo);
  std::string tmap_title = " Tracker Map from  " + map_type;
  trackerMap_->setTitle(tmap_title);
 
  nDet     = 0;
  tkMapMax_ = 0.0; 
  tkMapMin_ = 0.0; 
  meanToMaxFactor_ = 2.5;
  useSSQuality_ = false;
  ssqLabel_ = "";
  stripTopLevelDir_="SiStrip";

  if (map_type == "QTestAlarm") {
    setTkMapFromAlarm(dqm_store, eSetup);
    /*
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
    */
  } else {
    trackerMap_->fill_all_blank();
    setTkMapFromHistogram(dqm_store, map_type, eSetup);
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
						DQMStore* dqm_store, std::string& map_type,
                                                const edm::EventSetup& eSetup) {

  // Determine the strip top level dirctory in the DQM file: it is the path where MechanicalView is minus one directory

  std::string mdir = "MechanicalView";
  dqm_store->cd();
  if (!SiStripUtility::goToDir(dqm_store, mdir)) {
    edm::LogError("SiStripTopLevelDirNotFound") << "I cannot find the SiStrip top level directory in the DQM file";
  }
  else {
    std::string mechanicalview_dir = dqm_store->pwd();
    stripTopLevelDir_=mechanicalview_dir.substr(0,mechanicalview_dir.find_last_of("/"));    
    edm::LogInfo("SiStripTopLevelDirFound") << "SiStrip top level directory is " << stripTopLevelDir_;
  }
  dqm_store->cd();

  //
  const SiStripFedCabling* fedcabling = detcabling_->fedCabling();

  if(!fedcabling) return;

  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  if (trackerMap_) delete trackerMap_;
  trackerMap_ = new TrackerMap(tkmapPset,fedcabling,tTopo);

  meanToMaxFactor_ = tkmapPset.getUntrackedParameter<double>("meanToMaxFact",2.5);
  useSSQuality_ = tkmapPset.getUntrackedParameter<bool>("useSSQuality",false);
  ssqLabel_ = tkmapPset.getUntrackedParameter<std::string>("ssqLabel","");
  bool tkMapPSU = tkmapPset.getUntrackedParameter<bool>("psuMap",false);
  bool tkMapFED = tkmapPset.getUntrackedParameter<bool>("fedMap",false);
  std::string namesuffix = tkmapPset.getUntrackedParameter<std::string>("mapSuffix",""); 
 
  std::string tmap_title = " Tracker Map from  " + map_type;
  trackerMap_->setTitle(tmap_title);

  if(tkmapPset.exists("TopModules"))
    topModules=tkmapPset.getUntrackedParameter<bool>("TopModules");
  else
    topModules=false;

  if(tkmapPset.exists("numberTopModules"))
      numTopModules=tkmapPset.getUntrackedParameter<int32_t>("numberTopModules");
  else
      numTopModules = 20;
  
  if (tkmapPset.exists("topModLabel"))
        topModLabel=tkmapPset.getUntrackedParameter<int32_t>("topModLabel");
  else
        topModLabel = " top Modules " + map_type;

  if (map_type == "QTestAlarm") {
    setTkMapFromAlarm(dqm_store, eSetup);
  }
  else {
    setTkMapFromHistogram(dqm_store, map_type, eSetup);
  }
  // if not overwitten by manual configuration min=0 and max= mean value * meanToMaxFactor_
  setTkMapRangeOffline();

  // check manual setting
  
  if(tkmapPset.exists("mapMax")) tkMapMax_ = tkmapPset.getUntrackedParameter<double>("mapMax");
  if(tkmapPset.exists("mapMin")) tkMapMin_ = tkmapPset.getUntrackedParameter<double>("mapMin");
  
  edm::LogInfo("TkMapToBeSaved") << "Ready to save TkMap " << map_type << namesuffix << " with range set to " << tkMapMin_ << " - " << tkMapMax_;
  
  trackerMap_->save(true, tkMapMin_,tkMapMax_, map_type+namesuffix+".svg");  
  trackerMap_->save(true, tkMapMin_,tkMapMax_, map_type+namesuffix+".png",4500,2400);

  if(tkMapPSU) {

    edm::LogInfo("PSUMapToBeSaved") << "Ready to save PSU TkMap " << map_type << namesuffix << " with range set to " << tkMapMin_ << " - " << tkMapMax_;
    //    trackerMap_->save_as_psutrackermap(true, tkMapMin_,tkMapMax_, map_type+namesuffix+"_psu.svg");
    trackerMap_->save_as_psutrackermap(true, tkMapMin_,tkMapMax_, map_type+namesuffix+"_psu.png");

  }

  if(tkMapFED) {

    edm::LogInfo("FEDMapToBeSaved") << "Ready to save FED TkMap " << map_type << namesuffix << " with range set to " << tkMapMin_ << " - " << tkMapMax_;
    //    trackerMap_->save_as_fedtrackermap(true, tkMapMin_,tkMapMax_, map_type+"_fed.svg");
    trackerMap_->save_as_fedtrackermap(true, tkMapMin_,tkMapMax_, map_type+namesuffix+"_fed.png");

  }

  delete trackerMap_;
  trackerMap_ = 0;
}
//
// -- Fill Tracker Map with QTest Alarms and SiStripQuality bad modules
void SiStripTrackerMapCreator::setTkMapFromAlarm(DQMStore* dqm_store, const edm::EventSetup& eSetup) {

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  nDet     = 0;
  tkMapMax_ = 0.0; 
  tkMapMin_ = 0.0; 

  edm::ESHandle<SiStripQuality> ssq;

  if(useSSQuality_) { eSetup_.get<SiStripQualityRcd>().get(ssqLabel_,ssq);  }

  trackerMap_->fillc_all_blank();

  std::map<unsigned int,std::string>* badmodmap = new std::map<unsigned int,std::string>;

  // used to avoid multiple checks on the same detid since the loop is done on the FED channels
    uint32_t detId_save = 0;
    // example of loop using SiStripDetCabling
    for(std::map< uint32_t, std::vector<const FedChannelConnection *> >::const_iterator module = detcabling_->getDetCabling().begin(); 
	module!=detcabling_->getDetCabling().end();++module) {
      uint32_t detId = module->first;
      if (detId == 0 || detId == 0xFFFFFFFF)  continue;
      if (detId_save != detId) {
	detId_save = detId;
	bool isBad = useSSQuality_ && ssq->IsModuleBad(detId);
	paintTkMapFromAlarm(detId, tTopo, dqm_store,isBad,badmodmap);
      } 
      else {
	edm::LogWarning("TwiceTheSameDetId") << "The detid " << detId << " was found already in the loop on SiStripDetCabling";
      }
    }
    //
    printBadModuleList(badmodmap, eSetup);
    delete badmodmap;
}
//
void SiStripTrackerMapCreator::printBadModuleList(std::map<unsigned int,std::string>* badmodmap, const edm::EventSetup& eSetup) {

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  bool tibDone=false,tidSide1Done=false,tidSide2Done=false,tobDone=false,tecSide1Done=false,tecSide2Done=false;
  unsigned int tibFirst=369120277-1,
    tidSide1First=402664197-1,tidSide2First=402672389-1,
    tobFirst=436228134-1,
    tecSide1First=470044965-1,tecSide2First=470307109-1;

  int ntib=0,ntids1=0,ntids2=0,ntob=0,ntecs1=0,ntecs2=0;

  for(std::map<unsigned int,std::string>::const_iterator badmod = badmodmap->begin(); badmod!= badmodmap->end(); ++badmod) {
    SiStripDetId ssdetid(badmod->first);
    if(ssdetid.subDetector()==SiStripDetId::TIB) ntib++;
    if(ssdetid.subDetector()==SiStripDetId::TID) {
      
      if(tTopo->tidSide(ssdetid)==1) ntids1++;
      if(tTopo->tidSide(ssdetid)==2) ntids2++;
    }
    if(ssdetid.subDetector()==SiStripDetId::TOB) ntob++;
    if(ssdetid.subDetector()==SiStripDetId::TEC) {
      
      if(tTopo->tecSide(ssdetid)==1) ntecs1++;
      if(tTopo->tecSide(ssdetid)==2) ntecs2++;
    }
  }

  edm::LogVerbatim("BadModuleList") << "Number of bad modules in total:";
  edm::LogVerbatim("BadModuleList") << "--------------------------------------------------------------";
  edm::LogVerbatim("BadModuleList") << "TIB: " << ntib;
  edm::LogVerbatim("BadModuleList") << "TID/MINUS: " << ntids1;
  edm::LogVerbatim("BadModuleList") << "TID/PLUS: " << ntids2;
  edm::LogVerbatim("BadModuleList") << "TOB: " << ntob;
  edm::LogVerbatim("BadModuleList") << "TEC/MINUS: " << ntecs1;
  edm::LogVerbatim("BadModuleList") << "TEC/PLUS: " << ntecs2;
  edm::LogVerbatim("BadModuleList") << "-------------------------------";
  edm::LogVerbatim("BadModuleList") ;
  edm::LogVerbatim("BadModuleList") << "List of bad modules per partition:";
  edm::LogVerbatim("BadModuleList") << "----------------------------------";
  
  for(std::map<unsigned int,std::string>::const_iterator badmod = badmodmap->begin(); badmod!= badmodmap->end(); ++badmod) {
    if(!tibDone && badmod->first >= tibFirst) {
      tibDone = true;
      edm::LogVerbatim("BadModuleList") ;
      edm::LogVerbatim("BadModuleList") << "SubDetector TIB";
      edm::LogVerbatim("BadModuleList") ;
    }
    if(!tidSide1Done && badmod->first >= tidSide1First) {
      tidSide1Done = true;
      edm::LogVerbatim("BadModuleList") ;
      edm::LogVerbatim("BadModuleList") << "SubDetector TID/MINUS";
      edm::LogVerbatim("BadModuleList") ;
    }
    if(!tidSide2Done && badmod->first >= tidSide2First) {
      tidSide2Done = true;
      edm::LogVerbatim("BadModuleList") ;
      edm::LogVerbatim("BadModuleList") << "SubDetector TID/PLUS";
      edm::LogVerbatim("BadModuleList") ;
    }
    if(!tobDone && badmod->first >= tobFirst) {
      tobDone = true;
      edm::LogVerbatim("BadModuleList") ;
      edm::LogVerbatim("BadModuleList") << "SubDetector TOB";
      edm::LogVerbatim("BadModuleList") ;
    }
    if(!tecSide1Done && badmod->first >= tecSide1First) {
      tecSide1Done = true;
      edm::LogVerbatim("BadModuleList") ;
      edm::LogVerbatim("BadModuleList") << "SubDetector TEC/MINUS";
      edm::LogVerbatim("BadModuleList") ;
    }
    if(!tecSide2Done && badmod->first >= tecSide2First) {
      tecSide2Done = true;
      edm::LogVerbatim("BadModuleList") ;
      edm::LogVerbatim("BadModuleList") << "SubDetector TEC/PLUS";
      edm::LogVerbatim("BadModuleList") ;
    }
    edm::LogVerbatim("BadModuleList") << badmod->second;
  }
}

//
// -- Paint Tracker Map with QTest Alarms 
//
void SiStripTrackerMapCreator::paintTkMapFromAlarm(uint32_t det_id, const TrackerTopology* tTopo,
                                                   DQMStore* dqm_store, bool isBad, std::map<unsigned int,std::string>* badmodmap) {
  std::ostringstream comment;
  uint16_t flag = 0; 
  flag = getDetectorFlagAndComment(dqm_store, det_id, tTopo, comment);

  int rval, gval, bval;
  SiStripUtility::getDetectorStatusColor(flag, rval, gval, bval);
  if(isBad) { rval=255; gval=255; bval = 0; comment << " PCLBadModule ";}
  trackerMap_->setText(det_id, comment.str());
  trackerMap_->fillc(det_id, rval, gval, bval);
  if(badmodmap && (flag!=0 || isBad)) (*badmodmap)[det_id] = comment.str();
}

//
// --  Paint Tracker Map from TkHistoMap Histograms
void SiStripTrackerMapCreator::setTkMapFromHistogram(DQMStore* dqm_store, std::string& htype, const edm::EventSetup& eSetup) {
  dqm_store->cd();

  std::string mdir = "MechanicalView";
  if (!SiStripUtility::goToDir(dqm_store, mdir)) return;
  std::string mechanicalview_dir = dqm_store->pwd();

  std::vector<std::string> subdet_folder;
  subdet_folder.push_back("TIB");
  subdet_folder.push_back("TOB");
  subdet_folder.push_back("TEC/MINUS");
  subdet_folder.push_back("TEC/PLUS");
  subdet_folder.push_back("TID/MINUS");
  subdet_folder.push_back("TID/PLUS");

  nDet     = 0;
  tkMapMax_ = 0.0; 
  tkMapMin_ = 0.0;
  std::vector<std::pair<float,uint32_t> >* topNmodVec = new std::vector<std::pair<float,uint32_t> >; 

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
	  edm::LogError("ItShouldNotBeHere") << "QTestAlarm map: you should not be here!";
	  tkhmap_me = (*itkh);
	  break;
	} else if (name.find(htype) != std::string::npos) {
	  tkhmap_me = (*itkh);
	  break; 
	} 
      }
      if (tkhmap_me != 0) {
	if (topModules){
        	paintTkMapFromHistogram(dqm_store,tkhmap_me, htype, topNmodVec);
	}
	else paintTkMapFromHistogram(dqm_store,tkhmap_me, htype, 0);
      } 
    }
    dqm_store->cd(mechanicalview_dir);
  }
  dqm_store->cd();
  if (topModules) printTopModules(topNmodVec, eSetup);
  delete topNmodVec;
}

void SiStripTrackerMapCreator::printTopModules(std::vector<std::pair<float,uint32_t> >* topNmodVec, const edm::EventSetup& eSetup){

   //////////////Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHandle;
   eSetup.get<IdealGeometryRecord>().get(tTopoHandle);
   const TrackerTopology* const tTopo = tTopoHandle.product();  

   std::sort(topNmodVec->rbegin(), topNmodVec->rend());
   topNmodVec->resize(numTopModules);
   
   edm::LogVerbatim("TopModules") << topModLabel;
   edm::LogVerbatim("TopModules") << "------------------------------------------------------";
   
   for (std::vector<std::pair<float, uint32_t> >::const_iterator itNmod = topNmodVec->begin(); itNmod != topNmodVec->end(); itNmod++){
       std::pair<float, uint32_t> aPair=(*itNmod);
       uint32_t det_id = aPair.second;
       std::ostringstream comment;
       std::string subdetector;
       SiStripDetId ssdetid(aPair.second);
       if(ssdetid.subDetector()==SiStripDetId::TIB) subdetector = "TIB       ";
       if(ssdetid.subDetector()==SiStripDetId::TID){
	  if(tTopo->tidSide(ssdetid)==1) subdetector = "TID/MINUS ";
          if(tTopo->tidSide(ssdetid)==2) subdetector = "TID/PLUS  ";
       }
       if(ssdetid.subDetector()==SiStripDetId::TOB) subdetector = "TOB       ";
       if(ssdetid.subDetector()==SiStripDetId::TEC){
	  if(tTopo->tecSide(ssdetid)==1) subdetector = "TEC/MINUS ";
	  if(tTopo->tecSide(ssdetid)==2) subdetector = "TEC/PLUS  ";
       }
       uint16_t flag = getDetectorFlagAndComment(0, det_id, tTopo, comment);
       if (flag == 0) edm::LogVerbatim("TopModules") << subdetector << comment.str() << " value: "<< aPair.first;
   }
   edm::LogVerbatim("TopModules") << "------------------------------------------------------";
}


void SiStripTrackerMapCreator::paintTkMapFromHistogram(DQMStore* dqm_store, MonitorElement* me, std::string& htype, std::vector<std::pair<float, uint32_t> >* topNmodVec) {

  //  edm::ESHandle<SiStripQuality> ssq;

  //  if(useSSQuality_) { eSetup_.get<SiStripQualityRcd>().get(ssqLabel_,ssq);  }

  std::string name  = me->getName();
  std::string lname = name.substr(name.find("TkHMap_")+7);  
  lname = lname.substr(lname.find("_T")+1);
  std::vector<uint32_t> layer_detids;
  tkDetMap_->getDetsForLayer(tkDetMap_->getLayerNum(lname), layer_detids);
  for (std::vector<uint32_t>::const_iterator idet = layer_detids.begin(); idet != layer_detids.end(); idet++) {
    uint32_t det_id= (*idet);
    if (det_id <= 0) continue;
    nDet++;
    const TkLayerMap::XYbin& xyval = tkDetMap_->getXY(det_id , cached_detid , cached_layer , cached_XYbin);
    float fval = 0.0;
    if ( (name.find("NumberOfOfffTrackCluster") != std::string::npos) || 
         (name.find("NumberOfOnTrackCluster") != std::string::npos) ) {
      if (me->kind() == MonitorElement::DQM_KIND_TPROFILE2D) {   
	TProfile2D* tp = me->getTProfile2D() ;
	fval =  tp->GetBinEntries(tp->GetBin(xyval.ix, xyval.iy)) * tp->GetBinContent(xyval.ix, xyval.iy);
      }
    } else  fval = me->getBinContent(xyval.ix, xyval.iy);
    if (htype == "QTestAlarm") {
      edm::LogError("ItShouldNotBeHere") << "QTestAlarm map: you should not be here!";
      /*
      int rval, gval, bval;
      std::ostringstream comment;
      uint32_t flag = 0;
      flag = getDetectorFlagAndComment(dqm_store, det_id, comment);
      SiStripUtility::getDetectorStatusColor(flag, rval, gval, bval);
      if(useSSQuality_ && ssq->IsModuleBad(det_id)) { rval=255; gval=255; bval = 0;}
      trackerMap_->fillc(det_id, rval, gval, bval);
      trackerMap_->setText(det_id, comment.str());
      */
    } else {
      if (fval == 0.0) trackerMap_->fillc(det_id,255, 255, 255);  
      else {
 	trackerMap_->fill_current_val(det_id, fval);
      }
      tkMapMax_ += fval;
      if(topNmodVec){
	//std::ostringstream ss; ss << fval;
        auto detPair = std::make_pair(fval,det_id);
        topNmodVec->push_back(detPair);
      }
    }
  }
} 
//
// -- Get Tracker Map Fill Range
//
void SiStripTrackerMapCreator::setTkMapRange(std::string& map_type) {
  tkMapMin_ = 0.0;
  if (tkMapMax_ == 0.0) { 
    if (map_type.find("FractionOfBadChannels") != std::string::npos)        tkMapMax_ = 1.0;
    else if (map_type.find("NumberOfCluster") != std::string::npos)         tkMapMax_ = 0.01;
    else if (map_type.find("NumberOfDigi") != std::string::npos)            tkMapMax_ = 0.6;
    else if (map_type.find("NumberOfOffTrackCluster") != std::string::npos) tkMapMax_ = 100.0;
    else if (map_type.find("NumberOfOnTrackCluster") != std::string::npos)  tkMapMax_ = 50.0;
    else if (map_type.find("StoNCorrOnTrack") != std::string::npos)         tkMapMax_ = 200.0;
  } else {
    tkMapMax_ = tkMapMax_/nDet*1.0;
    tkMapMax_ = tkMapMax_ * meanToMaxFactor_;
 }
  trackerMap_->setRange(tkMapMin_, tkMapMax_);
}
void SiStripTrackerMapCreator::setTkMapRangeOffline() {
  tkMapMin_ = 0.0;
  if (tkMapMax_ != 0.0) { 
    tkMapMax_ = tkMapMax_/(nDet*1.0);
    tkMapMax_ = tkMapMax_ * meanToMaxFactor_;
 }
  // the following line seems to be useless and misleading: in the offline map creation the range is set with the save... methods
  //  trackerMap_->setRange(tkMapMin_, tkMapMax_);
}
//
// -- Get Flag and status Comment
//
uint16_t SiStripTrackerMapCreator::getDetectorFlagAndComment(DQMStore* dqm_store, uint32_t det_id,
                                                             const TrackerTopology* tTopo, std::ostringstream& comment) {
  //  comment << " DetId " << det_id << " : ";
  comment << "Module " << det_id;
  uint16_t flag = 0;

  // get FED channels corresponding to the det_id

  //  if(detcabling_) {
  comment << " FEDCHs ";
  std::vector<const FedChannelConnection*> conns = detcabling_->getConnections(det_id);
  for(unsigned int i=0; i< conns.size() ;++i) {
    if(conns[i]) {
      comment << std::setw(3) << conns[i]->fedId() << "/" << std::setw(2) << conns[i]->fedCh()<< " ";
    }
    else {
      comment << "       ";
    }
  }
  if(conns.size()==0) {	comment << "                     ";      }
  if(conns.size()==1) {	comment << "              ";      }
  if(conns.size()==2) {	comment << "       ";      }
    //  }
  if(!dqm_store) return flag;
  // get PSU channel corresponding to the det_id
  /*
  comment << " PSU: ";
  comment << psumap_.getPSUName(det_id);
  */
  //

  SiStripFolderOrganizer folder_organizer;
  std::string subdet_folder, badmodule_folder;

  dqm_store->cd();

  folder_organizer.setSiStripFolderName(stripTopLevelDir_);
  folder_organizer.getSubDetFolder(det_id, tTopo, subdet_folder);

  LogDebug("SearchBadModule") << det_id << " " << subdet_folder << " " << stripTopLevelDir_;

  if (dqm_store->dirExists(subdet_folder)){ 
    badmodule_folder = subdet_folder + "/BadModuleList";
    LogDebug("SearchBadModule") << subdet_folder << " exists: " << badmodule_folder;
  } else {
    //    badmodule_folder = dqm_store->pwd() + "/BadModuleList"; 
    edm::LogError("SubDetFolderNotFound") << subdet_folder << " does not exist for detid " << det_id;
    return flag;
  }
  if (!dqm_store->dirExists(badmodule_folder))  {
    LogDebug("BadModuleFolderNotFound") << badmodule_folder << " does not exist for detid " << det_id;
    return flag;
  }
  std::ostringstream badmodule_path;
  badmodule_path << badmodule_folder << "/" << det_id;
  LogDebug("SearchBadModule") << badmodule_folder << " exists: " << badmodule_path;

  MonitorElement* bad_module_me = dqm_store->get(badmodule_path.str());
  if (bad_module_me && bad_module_me->kind() == MonitorElement::DQM_KIND_INT) {
    LogDebug("SearchBadModule") << "Monitor Element found";
    flag = bad_module_me->getIntValue();
    std::string message;
    SiStripUtility::getBadModuleStatus(flag, message);
    comment << message.c_str();
  }
  return flag;
}
