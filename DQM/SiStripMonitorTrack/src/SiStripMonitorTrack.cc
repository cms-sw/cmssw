#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDCSStatus.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DQM/SiStripMonitorTrack/interface/SiStripMonitorTrack.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "TMath.h"

SiStripMonitorTrack::SiStripMonitorTrack(const edm::ParameterSet& conf):
  conf_(conf),
  tracksCollection_in_EventTree(true),
  firstEvent(-1),
  genTriggerEventFlag_(new GenericTriggerEventFlag(conf.getParameter<edm::ParameterSet>("genericTriggerEventPSet"), consumesCollector(), *this))
{
  Cluster_src_   = conf.getParameter<edm::InputTag>("Cluster_src");
  Mod_On_        = conf.getParameter<bool>("Mod_On");
  Trend_On_      = conf.getParameter<bool>("Trend_On");
  TkHistoMap_On_ = conf.getParameter<bool>("TkHistoMap_On");

  TrackProducer_ = conf_.getParameter<std::string>("TrackProducer");
  TrackLabel_    = conf_.getParameter<std::string>("TrackLabel");

  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");

  digiToken_       = consumes<edm::DetSetVector<SiStripDigi>> (  conf.getParameter<edm::InputTag>("ADCDigi_src") );
  clusterToken_    = consumes<edmNew::DetSetVector<SiStripCluster> >(Cluster_src_);
  trackToken_      = consumes<reco::TrackCollection>(edm::InputTag(TrackProducer_,TrackLabel_) );

  // cluster quality conditions
  edm::ParameterSet cluster_condition = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
  applyClusterQuality_ = cluster_condition.getParameter<bool>("On");
  sToNLowerLimit_      = cluster_condition.getParameter<double>("minStoN");
  sToNUpperLimit_      = cluster_condition.getParameter<double>("maxStoN");
  widthLowerLimit_     = cluster_condition.getParameter<double>("minWidth");
  widthUpperLimit_     = cluster_condition.getParameter<double>("maxWidth");


  // Create DCS Status
  bool checkDCS    = conf_.getParameter<bool>("UseDCSFiltering");
  if (checkDCS) dcsStatus_ = new SiStripDCSStatus(consumesCollector());
  else dcsStatus_ = 0;
}

//------------------------------------------------------------------------
SiStripMonitorTrack::~SiStripMonitorTrack() {
  if (dcsStatus_) delete dcsStatus_;
  if (genTriggerEventFlag_) delete genTriggerEventFlag_;
}

//------------------------------------------------------------------------
void SiStripMonitorTrack::dqmBeginRun(const edm::Run& run, const edm::EventSetup& es)
{
  //get geom
  es.get<TrackerDigiGeometryRecord>().get( tkgeom_ );
  LogDebug("SiStripMonitorTrack") << "[SiStripMonitorTrack::beginRun] There are "<<tkgeom_->detUnits().size() <<" detectors instantiated in the geometry" << std::endl;
  es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );


  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() )genTriggerEventFlag_->initRun( run, es );
}

//------------------------------------------------------------------------
void SiStripMonitorTrack::bookHistograms(DQMStore::IBooker & ibooker , const edm::Run & run, const edm::EventSetup & es)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  book(ibooker , tTopo);
}

// ------------ method called to produce the data  ------------
void SiStripMonitorTrack::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  // Filter out events if DCS checking is requested
  if (dcsStatus_ && !dcsStatus_->getStatus(e,es)) return;

  // Filter out events if Trigger Filtering is requested
  if (genTriggerEventFlag_->on()&& ! genTriggerEventFlag_->accept( e, es) ) return;

  //initialization of global quantities
  LogDebug("SiStripMonitorTrack") << "[SiStripMonitorTrack::analyse]  " << "Run " << e.id().run() << " Event " << e.id().event() << std::endl;
  eventNb = e.id().event();
  vPSiStripCluster.clear();

  iLumisection = e.orbitNumber()/262144.0;

  // initialise # of clusters
  for (std::map<std::string, SubDetMEs>::iterator iSubDet = SubDetMEsMap.begin();
       iSubDet != SubDetMEsMap.end(); iSubDet++) {
    iSubDet->second.totNClustersOnTrack = 0;
    iSubDet->second.totNClustersOffTrack = 0;
  }

  //Perform track study
  trackStudy(e, es);

  //Perform Cluster Study (irrespectively to tracks)

   AllClusters(e, es); //analyzes the off Track Clusters

  //Summary Counts of clusters
  std::map<std::string, MonitorElement*>::iterator iME;
  std::map<std::string, LayerMEs>::iterator        iLayerME;

  if (Trend_On_) {
 // for (std::map<std::string, SubDetMEs>::iterator iSubDet = SubDetMEsMap.begin(), iterEnd=SubDetMEsMaps.end();
 //      iSubDet != iterEnd; ++iSubDet) {
    for (auto const &iSubDet : SubDetMEsMap) {
      SubDetMEs subdet_mes = iSubDet.second;
      if (subdet_mes.totNClustersOnTrack > 0) {
        fillME(subdet_mes.nClustersOnTrack, subdet_mes.totNClustersOnTrack);
      }
      fillME(subdet_mes.nClustersOffTrack, subdet_mes.totNClustersOffTrack);
      fillME(subdet_mes.nClustersTrendOnTrack,iLumisection,subdet_mes.totNClustersOnTrack);
      fillME(subdet_mes.nClustersTrendOffTrack,iLumisection,subdet_mes.totNClustersOffTrack);
    }
  } else {
    for (auto const &iSubDet : SubDetMEsMap) {
      SubDetMEs subdet_mes = iSubDet.second;
      if (subdet_mes.totNClustersOnTrack > 0) {
        fillME(subdet_mes.nClustersOnTrack, subdet_mes.totNClustersOnTrack);
      }
      fillME(subdet_mes.nClustersOffTrack, subdet_mes.totNClustersOffTrack);
    }
  }
}

//------------------------------------------------------------------------
void SiStripMonitorTrack::book(DQMStore::IBooker & ibooker , const TrackerTopology* tTopo)
{

  SiStripFolderOrganizer folder_organizer;
  folder_organizer.setSiStripFolderName(topFolderName_);
  //******** TkHistoMaps
  if (TkHistoMap_On_) {
    tkhisto_StoNCorrOnTrack = new TkHistoMap(ibooker , topFolderName_ ,"TkHMap_StoNCorrOnTrack",         0.0,true);
    tkhisto_NumOnTrack      = new TkHistoMap(ibooker , topFolderName_, "TkHMap_NumberOfOnTrackCluster",  0.0,true);
    tkhisto_NumOffTrack     = new TkHistoMap(ibooker , topFolderName_, "TkHMap_NumberOfOfffTrackCluster",0.0,true);
    tkhisto_ClChPerCMfromTrack  = new TkHistoMap(ibooker , topFolderName_, "TkHMap_ChargePerCMfromTrack",0.0,true);
    tkhisto_NumMissingHits      = new TkHistoMap(ibooker , topFolderName_, "TkHMap_NumberMissingHits",0.0,true);
    tkhisto_NumberInactiveHits  = new TkHistoMap(ibooker , topFolderName_, "TkHMap_NumberInactiveHits",0.0,true);
    tkhisto_NumberValidHits     = new TkHistoMap(ibooker , topFolderName_, "TkHMap_NumberValidHits",0.0,true);
  }
  if (clchCMoriginTkHmap_On_)
    tkhisto_ClChPerCMfromOrigin = new TkHistoMap(ibooker , topFolderName_, "TkHMap_ChargePerCMfromOrigin",0.0,true);
  //******** TkHistoMaps

  std::vector<uint32_t> vdetId_;
  SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
  const char* tec = "TEC";
  const char* tid = "TID";
  //Histos for each detector, layer and module
  SiStripHistoId hidmanager;

  if(Mod_On_) {
    for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin(),detid_end=vdetId_.end();detid_iter!=detid_end;++detid_iter){  //loop on all the active detid
      uint32_t detid = *detid_iter;

      if (detid < 1){
        edm::LogError("SiStripMonitorTrack")<< "[" <<__PRETTY_FUNCTION__ << "] invalid detid " << detid<< std::endl;
        continue;
      }


      //std::string name;

      // book Layer and RING plots
      std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid,tTopo,false);
      /*
      std::string thickness;
      std::pair<std::string,int32_t> det_layer_pair_test = folder_organizer.GetSubDetAndLayerThickness(detid,tTopo,thickness);
      std::cout << "[SiStripMonitorTrack::book] det_layer_pair " << det_layer_pair.first << " " << det_layer_pair.second << " " << thickness << std::endl;
      */

      std::string layer_id = hidmanager.getSubdetid(detid, tTopo, false);

      std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(layer_id);
      if(iLayerME==LayerMEsMap.end()){
        folder_organizer.setLayerFolder(detid, tTopo, det_layer_pair.second, false);
        bookLayerMEs(ibooker , detid, layer_id);
      }

      const char* subdet = det_layer_pair.first.c_str();
      if ( std::strstr(subdet, tec) != NULL || std::strstr(subdet, tid) != NULL ) {
        std::string ring_id = hidmanager.getSubdetid(detid, tTopo, true);
        std::map<std::string, RingMEs>::iterator iRingME  = RingMEsMap.find(ring_id);
        if(iRingME==RingMEsMap.end()){
	  std::pair<std::string,int32_t> det_ring_pair = folder_organizer.GetSubDetAndLayer(detid,tTopo,true);
	  folder_organizer.setLayerFolder(detid, tTopo, det_ring_pair.second, true);
	  bookRingMEs(ibooker , detid, ring_id);
        }
      }

      // book sub-detector plots
      std::pair<std::string,std::string> sdet_pair = folder_organizer.getSubDetFolderAndTag(detid, tTopo);
      if (SubDetMEsMap.find(sdet_pair.second) == SubDetMEsMap.end()){
        ibooker.setCurrentFolder(sdet_pair.first);
        bookSubDetMEs(ibooker , sdet_pair.second);
      }
      // book module plots
      folder_organizer.setDetectorFolder(detid,tTopo);
      bookModMEs(ibooker , *detid_iter);
   }//end loop on detectors detid
  } else {
    for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin(),detid_end=vdetId_.end();detid_iter!=detid_end;++detid_iter){  //loop on all the active detid
      uint32_t detid = *detid_iter;

      if (detid < 1){
        edm::LogError("SiStripMonitorTrack")<< "[" <<__PRETTY_FUNCTION__ << "] invalid detid " << detid<< std::endl;
        continue;
      }


      //std::string name;

      // book Layer and RING plots
      std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid,tTopo,false);
      /*
      std::string thickness;
      std::pair<std::string,int32_t> det_layer_pair_test = folder_organizer.GetSubDetAndLayerThickness(detid,tTopo,thickness);
      std::cout << "[SiStripMonitorTrack::book] det_layer_pair " << det_layer_pair.first << " " << det_layer_pair.second << " " << thickness << std::endl;
      */

      std::string layer_id = hidmanager.getSubdetid(detid, tTopo, false);

      std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(layer_id);
      if(iLayerME==LayerMEsMap.end()){
        folder_organizer.setLayerFolder(detid, tTopo, det_layer_pair.second, false);
        bookLayerMEs(ibooker , detid, layer_id);
      }

      const char* subdet = det_layer_pair.first.c_str();
      if ( std::strstr(subdet, tec) != NULL || std::strstr(subdet, tid) != NULL ) {
        std::string ring_id = hidmanager.getSubdetid(detid, tTopo, true);
        std::map<std::string, RingMEs>::iterator iRingME  = RingMEsMap.find(ring_id);
        if(iRingME==RingMEsMap.end()){
	  std::pair<std::string,int32_t> det_ring_pair = folder_organizer.GetSubDetAndLayer(detid,tTopo,true);
	  folder_organizer.setLayerFolder(detid, tTopo, det_ring_pair.second, true);
	  bookRingMEs(ibooker , detid, ring_id);
        }
      }

      // book sub-detector plots
      std::pair<std::string,std::string> sdet_pair = folder_organizer.getSubDetFolderAndTag(detid, tTopo);
      if (SubDetMEsMap.find(sdet_pair.second) == SubDetMEsMap.end()){
        ibooker.setCurrentFolder(sdet_pair.first);
        bookSubDetMEs(ibooker , sdet_pair.second);
      }
    }//end loop on detectors detid
  }
}

//--------------------------------------------------------------------------------
void SiStripMonitorTrack::bookModMEs(DQMStore::IBooker & ibooker , const uint32_t id)//Histograms at MODULE level
{
  std::string name = "det";
  SiStripHistoId hidmanager;
  std::string hid = hidmanager.createHistoId("",name,id);
  std::map<std::string, ModMEs>::iterator iModME  = ModMEsMap.find(hid);
  if(iModME==ModMEsMap.end()){
    ModMEs theModMEs;

    // Cluster Width
    theModMEs.ClusterWidth=bookME1D(ibooker , "TH1ClusterWidth", hidmanager.createHistoId("ClusterWidth_OnTrack",name,id).c_str());
    ibooker.tag(theModMEs.ClusterWidth,id);
    // Cluster Gain
    theModMEs.ClusterGain=bookME1D(ibooker , "TH1ClusterGain", hidmanager.createHistoId("ClusterGain",name,id).c_str());
    ibooker.tag(theModMEs.ClusterGain,id);
    // Cluster Charge
    theModMEs.ClusterCharge=bookME1D(ibooker , "TH1ClusterCharge", hidmanager.createHistoId("ClusterCharge_OnTrack",name,id).c_str());
    ibooker.tag(theModMEs.ClusterCharge,id);
    // Cluster Charge Raw (no gain )
    theModMEs.ClusterChargeRaw=bookME1D(ibooker , "TH1ClusterChargeRaw", hidmanager.createHistoId("ClusterChargeRaw_OnTrack",name,id).c_str());
    ibooker.tag(theModMEs.ClusterChargeRaw,id);
    // Cluster Charge Corrected
    theModMEs.ClusterChargeCorr=bookME1D(ibooker , "TH1ClusterChargeCorr", hidmanager.createHistoId("ClusterChargeCorr_OnTrack",name,id).c_str());
    ibooker.tag(theModMEs.ClusterChargeCorr,id);
    // Cluster StoN Corrected
    theModMEs.ClusterStoNCorr=bookME1D(ibooker , "TH1ClusterStoNCorrMod", hidmanager.createHistoId("ClusterStoNCorr_OnTrack",name,id).c_str());
    ibooker.tag(theModMEs.ClusterStoNCorr,id);
    // Cluster Position
    short total_nr_strips = SiStripDetCabling_->nApvPairs(id) * 2 * 128;
    theModMEs.ClusterPos=ibooker.book1D(hidmanager.createHistoId("ClusterPosition_OnTrack",name,id).c_str(),hidmanager.createHistoId("ClusterPosition_OnTrack",name,id).c_str(),total_nr_strips,0.5,total_nr_strips+0.5);
    ibooker.tag(theModMEs.ClusterPos,id);
    // Cluster PGV
    theModMEs.ClusterPGV=bookMEProfile(ibooker , "TProfileClusterPGV", hidmanager.createHistoId("PGV_OnTrack",name,id).c_str());
    ibooker.tag(theModMEs.ClusterPGV,id);
    // Cluster Charge per cm
    theModMEs.ClusterChargePerCMfromTrack = bookME1D(ibooker , "TH1ClusterChargePerCM", hidmanager.createHistoId("ClusterChargePerCMfromTrack",name,id).c_str());
    ibooker.tag(theModMEs.ClusterChargePerCMfromTrack,id);

    theModMEs.ClusterChargePerCMfromOrigin = bookME1D(ibooker , "TH1ClusterChargePerCM", hidmanager.createHistoId("ClusterChargePerCMfromOrigin",name,id).c_str());
    ibooker.tag(theModMEs.ClusterChargePerCMfromOrigin,id);

    ModMEsMap[hid]=theModMEs;
  }
}

MonitorElement* SiStripMonitorTrack::handleBookMEs(DQMStore::IBooker & ibooker , std::string& viewParameter, std::string& id, std::string& histoParameters, std::string& histoName) {

  MonitorElement* me = NULL;
  bool view = false;
  view = (conf_.getParameter<edm::ParameterSet>(histoParameters.c_str())).getParameter<bool>(viewParameter.c_str());
  if ( id.find("TEC") == std::string::npos && id.find("TID") == std::string::npos ) {
    me = bookME1D(ibooker , histoParameters.c_str(), histoName.c_str());
  } else {
    if (view) {
      //      histoName = histoName + "__" + thickness;
      me = bookME1D(ibooker , histoParameters.c_str(), histoName.c_str());
    }
  }
  return me;
}

//
// -- Book Layer Level Histograms and Trend plots
//
//------------------------------------------------------------------------
void SiStripMonitorTrack::bookLayerMEs(DQMStore::IBooker & ibooker , const uint32_t mod_id, std::string& layer_id)
{
  std::string name = "layer";
  std::string view = "layerView";
  std::string hname;
  std::string hpar;
  SiStripHistoId hidmanager;

  LayerMEs theLayerMEs;

  // Signal/Noise (w/ cluster harge corrected)
  hname = hidmanager.createHistoLayer("Summary_ClusterStoNCorr",name,layer_id,"OnTrack");
  hpar  = "TH1ClusterStoNCorr";
  theLayerMEs.ClusterStoNCorrOnTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  // Cluster Gain
  hname = hidmanager.createHistoLayer("Summary_ClusterGain",name,layer_id,"");
  hpar = "TH1ClusterGain";
  theLayerMEs.ClusterGain = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  // Cluster Charge Corrected
  hname = hidmanager.createHistoLayer("Summary_ClusterChargeCorr",name,layer_id,"OnTrack");
  hpar = "TH1ClusterChargeCorr";
  theLayerMEs.ClusterChargeCorrOnTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  // Cluster Charge (On and Off Track)
  hname = hidmanager.createHistoLayer("Summary_ClusterCharge",name,layer_id,"OnTrack");
  hpar  = "TH1ClusterCharge";
  theLayerMEs.ClusterChargeOnTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterCharge",name,layer_id,"OffTrack");
  hpar  = "TH1ClusterCharge";
  theLayerMEs.ClusterChargeOffTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  // Cluster Charge Raw (On and Off Track)
  hname = hidmanager.createHistoLayer("Summary_ClusterChargeRaw",name,layer_id,"OnTrack");
  hpar  = "TH1ClusterChargeRaw";
  theLayerMEs.ClusterChargeRawOnTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterChargeRaw",name,layer_id,"OffTrack");
  hpar  = "TH1ClusterChargeRaw";
  theLayerMEs.ClusterChargeRawOffTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  // Cluster Noise (On and Off Track)
  hname = hidmanager.createHistoLayer("Summary_ClusterNoise",name,layer_id,"OnTrack");
  hpar  = "TH1ClusterNoise";
  theLayerMEs.ClusterNoiseOnTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterNoise",name,layer_id,"OffTrack");
  hpar  = "TH1ClusterNoise";
  theLayerMEs.ClusterNoiseOffTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  // Cluster Width (On and Off Track)
  hname = hidmanager.createHistoLayer("Summary_ClusterWidth",name,layer_id,"OnTrack");
  hpar  = "TH1ClusterWidth";
  theLayerMEs.ClusterWidthOnTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterWidth",name,layer_id,"OffTrack");
  hpar  = "TH1ClusterWidth";
  theLayerMEs.ClusterWidthOffTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  //Cluster Position
  short total_nr_strips = SiStripDetCabling_->nApvPairs(mod_id) * 2 * 128;
  if (layer_id.find("TEC") != std::string::npos)  total_nr_strips = 3 * 2 * 128;

  hname = hidmanager.createHistoLayer("Summary_ClusterPosition",name,layer_id,"OnTrack");
  hpar = "TH1ClusterPos";
  if ( layer_id.find("TIB") != std::string::npos || layer_id.find("TOB") != std::string::npos || (conf_.getParameter<edm::ParameterSet>(hpar.c_str())).getParameter<bool>(view.c_str()) ) theLayerMEs.ClusterPosOnTrack = ibooker.book1D(hname, hname, total_nr_strips, 0.5,total_nr_strips+0.5);

  hname = hidmanager.createHistoLayer("Summary_ClusterPosition",name,layer_id,"OffTrack");
  hpar = "TH1ClusterPos";
  if ( layer_id.find("TIB") != std::string::npos || layer_id.find("TOB") != std::string::npos || (conf_.getParameter<edm::ParameterSet>(hpar.c_str())).getParameter<bool>(view.c_str()) ) theLayerMEs.ClusterPosOffTrack = ibooker.book1D(hname, hname, total_nr_strips, 0.5,total_nr_strips+0.5);

  // dQ/dx
  hname = hidmanager.createHistoLayer("Summary_ClusterChargePerCMfromTrack",name,layer_id,"");
  hpar  = "TH1ClusterChargePerCM";
  theLayerMEs.ClusterChargePerCMfromTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterChargePerCMfromOrigin",name,layer_id,"OnTrack");
  hpar  = "TH1ClusterChargePerCM";
  theLayerMEs.ClusterChargePerCMfromOriginOnTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterChargePerCMfromOrigin",name,layer_id,"OffTrack");
  hpar  = "TH1ClusterChargePerCM";
  theLayerMEs.ClusterChargePerCMfromOriginOffTrack = handleBookMEs(ibooker,view,layer_id,hpar,hname);

  //bookeeping
  LayerMEsMap[layer_id]=theLayerMEs;

}

void SiStripMonitorTrack::bookRingMEs(DQMStore::IBooker & ibooker , const uint32_t mod_id, std::string& ring_id)
{

  std::string name = "ring";
  std::string view = "ringView";
  std::string hname;
  std::string hpar;
  SiStripHistoId hidmanager;

  RingMEs theRingMEs;

  hname = hidmanager.createHistoLayer("Summary_ClusterStoNCorr",name,ring_id,"OnTrack");
  hpar  = "TH1ClusterStoNCorr";
  theRingMEs.ClusterStoNCorrOnTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  // Cluster Gain
  hname = hidmanager.createHistoLayer("Summary_ClusterGain",name,ring_id,"");
  hpar = "TH1ClusterGain";
  theRingMEs.ClusterGain = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  // Cluster Charge Corrected
  hname = hidmanager.createHistoLayer("Summary_ClusterChargeCorr",name,ring_id,"OnTrack");
  hpar = "TH1ClusterChargeCorr";
  theRingMEs.ClusterChargeCorrOnTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  // Cluster Charge (On and Off Track)
  hname = hidmanager.createHistoLayer("Summary_ClusterCharge",name,ring_id,"OnTrack");
  hpar  = "TH1ClusterCharge";
  theRingMEs.ClusterChargeOnTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterCharge",name,ring_id,"OffTrack");
  hpar  = "TH1ClusterCharge";
  theRingMEs.ClusterChargeOffTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  // Cluster Charge Raw (no-gain), On and off track
  hname = hidmanager.createHistoLayer("Summary_ClusterChargeRaw",name,ring_id,"OnTrack");
  hpar  = "TH1ClusterChargeRaw";
  theRingMEs.ClusterChargeRawOnTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterChargeRaw",name,ring_id,"OffTrack");
  hpar  = "TH1ClusterChargeRaw";
  theRingMEs.ClusterChargeRawOffTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  // Cluster Noise (On and Off Track)
  hname = hidmanager.createHistoLayer("Summary_ClusterNoise",name,ring_id,"OnTrack");
  hpar  = "TH1ClusterNoise";
  theRingMEs.ClusterNoiseOnTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterNoise",name,ring_id,"OffTrack");
  hpar  = "TH1ClusterNoise";
  theRingMEs.ClusterNoiseOffTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  // Cluster Width (On and Off Track)
  hname = hidmanager.createHistoLayer("Summary_ClusterWidth",name,ring_id,"OnTrack");
  hpar  = "TH1ClusterWidth";
  theRingMEs.ClusterWidthOnTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterWidth",name,ring_id,"OffTrack");
  hpar  = "TH1ClusterWidth";
  theRingMEs.ClusterWidthOffTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  //Cluster Position
  short total_nr_strips = SiStripDetCabling_->nApvPairs(mod_id) * 2 * 128;
  if (ring_id.find("TEC") != std::string::npos)  total_nr_strips = 3 * 2 * 128;

  hname = hidmanager.createHistoLayer("Summary_ClusterPosition",name,ring_id,"OnTrack");
  hpar = "TH1ClusterPos";
  if ( (conf_.getParameter<edm::ParameterSet>(hpar.c_str())).getParameter<bool>(view.c_str()) ) theRingMEs.ClusterPosOnTrack = ibooker.book1D(hname, hname, total_nr_strips, 0.5,total_nr_strips+0.5);

  hname = hidmanager.createHistoLayer("Summary_ClusterPosition",name,ring_id,"OffTrack");
  hpar = "TH1ClusterPos";
  if ( (conf_.getParameter<edm::ParameterSet>(hpar.c_str())).getParameter<bool>(view.c_str()) ) theRingMEs.ClusterPosOffTrack = ibooker.book1D(hname, hname, total_nr_strips, 0.5,total_nr_strips+0.5);

  // dQ/dx
  hname = hidmanager.createHistoLayer("Summary_ClusterChargePerCMfromTrack",name,ring_id,"");
  hpar  = "TH1ClusterChargePerCM";
  theRingMEs.ClusterChargePerCMfromTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterChargePerCMfromOrigin",name,ring_id,"OnTrack");
  hpar  = "TH1ClusterChargePerCM";
  theRingMEs.ClusterChargePerCMfromOriginOnTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  hname = hidmanager.createHistoLayer("Summary_ClusterChargePerCMfromOrigin",name,ring_id,"OffTrack");
  hpar  = "TH1ClusterChargePerCM";
  theRingMEs.ClusterChargePerCMfromOriginOffTrack = handleBookMEs(ibooker,view,ring_id,hpar,hname);

  //bookeeping
  RingMEsMap[ring_id]=theRingMEs;

}
//------------------------------------------------------------------------
//
// -- Book Histograms at Sub-Detector Level
//
void SiStripMonitorTrack::bookSubDetMEs(DQMStore::IBooker & ibooker , std::string& name){

  std::string subdet_tag;
  subdet_tag = "__" + name;
  std::string completeName;
  std::string axisName;

  SubDetMEs theSubDetMEs;

  // TotalNumber of Cluster OnTrack
  completeName = "Summary_TotalNumberOfClusters_OnTrack" + subdet_tag;
  axisName = "Number of on-track clusters in " + name;
  theSubDetMEs.nClustersOnTrack = bookME1D(ibooker , "TH1nClustersOn", completeName.c_str());
  theSubDetMEs.nClustersOnTrack->setAxisTitle(axisName.c_str());
  theSubDetMEs.nClustersOnTrack->getTH1()->StatOverflows(kTRUE);

  // TotalNumber of Cluster OffTrack
  completeName = "Summary_TotalNumberOfClusters_OffTrack" + subdet_tag;
  axisName = "Number of off-track clusters in " + name;
  theSubDetMEs.nClustersOffTrack = bookME1D(ibooker , "TH1nClustersOff", completeName.c_str());
  theSubDetMEs.nClustersOffTrack->setAxisTitle(axisName.c_str());
  theSubDetMEs.nClustersOffTrack->getTH1()->StatOverflows(kTRUE);

  // Cluster Gain
  completeName = "Summary_ClusterGain" + subdet_tag;
  theSubDetMEs.ClusterGain=bookME1D(ibooker , "TH1ClusterGain", completeName.c_str());

  // Cluster StoN On Track
  completeName = "Summary_ClusterStoNCorr_OnTrack"  + subdet_tag;
  theSubDetMEs.ClusterStoNCorrOnTrack = bookME1D(ibooker , "TH1ClusterStoNCorr", completeName.c_str());

  completeName = "Summary_ClusterStoNCorrThin_OnTrack"  + subdet_tag;
  if ( subdet_tag.find("TEC") != std::string::npos ) theSubDetMEs.ClusterStoNCorrThinOnTrack = bookME1D(ibooker , "TH1ClusterStoNCorr", completeName.c_str());

  completeName = "Summary_ClusterStoNCorrThick_OnTrack"  + subdet_tag;
  if ( subdet_tag.find("TEC") != std::string::npos ) theSubDetMEs.ClusterStoNCorrThickOnTrack = bookME1D(ibooker , "TH1ClusterStoNCorr", completeName.c_str());

  // Cluster Charge Corrected
  completeName = "Summary_ClusterChargeCorr_OnTrack"  + subdet_tag;
  theSubDetMEs.ClusterChargeCorrOnTrack = bookME1D(ibooker , "TH1ClusterChargeCorr", completeName.c_str());

  completeName = "Summary_ClusterChargeCorrThin_OnTrack"  + subdet_tag;
  if ( subdet_tag.find("TEC") != std::string::npos ) theSubDetMEs.ClusterChargeCorrThinOnTrack = bookME1D(ibooker , "TH1ClusterChargeCorr", completeName.c_str());

  completeName = "Summary_ClusterChargeCorrThick_OnTrack"  + subdet_tag;
  if ( subdet_tag.find("TEC") != std::string::npos ) theSubDetMEs.ClusterChargeCorrThickOnTrack = bookME1D(ibooker , "TH1ClusterChargeCorr", completeName.c_str());

  // Cluster Charge On Track
  completeName = "Summary_ClusterCharge_OnTrack" + subdet_tag;
  theSubDetMEs.ClusterChargeOnTrack=bookME1D(ibooker , "TH1ClusterCharge", completeName.c_str());

  // Cluster Charge On Track, Raw (no-gain)
  completeName = "Summary_ClusterChargeRaw_OnTrack" + subdet_tag;
  theSubDetMEs.ClusterChargeRawOnTrack=bookME1D(ibooker , "TH1ClusterChargeRaw", completeName.c_str());

  // Cluster Charge Off Track
  completeName = "Summary_ClusterCharge_OffTrack" + subdet_tag;
  theSubDetMEs.ClusterChargeOffTrack=bookME1D(ibooker , "TH1ClusterCharge", completeName.c_str());

  // Cluster Charge Off Track, Raw (no-gain)
  completeName = "Summary_ClusterChargeRaw_OffTrack" + subdet_tag;
  theSubDetMEs.ClusterChargeRawOffTrack=bookME1D(ibooker , "TH1ClusterChargeRaw", completeName.c_str());

  // Cluster Charge StoN Off Track
  completeName = "Summary_ClusterStoN_OffTrack"  + subdet_tag;
  theSubDetMEs.ClusterStoNOffTrack = bookME1D(ibooker , "TH1ClusterStoN", completeName.c_str());

  // cluster charge per cm on track
  completeName = "Summary_ClusterChargePerCMfromTrack" + subdet_tag;
  theSubDetMEs.ClusterChargePerCMfromTrack=bookME1D(ibooker , "TH1ClusterChargePerCM", completeName.c_str());

  // cluster charge per cm on track
  completeName = "Summary_ClusterChargePerCMfromOrigin_OnTrack" + subdet_tag;
  theSubDetMEs.ClusterChargePerCMfromOriginOnTrack=bookME1D(ibooker , "TH1ClusterChargePerCM", completeName.c_str());

  // cluster charge per cm off track
  completeName = "Summary_ClusterChargePerCMfromOrigin_OffTrack" + subdet_tag;
  theSubDetMEs.ClusterChargePerCMfromOriginOffTrack=bookME1D(ibooker , "TH1ClusterChargePerCM", completeName.c_str());

  if(Trend_On_){
    // TotalNumber of Cluster
    completeName = "Trend_TotalNumberOfClusters_OnTrack"  + subdet_tag;
    theSubDetMEs.nClustersTrendOnTrack = bookMETrend(ibooker , completeName.c_str());
    completeName = "Trend_TotalNumberOfClusters_OffTrack"  + subdet_tag;
    theSubDetMEs.nClustersTrendOffTrack = bookMETrend(ibooker , completeName.c_str());
  }

  //bookeeping
  SubDetMEsMap[name]=theSubDetMEs;

}
//--------------------------------------------------------------------------------

inline MonitorElement* SiStripMonitorTrack::bookME1D(DQMStore::IBooker & ibooker , const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return ibooker.book1D(HistoName,HistoName,
		         Parameters.getParameter<int32_t>("Nbinx"),
		         Parameters.getParameter<double>("xmin"),
		         Parameters.getParameter<double>("xmax")
		    );
}

//--------------------------------------------------------------------------------
inline MonitorElement* SiStripMonitorTrack::bookME2D(DQMStore::IBooker & ibooker , const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return ibooker.book2D(HistoName,HistoName,
		     Parameters.getParameter<int32_t>("Nbinx"),
		     Parameters.getParameter<double>("xmin"),
		     Parameters.getParameter<double>("xmax"),
		     Parameters.getParameter<int32_t>("Nbiny"),
		     Parameters.getParameter<double>("ymin"),
		     Parameters.getParameter<double>("ymax")
		     );
}

//--------------------------------------------------------------------------------
inline MonitorElement* SiStripMonitorTrack::bookME3D(DQMStore::IBooker & ibooker , const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return ibooker.book3D(HistoName,HistoName,
		     Parameters.getParameter<int32_t>("Nbinx"),
		     Parameters.getParameter<double>("xmin"),
		     Parameters.getParameter<double>("xmax"),
		     Parameters.getParameter<int32_t>("Nbiny"),
		     Parameters.getParameter<double>("ymin"),
		     Parameters.getParameter<double>("ymax"),
		     Parameters.getParameter<int32_t>("Nbinz"),
		     Parameters.getParameter<double>("zmin"),
		     Parameters.getParameter<double>("zmax")
		     );
}

//--------------------------------------------------------------------------------
inline MonitorElement* SiStripMonitorTrack::bookMEProfile(DQMStore::IBooker & ibooker , const char* ParameterSetLabel, const char* HistoName)
{
    Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
    return ibooker.bookProfile(HistoName,HistoName,
                            Parameters.getParameter<int32_t>("Nbinx"),
                            Parameters.getParameter<double>("xmin"),
                            Parameters.getParameter<double>("xmax"),
                            Parameters.getParameter<int32_t>("Nbiny"),
                            Parameters.getParameter<double>("ymin"),
                            Parameters.getParameter<double>("ymax"),
                            "" );
}

//--------------------------------------------------------------------------------
MonitorElement* SiStripMonitorTrack::bookMETrend(DQMStore::IBooker & ibooker , const char* HistoName)
{
  edm::ParameterSet ParametersTrend =  conf_.getParameter<edm::ParameterSet>("Trending");
  MonitorElement* me = ibooker.bookProfile(HistoName,HistoName,
					   ParametersTrend.getParameter<int32_t>("Nbins"),
					   ParametersTrend.getParameter<double>("xmin"),
					   ParametersTrend.getParameter<double>("xmax"),
					   0 , 0 , "" );
  if (me->kind() == MonitorElement::DQM_KIND_TPROFILE) me->getTH1()->SetCanExtend(TH1::kAllAxes);

  if(!me) return me;
  me->setAxisTitle("Lumisection",1);
  return me;
}

//------------------------------------------------------------------------------------------
void SiStripMonitorTrack::trajectoryStudy(
  const reco::Track & track,
  const edm::Event&      ev,
  const edm::EventSetup& es,
  bool track_ok
) {

  auto const & trajParams = track.extra()->trajParams();
  assert(trajParams.size()==track.recHitsSize());
  auto hb = track.recHitsBegin();
  for(unsigned int h=0;h<track.recHitsSize();h++){
    auto ttrh = *(hb+h);


    if (TkHistoMap_On_ ) {
      uint32_t thedetid=ttrh->rawId();
      if ( SiStripDetId(thedetid).subDetector() >=3 &&  SiStripDetId(thedetid).subDetector() <=6) { //TIB/TID + TOB + TEC only
        if ( (ttrh->getType()==1) )
          tkhisto_NumMissingHits->add(thedetid,1.);
        if ( (ttrh->getType()==2) )
          tkhisto_NumberInactiveHits->add(thedetid,1.);
        if ( (ttrh->getType()==0) )
          tkhisto_NumberValidHits->add(thedetid,1.);
      }
    }

    if (!ttrh->isValid()) continue;

    //trajectory local direction and position on detector
    auto statedirection = trajParams[h].momentum();


    const ProjectedSiStripRecHit2D* projhit  = dynamic_cast<const ProjectedSiStripRecHit2D*>( ttrh->hit() );
    const SiStripMatchedRecHit2D* matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>( ttrh->hit() );
    const SiStripRecHit2D* hit2D             = dynamic_cast<const SiStripRecHit2D*>( ttrh->hit() );
    const SiStripRecHit1D* hit1D             = dynamic_cast<const SiStripRecHit1D*>( ttrh->hit() );

    //      RecHitType type=Single;

    if(matchedhit){
      LogTrace("SiStripMonitorTrack")<<"\nMatched recHit found"<< std::endl;
      //	type=Matched;

      const GluedGeomDet * gdet=static_cast<const GluedGeomDet *>(tkgeom_->idToDet(matchedhit->geographicalId()));
      GlobalVector gtrkdirup=gdet->toGlobal(statedirection);
      //mono side
      const GeomDetUnit * monodet=gdet->monoDet();
      statedirection=monodet->toLocal(gtrkdirup);
      SiStripRecHit2D m = matchedhit->monoHit();
      //      if(statedirection.mag() != 0)	  RecHitInfo<SiStripRecHit2D>(&m,statedirection,trackref,es);
      if(statedirection.mag())	  RecHitInfo<SiStripRecHit2D>(&m,statedirection,ev,es,track_ok);
      //stereo side
      const GeomDetUnit * stereodet=gdet->stereoDet();
      statedirection=stereodet->toLocal(gtrkdirup);
      SiStripRecHit2D s = matchedhit->stereoHit();
      //      if(statedirection.mag() != 0)	  RecHitInfo<SiStripRecHit2D>(&s,statedirection,trackref,es);
      if(statedirection.mag())	  RecHitInfo<SiStripRecHit2D>(&s,statedirection,ev,es,track_ok);
    }
    else if(projhit){
      LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found"<< std::endl;
      //	type=Projected;
      const GluedGeomDet * gdet=static_cast<const GluedGeomDet *>(tkgeom_->idToDet(projhit->geographicalId()));

      GlobalVector gtrkdirup=gdet->toGlobal(statedirection);
      const SiStripRecHit2D  originalhit=projhit->originalHit();
      const GeomDetUnit * det;
      if(!StripSubdetector(originalhit.geographicalId().rawId()).stereo()){
	//mono side
	LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found  MONO"<< std::endl;
	det=gdet->monoDet();
	statedirection=det->toLocal(gtrkdirup);
	if(statedirection.mag()) RecHitInfo<SiStripRecHit2D>(&(originalhit),statedirection,ev,es,track_ok);
      }
      else{
	LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found STEREO"<< std::endl;
	//stereo side
	det=gdet->stereoDet();
	statedirection=det->toLocal(gtrkdirup);
	if(statedirection.mag()) RecHitInfo<SiStripRecHit2D>(&(originalhit),statedirection,ev,es,track_ok);
      }
    }else if (hit2D){
      if(statedirection.mag()) RecHitInfo<SiStripRecHit2D>(hit2D,statedirection,ev,es,track_ok);
    } else if (hit1D) {
      if(statedirection.mag()) RecHitInfo<SiStripRecHit1D>(hit1D,statedirection,ev,es,track_ok);
    } else {
      LogDebug ("SiStripMonitorTrack")
	<< " LocalMomentum: "<<statedirection
	<< "\nLocal x-z plane angle: "<<atan2(statedirection.x(),statedirection.z());
    }

  }

}
//------------------------------------------------------------------------
void SiStripMonitorTrack::hitStudy(
  const edm::Event& ev,
  const edm::EventSetup& es,
	const ProjectedSiStripRecHit2D* projhit,
	const SiStripMatchedRecHit2D*   matchedhit,
	const SiStripRecHit2D*          hit2D,
	const SiStripRecHit1D*          hit1D,
	      LocalVector               localMomentum,
	const bool                      track_ok
) {
  LocalVector statedirection;
  if(matchedhit){     // type=Matched;
    LogTrace("SiStripMonitorTrack")<<"\nMatched recHit found"<< std::endl;

    const GluedGeomDet * gdet=static_cast<const GluedGeomDet *>(tkgeom_->idToDet(matchedhit->geographicalId()));

    GlobalVector gtrkdirup=gdet->toGlobal(localMomentum);

    //mono side
    const GeomDetUnit * monodet=gdet->monoDet();
    statedirection=monodet->toLocal(gtrkdirup);
    SiStripRecHit2D m = matchedhit->monoHit();
    if(statedirection.mag())	  RecHitInfo<SiStripRecHit2D>(&m,statedirection,ev,es,track_ok);

    //stereo side
    const GeomDetUnit * stereodet=gdet->stereoDet();
    statedirection=stereodet->toLocal(gtrkdirup);
    SiStripRecHit2D s = matchedhit->stereoHit();
    if(statedirection.mag())	  RecHitInfo<SiStripRecHit2D>(&s,statedirection,ev,es,track_ok);
  }
  else if(projhit){    // type=Projected;
      LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found"<< std::endl;

      const GluedGeomDet * gdet=static_cast<const GluedGeomDet *>(tkgeom_->idToDet(projhit->geographicalId()));

      GlobalVector gtrkdirup=gdet->toGlobal(localMomentum);
      const SiStripRecHit2D  originalhit=projhit->originalHit();

      const GeomDetUnit * det;
      if(!StripSubdetector(originalhit.geographicalId().rawId()).stereo()){
	//mono side
	LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found  MONO"<< std::endl;
	det=gdet->monoDet();
	statedirection=det->toLocal(gtrkdirup);
	if(statedirection.mag()) RecHitInfo<SiStripRecHit2D>(&(originalhit),statedirection,ev,es,track_ok);
      }
      else{
	LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found STEREO"<< std::endl;
	//stereo side
	det=gdet->stereoDet();
	statedirection=det->toLocal(gtrkdirup);
	if(statedirection.mag()) RecHitInfo<SiStripRecHit2D>(&(originalhit),statedirection,ev,es,track_ok);
      }
  } else if (hit2D){ // type=2D
    statedirection=localMomentum;
    if(statedirection.mag()) RecHitInfo<SiStripRecHit2D>(hit2D,statedirection,ev,es,track_ok);
  } else if (hit1D) { // type=1D
    statedirection=localMomentum;
    if(statedirection.mag()) RecHitInfo<SiStripRecHit1D>(hit1D,statedirection,ev,es,track_ok);
  } else {
    LogDebug ("SiStripMonitorTrack")
      << " LocalMomentum: "<<statedirection
      << "\nLocal x-z plane angle: "<<atan2(statedirection.x(),statedirection.z());
  }

}
//------------------------------------------------------------------------
void SiStripMonitorTrack::trackStudy(const edm::Event& ev, const edm::EventSetup& es){

using namespace std;
using namespace edm;
using namespace reco;


  //    edm::Handle<std::vector<Trajectory> > trajectories;
  //    ev.getByToken(trajectoryToken_, trajectories);

  // track input
  edm::Handle<reco::TrackCollection > trackCollectionHandle;
  ev.getByToken(trackToken_, trackCollectionHandle);//takes the track collection
  if (trackCollectionHandle.isValid()){
    trackStudyFromTrajectory(trackCollectionHandle,ev, es);
  } else {
    edm::LogError("SiStripMonitorTrack")<<"also Track Collection is not valid !! " << TrackLabel_<<std::endl;
    return;
  }
}

//------------------------------------------------------------------------
// Should return true if the track is good, false if it should be discarded.
bool SiStripMonitorTrack::trackFilter(const reco::Track& track) {
  if (track.pt() < 0.8) return false;
  if (track.p()  < 2.0) return false;
  if (track.hitPattern().numberOfValidTrackerHits()  <= 6) return false;
  if (track.normalizedChi2() > 10.0) return false;
  return true;
}
//------------------------------------------------------------------------
void SiStripMonitorTrack::trackStudyFromTrack(
  edm::Handle<reco::TrackCollection > trackCollectionHandle,
  const edm::Event& ev,
  const edm::EventSetup& es
) {

  //  edm::ESHandle<TransientTrackBuilder> builder;
  //  es.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
  //  const TransientTrackBuilder* transientTrackBuilder = builder.product();

  //numTracks = trackCollectionHandle->size();
  reco::TrackCollection trackCollection = *trackCollectionHandle;
  for (reco::TrackCollection::const_iterator track = trackCollection.begin(), etrack = trackCollection.end();
       track!=etrack; ++track) {

    bool track_ok = trackFilter(*track);
    //    const reco::TransientTrack transientTrack = transientTrackBuilder->build(track);

    for (trackingRecHit_iterator hit = track->recHitsBegin(), ehit = track->recHitsEnd();
	 hit!=ehit; ++hit) {

      if (TkHistoMap_On_ ) {
        uint32_t thedetid=(*hit)->rawId();
        if ( SiStripDetId(thedetid).subDetector() >=3 &&  SiStripDetId(thedetid).subDetector() <=6) { //TIB/TID + TOB + TEC only
          if ( ((*hit)->getType()==1) )
            tkhisto_NumMissingHits->add(thedetid,1.);
          if ( ((*hit)->getType()==2) )
            tkhisto_NumberInactiveHits->add(thedetid,1.);
          if ( ((*hit)->getType()==0) )
            tkhisto_NumberValidHits->add(thedetid,1.);
        }
      }

      if (!(*hit)->isValid()) continue;
      DetId detID = (*hit)->geographicalId();
      if (detID.det() != DetId::Tracker) continue;
      const TrackingRecHit* theHit = (*hit);
      const ProjectedSiStripRecHit2D* projhit    = dynamic_cast<const ProjectedSiStripRecHit2D*>( (theHit) );
      const SiStripMatchedRecHit2D*   matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>  ( (theHit) );
      const SiStripRecHit2D*          hit2D      = dynamic_cast<const SiStripRecHit2D*>         ( (theHit) );
      const SiStripRecHit1D*          hit1D      = dynamic_cast<const SiStripRecHit1D*>         ( (theHit) );

      //      GlobalPoint globalPoint = hit->globalPosition();
      //      reco::TrajectoryStateOnSurface stateOnSurface = transientTrack->stateOnSurface(globalPoint);

      LocalVector localMomentum;
      hitStudy(ev,es,projhit,matchedhit,hit2D,hit1D,localMomentum,track_ok);
    }

    // hit pattern of the track
   // const reco::HitPattern & hitsPattern = track->hitPattern();
    // loop over the hits of the track
    //    for (int i=0; i<hitsPattern.numberOfHits(); i++) {
   // for (int i=0; i<hitsPattern.numberOfHits(reco::HitPattern::TRACK_HITS); i++) {
   //   uint32_t hit = hitsPattern.getHitPattern(reco::HitPattern::TRACK_HITS,i);

      // if the hit is valid and in pixel barrel, print out the layer
   //   if (hitsPattern.validHitFilter(hit) && hitsPattern.pixelBarrelHitFilter(hit))

   //   if (!hitsPattern.validHitFilter(hit)) continue;
//      if (hitsPattern.pixelHitFilter(hit))       std::cout << "pixel"        << std::endl;       // pixel
//      if (hitsPattern.pixelBarrelHitFilter(hit)) std::cout << "pixel barrel" << std::endl; // pixel barrel
//      if (hitsPattern.pixelEndcapHitFilter(hit)) std::cout << "pixel endcap" << std::endl; // pixel endcap
//      if (hitsPattern.stripHitFilter(hit))       std::cout << "strip" << std::endl;       // strip
//      if (hitsPattern.stripTIBHitFilter(hit))    std::cout << "TIB" << std::endl;    // strip TIB
//      if (hitsPattern.stripTIDHitFilter(hit))    std::cout << "TID" << std::endl;    // strip TID
//      if (hitsPattern.stripTOBHitFilter(hit))    std::cout << "TOB" << std::endl;    // strip TOB
//      if (hitsPattern.stripTECHitFilter(hit))    std::cout << "TEC" << std::endl;    // strip TEC
//      if (hitsPattern.muonDTHitFilter(hit))      std::cout << "DT"  << std::endl;      // muon DT
//      if (hitsPattern.muonCSCHitFilter(hit))     std::cout << "CSC" << std::endl;     // muon CSC
//      if (hitsPattern.muonRPCHitFilter(hit))     std::cout << "RPC" << std::endl;     // muon RPC
//
//      // expert level: printout the hit in 10-bit binary format
//      std::cout << "hit in 10-bit binary format = ";
//      for (int j=9; j>=0; j--) {
//        int bit = (hit >> j) & 0x1;
//        std::cout << bit;
//      }
//      std::cout << std::endl;
   // }
  }
}
//------------------------------------------------------------------------
void SiStripMonitorTrack::trackStudyFromTrajectory(
  edm::Handle<reco::TrackCollection > trackCollectionHandle,
  const edm::Event& ev,
  const edm::EventSetup& es
) {
  //Perform track study
  int i=0;
  reco::TrackCollection trackCollection = *trackCollectionHandle;
  numTracks = trackCollection.size();
  for (reco::TrackCollection::const_iterator track = trackCollection.begin(), etrack = trackCollection.end();
       track!=etrack; ++track) {

    LogDebug("SiStripMonitorTrack")
      << "Track number "<< ++i << std::endl;
      //      << "\n\tmomentum: " << trackref->momentum()
      //      << "\n\tPT: " << trackref->pt()
      //      << "\n\tvertex: " << trackref->vertex()
      //      << "\n\timpact parameter: " << trackref->d0()
      //      << "\n\tcharge: " << trackref->charge()
      //      << "\n\tnormalizedChi2: " << trackref->normalizedChi2()
      //      <<"\n\tFrom EXTRA : "
      //      <<"\n\t\touter PT "<< trackref->outerPt()<<std::endl;

    //    trajectoryStudy(traj_iterator,trackref,es);
    bool track_ok = trackFilter(*track);
    trajectoryStudy(*track,ev,es,track_ok);

  }
}
//------------------------------------------------------------------------
template <class T> void SiStripMonitorTrack::RecHitInfo(const T* tkrecHit, LocalVector LV, const edm::Event& ev,  const edm::EventSetup& es, bool track_ok){

    if(!tkrecHit->isValid()){
      LogTrace("SiStripMonitorTrack") <<"\t\t Invalid Hit " << std::endl;
      return;
    }

    const uint32_t& detid = tkrecHit->geographicalId().rawId();

    LogTrace("SiStripMonitorTrack")
      <<"\n\t\tRecHit on det "<<detid
      <<"\n\t\tRecHit in LP "<<tkrecHit->localPosition()
      <<"\n\t\tRecHit in GP "<<tkgeom_->idToDet(tkrecHit->geographicalId())->surface().toGlobal(tkrecHit->localPosition())
      <<"\n\t\tRecHit trackLocal vector "<<LV.x() << " " << LV.y() << " " << LV.z() <<std::endl;


    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHandle;
    es.get<TrackerTopologyRcd>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();

    // Getting SiStrip Gain settings
    edm::ESHandle<SiStripGain>  gainHandle;
    es.get<SiStripGainRcd>().get( gainHandle );
    const SiStripGain* const stripGain = gainHandle.product();

    edm::ESHandle<SiStripQuality> qualityHandle;
    es.get<SiStripQualityRcd>().get("",qualityHandle);
    const SiStripQuality* stripQuality = qualityHandle.product();

    edm::Handle<edm::DetSetVector<SiStripDigi>> digihandle;
    ev.getByToken( digiToken_, digihandle );

    //Get SiStripCluster from SiStripRecHit
    if ( tkrecHit != NULL ){
      const SiStripCluster* SiStripCluster_ = &*(tkrecHit->cluster());
      SiStripClusterInfo SiStripClusterInfo_(*SiStripCluster_,es,detid);

      const Det2MEs MEs = findMEs(tTopo, detid);
      if (clusterInfos(&SiStripClusterInfo_,detid, OnTrack, track_ok, LV, MEs, tTopo,stripGain,stripQuality,*digihandle))
      {
        vPSiStripCluster.insert(SiStripCluster_);
      }
    }
    else
    {
     edm::LogError("SiStripMonitorTrack") << "NULL hit" << std::endl;
    }
  }

//------------------------------------------------------------------------
void SiStripMonitorTrack::AllClusters(const edm::Event& ev, const edm::EventSetup& es)
{

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::ESHandle<SiStripGain>   gainHandle;
  es.get<SiStripGainRcd>().get( gainHandle );
  const SiStripGain*   stripGain = gainHandle.product();

  edm::ESHandle<SiStripQuality> qualityHandle;
  es.get<SiStripQualityRcd>().get("",qualityHandle);
  const SiStripQuality* stripQuality = qualityHandle.product();

  edm::Handle<edm::DetSetVector<SiStripDigi>> digihandle;
  ev.getByToken( digiToken_, digihandle );

  edm::Handle< edmNew::DetSetVector<SiStripCluster> > siStripClusterHandle;
  ev.getByToken( clusterToken_, siStripClusterHandle);
  if (siStripClusterHandle.isValid()){
      //Loop on Dets
    for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=siStripClusterHandle->begin(), DSVEnd=siStripClusterHandle->end();
         DSViter!=DSVEnd; ++DSViter) {

      uint32_t detid=DSViter->id();
      const Det2MEs MEs = findMEs(tTopo, detid);

      LogDebug("SiStripMonitorTrack") << "on detid "<< detid << " N Cluster= " << DSViter->size();

      //Loop on Clusters
      for(edmNew::DetSet<SiStripCluster>::const_iterator ClusIter = DSViter->begin(), ClusEnd = DSViter->end();
          ClusIter!=ClusEnd; ++ClusIter) {

        if (vPSiStripCluster.find(&*ClusIter) == vPSiStripCluster.end()) {
          SiStripClusterInfo SiStripClusterInfo_(*ClusIter,es,detid);
          clusterInfos(&SiStripClusterInfo_,detid,OffTrack, /*track_ok*/ false,LV,MEs, tTopo,stripGain,stripQuality,*digihandle);
        }
      }
    }
  } else {
    edm::LogError("SiStripMonitorTrack")<< "ClusterCollection is not valid!!" << std::endl;
    return;
  }
}

//------------------------------------------------------------------------
SiStripMonitorTrack::Det2MEs SiStripMonitorTrack::findMEs(const TrackerTopology* tTopo, const uint32_t detid) {
  SiStripHistoId hidmanager1;

  std::string layer_id = hidmanager1.getSubdetid(detid, tTopo, false);
  std::string ring_id  = hidmanager1.getSubdetid(detid, tTopo, true);
  std::string sdet_tag = folderOrganizer_.getSubDetFolderAndTag(detid, tTopo).second;

  Det2MEs me;
  me.iLayer = nullptr;
  me.iRing = nullptr;
  me.iSubdet = nullptr;

  std::map<std::string, LayerMEs>::iterator iLayer  = LayerMEsMap.find(layer_id);
  if (iLayer != LayerMEsMap.end()) {
    me.iLayer = &(iLayer->second);
  }

  std::map<std::string, RingMEs>::iterator iRing  = RingMEsMap.find(ring_id);
  if (iRing != RingMEsMap.end()) {
    me.iRing = &(iRing->second);
  }

  std::map<std::string, SubDetMEs>::iterator iSubdet  = SubDetMEsMap.find(sdet_tag);
  if (iSubdet != SubDetMEsMap.end()) {
    me.iSubdet = &(iSubdet->second);
  }

  return me;
}

//------------------------------------------------------------------------
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
bool SiStripMonitorTrack::clusterInfos(
  SiStripClusterInfo* cluster,
  const uint32_t detid,
  enum ClusterFlags flag,
  bool track_ok,
  const LocalVector LV,
  const Det2MEs& MEs ,
  const TrackerTopology* tTopo,
  const SiStripGain*     stripGain,
  const SiStripQuality*  stripQuality,
  const edm::DetSetVector<SiStripDigi>& digilist
)
{

  if (cluster==NULL) return false;
  // if one imposes a cut on the clusters, apply it
  if( (applyClusterQuality_) &&
      (cluster->signalOverNoise() < sToNLowerLimit_ ||
       cluster->signalOverNoise() > sToNUpperLimit_ ||
       cluster->width() < widthLowerLimit_ ||
       cluster->width() > widthUpperLimit_) ) return false;
  // start of the analysis

  float cosRZ = -2;
  LogDebug("SiStripMonitorTrack")<< "\n\tLV " << LV.x() << " " << LV.y() << " " << LV.z() << " " << LV.mag() << std::endl;
  if (LV.mag()){
    cosRZ= fabs(LV.z())/LV.mag();
    LogDebug("SiStripMonitorTrack")<< "\n\t cosRZ " << cosRZ << std::endl;
  }

  // Filling SubDet/Layer Plots (on Track + off Track)
  float    StoN     = cluster->signalOverNoise();
  float    noise    = cluster->noiseRescaledByGain();
  uint16_t charge   = cluster->charge();
  uint16_t width    = cluster->width();
  float    position = cluster->baryStrip();

  // Getting raw charge with strip gain.
  double chargeraw   = 0;
  double clustergain = 0 ;
  auto   digi_it     = digilist.find(detid);
  // SiStripClusterInfo.stripCharges() <==> SiStripCluster.amplitudes()
  for( size_t chidx = 0 ; chidx < cluster->stripCharges().size() ; ++chidx ){
    if( cluster->stripCharges().at(chidx) <= 0 ){ continue ; } // nonzero amplitude
    if( stripQuality->IsStripBad(stripQuality->getRange(detid), cluster->firstStrip()+chidx)) { continue ; }
    clustergain += stripGain->getStripGain(cluster->firstStrip()+chidx, stripGain->getRange(detid));
    // Getting raw adc charge from digi collections
    if( digi_it == digilist.end() ){ continue; } // skipping if not found
    for( const auto& digiobj : *digi_it ){
      if( digiobj.strip() == cluster->firstStrip() + chidx  ){
        chargeraw += digiobj.adc();
      }
    }
  }
  clustergain /= double(cluster->stripCharges().size()) ;  // calculating average gain inside cluster


  // new dE/dx (chargePerCM)
  // https://indico.cern.ch/event/342236/session/5/contribution/10/material/slides/0.pdf
  float dQdx_fromTrack = siStripClusterTools::chargePerCM(detid, *cluster, LV);
  // from straigth line origin-sensor centre
  const StripGeomDetUnit* DetUnit = static_cast<const StripGeomDetUnit*>(tkgeom_->idToDetUnit(DetId(detid)));
  LocalPoint locVtx = DetUnit->toLocal(GlobalPoint(0.0, 0.0, 0.0));
  LocalVector locDir(locVtx.x(), locVtx.y(), locVtx.z());
  float dQdx_fromOrigin = siStripClusterTools::chargePerCM(detid, *cluster, locDir);

  if (TkHistoMap_On_ && (flag == OnTrack)) {
      uint32_t adet=cluster->detId();
      if(track_ok) tkhisto_ClChPerCMfromTrack->fill(adet,dQdx_fromTrack);
  }
  if (clchCMoriginTkHmap_On_ && (flag == OffTrack)){
    uint32_t adet=cluster->detId();
    if(track_ok) tkhisto_ClChPerCMfromOrigin->fill(adet,dQdx_fromOrigin);
  }

  if(flag==OnTrack){
    if (MEs.iSubdet != nullptr) MEs.iSubdet->totNClustersOnTrack++;
    // layerMEs
    if (MEs.iLayer != nullptr) {
      if(noise > 0.0) fillME(MEs.iLayer->ClusterStoNCorrOnTrack, StoN*cosRZ);
      if(noise == 0.0) LogDebug("SiStripMonitorTrack") << "Module " << detid << " in Event " << eventNb << " noise " << cluster->noiseRescaledByGain() << std::endl;
      fillME(MEs.iLayer->ClusterGain, clustergain);
      fillME(MEs.iLayer->ClusterChargeCorrOnTrack, charge*cosRZ);
      fillME(MEs.iLayer->ClusterChargeOnTrack, charge);
      fillME(MEs.iLayer->ClusterChargeRawOnTrack, chargeraw);
      fillME(MEs.iLayer->ClusterNoiseOnTrack, noise);
      fillME(MEs.iLayer->ClusterWidthOnTrack, width);
      fillME(MEs.iLayer->ClusterPosOnTrack, position);
      if(track_ok) fillME(MEs.iLayer->ClusterChargePerCMfromTrack, dQdx_fromTrack);
      if(track_ok) fillME(MEs.iLayer->ClusterChargePerCMfromOriginOnTrack, dQdx_fromOrigin);
    }
    // ringMEs
    if (MEs.iRing != nullptr) {
      if(noise > 0.0) fillME(MEs.iRing->ClusterStoNCorrOnTrack, StoN*cosRZ);
      if(noise == 0.0) LogDebug("SiStripMonitorTrack") << "Module " << detid << " in Event " << eventNb << " noise " << cluster->noiseRescaledByGain() << std::endl;
      fillME(MEs.iRing->ClusterGain, clustergain);
      fillME(MEs.iRing->ClusterChargeCorrOnTrack, charge*cosRZ);
      fillME(MEs.iRing->ClusterChargeOnTrack, charge);
      fillME(MEs.iRing->ClusterChargeRawOnTrack, chargeraw);
      fillME(MEs.iRing->ClusterNoiseOnTrack, noise);
      fillME(MEs.iRing->ClusterWidthOnTrack, width);
      fillME(MEs.iRing->ClusterPosOnTrack, position);
      if(track_ok) fillME(MEs.iRing->ClusterChargePerCMfromTrack, dQdx_fromTrack);
      if(track_ok) fillME(MEs.iRing->ClusterChargePerCMfromOriginOnTrack, dQdx_fromOrigin);
    }
    // subdetMEs
    if(MEs.iSubdet != nullptr){
      fillME(MEs.iSubdet->ClusterGain, clustergain);
      fillME(MEs.iSubdet->ClusterChargeOnTrack,charge);
      fillME(MEs.iSubdet->ClusterChargeRawOnTrack,chargeraw);
      if(noise > 0.0) fillME(MEs.iSubdet->ClusterStoNCorrOnTrack,StoN*cosRZ);
      fillME(MEs.iSubdet->ClusterChargeCorrOnTrack, charge*cosRZ);
      if(track_ok) fillME(MEs.iSubdet->ClusterChargePerCMfromTrack,dQdx_fromTrack);
      if(track_ok) fillME(MEs.iSubdet->ClusterChargePerCMfromOriginOnTrack,dQdx_fromOrigin);
      if( tTopo->moduleGeometry(detid) == SiStripDetId::ModuleGeometry::W5 || tTopo->moduleGeometry(detid) == SiStripDetId::ModuleGeometry::W6 || tTopo->moduleGeometry(detid) == SiStripDetId::ModuleGeometry::W7) {
        if(noise > 0.0) fillME(MEs.iSubdet->ClusterStoNCorrThickOnTrack, StoN*cosRZ);
        fillME(MEs.iSubdet->ClusterChargeCorrThickOnTrack, charge*cosRZ);
      } else  {
        if(noise > 0.0) fillME(MEs.iSubdet->ClusterStoNCorrThinOnTrack, StoN*cosRZ);
        fillME(MEs.iSubdet->ClusterChargeCorrThinOnTrack, charge*cosRZ);
      }
    }
    //******** TkHistoMaps
    if (TkHistoMap_On_) {
      uint32_t adet=cluster->detId();
      tkhisto_NumOnTrack->add(adet,1.);
      if(noise > 0.0) tkhisto_StoNCorrOnTrack->fill(adet,cluster->signalOverNoise()*cosRZ);
      if(noise == 0.0)
	LogDebug("SiStripMonitorTrack") << "Module " << detid << " in Event " << eventNb << " noise " << noise << std::endl;
    }
    // Module plots filled only for onTrack Clusters
    if(Mod_On_){
      SiStripHistoId hidmanager2;
      std::string name = hidmanager2.createHistoId("","det",detid);
      //fillModMEs
      std::map<std::string, ModMEs>::iterator iModME  = ModMEsMap.find(name);
      if(iModME!=ModMEsMap.end()){
        if(noise > 0.0) fillME(iModME->second.ClusterStoNCorr ,StoN*cosRZ);
        if(noise == 0.0) LogDebug("SiStripMonitorTrack") << "Module " << name << " in Event " << eventNb << " noise " << noise << std::endl;
        fillME(iModME->second.ClusterGain, clustergain);
        fillME(iModME->second.ClusterCharge,charge);
        fillME(iModME->second.ClusterChargeRaw,chargeraw);

        fillME(iModME->second.ClusterChargeCorr,charge*cosRZ);

        fillME(iModME->second.ClusterWidth ,width);
        fillME(iModME->second.ClusterPos   ,position);

        if(track_ok) fillME(iModME->second.ClusterChargePerCMfromTrack,  dQdx_fromTrack);
        if(track_ok) fillME(iModME->second.ClusterChargePerCMfromOrigin, dQdx_fromOrigin);

        //fill the PGV histo
        float PGVmax = cluster->maxCharge();
        int PGVposCounter = cluster->maxIndex();
        for (int i= int(conf_.getParameter<edm::ParameterSet>("TProfileClusterPGV").getParameter<double>("xmin"));i<PGVposCounter;++i)
          fillME(iModME->second.ClusterPGV, i,0.);
        for (auto it=cluster->stripCharges().begin();it<cluster->stripCharges().end();++it) {
          fillME(iModME->second.ClusterPGV, PGVposCounter++,(*it)/PGVmax);
        }
        for (int i= PGVposCounter;i<int(conf_.getParameter<edm::ParameterSet>("TProfileClusterPGV").getParameter<double>("xmax"));++i)
          fillME(iModME->second.ClusterPGV, i,0.);
        //end fill the PGV histo
      }
    }
  } else {
    if (flag == OffTrack) {
      if (MEs.iSubdet != nullptr) MEs.iSubdet->totNClustersOffTrack++;
      //******** TkHistoMaps
      if (TkHistoMap_On_) {
        uint32_t adet=cluster->detId();
        tkhisto_NumOffTrack->add(adet,1.);
        if(charge > 250){
	  LogDebug("SiStripMonitorTrack") << "Module firing " << detid << " in Event " << eventNb << std::endl;
        }
      }
    }
    // layerMEs
    if (MEs.iLayer != nullptr) {
      fillME(MEs.iLayer->ClusterGain, clustergain);
      fillME(MEs.iLayer->ClusterChargeOffTrack, charge);
      fillME(MEs.iLayer->ClusterChargeRawOffTrack, chargeraw);
      fillME(MEs.iLayer->ClusterNoiseOffTrack, noise);
      fillME(MEs.iLayer->ClusterWidthOffTrack, width);
      fillME(MEs.iLayer->ClusterPosOffTrack, position);
      fillME(MEs.iLayer->ClusterChargePerCMfromOriginOffTrack, dQdx_fromOrigin);
    }
    // ringMEs
    if (MEs.iRing != nullptr) {
      fillME(MEs.iRing->ClusterGain, clustergain);
      fillME(MEs.iRing->ClusterChargeOffTrack, charge);
      fillME(MEs.iRing->ClusterChargeRawOffTrack, chargeraw);
      fillME(MEs.iRing->ClusterNoiseOffTrack, noise);
      fillME(MEs.iRing->ClusterWidthOffTrack, width);
      fillME(MEs.iRing->ClusterPosOffTrack, position);
      fillME(MEs.iRing->ClusterChargePerCMfromOriginOffTrack, dQdx_fromOrigin);
    }
    // subdetMEs
    if(MEs.iSubdet != nullptr){
      fillME(MEs.iSubdet->ClusterGain, clustergain);
      fillME(MEs.iSubdet->ClusterChargeOffTrack,charge);
      fillME(MEs.iSubdet->ClusterChargeRawOffTrack,chargeraw);
      if(noise > 0.0) fillME(MEs.iSubdet->ClusterStoNOffTrack,StoN);
      fillME(MEs.iSubdet->ClusterChargePerCMfromOriginOffTrack,dQdx_fromOrigin);
    }
  }
  return true;
}
//--------------------------------------------------------------------------------
