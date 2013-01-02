#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

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

#include "DQM/SiStripMonitorTrack/interface/SiStripMonitorTrack.h"

#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "TMath.h"

SiStripMonitorTrack::SiStripMonitorTrack(const edm::ParameterSet& conf): 
  dbe(edm::Service<DQMStore>().operator->()),
  conf_(conf),
  tracksCollection_in_EventTree(true),
  firstEvent(-1),
  genTriggerEventFlag_(new GenericTriggerEventFlag(conf))
{
  Cluster_src_   = conf.getParameter<edm::InputTag>("Cluster_src");
  Mod_On_        = conf.getParameter<bool>("Mod_On");
  Trend_On_      = conf.getParameter<bool>("Trend_On");
  flag_ring      = conf.getParameter<bool>("RingFlag_On");
  TkHistoMap_On_ = conf.getParameter<bool>("TkHistoMap_On");

  edm::ParameterSet ParametersClustersOn =  conf_.getParameter<edm::ParameterSet>("TH1nClustersOn");
  layerontrack = ParametersClustersOn.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClustersOff =  conf_.getParameter<edm::ParameterSet>("TH1nClustersOff");
  layerofftrack = ParametersClustersOff.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterCharge =  conf_.getParameter<edm::ParameterSet>("TH1ClusterCharge");
  layercharge = ParametersClusterCharge.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterStoN =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoN");
  layerston = ParametersClusterStoN.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterChargeCorr =  conf_.getParameter<edm::ParameterSet>("TH1ClusterChargeCorr");
  layerchargecorr = ParametersClusterChargeCorr.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterStoNCorrOn =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoNCorr");
  layerstoncorrontrack = ParametersClusterStoNCorrOn.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterNoise =  conf_.getParameter<edm::ParameterSet>("TH1ClusterNoise");
  layernoise = ParametersClusterNoise.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersClusterWidth =  conf_.getParameter<edm::ParameterSet>("TH1ClusterWidth");
  layerwidth = ParametersClusterWidth.getParameter<bool>("layerswitchon");

  TrackProducer_ = conf_.getParameter<std::string>("TrackProducer");
  TrackLabel_ = conf_.getParameter<std::string>("TrackLabel");

  // cluster quality conditions 
  edm::ParameterSet cluster_condition = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
  applyClusterQuality_ = cluster_condition.getParameter<bool>("On");
  sToNLowerLimit_      = cluster_condition.getParameter<double>("minStoN");
  sToNUpperLimit_      = cluster_condition.getParameter<double>("maxStoN");
  widthLowerLimit_     = cluster_condition.getParameter<double>("minWidth"); 
  widthUpperLimit_     = cluster_condition.getParameter<double>("maxWidth"); 


  // Create DCS Status
  bool checkDCS    = conf_.getParameter<bool>("UseDCSFiltering");
  if (checkDCS) dcsStatus_ = new SiStripDCSStatus();
  else dcsStatus_ = 0; 
}

//------------------------------------------------------------------------
SiStripMonitorTrack::~SiStripMonitorTrack() { 
  if (dcsStatus_) delete dcsStatus_;
  if (genTriggerEventFlag_) delete genTriggerEventFlag_;
}

//------------------------------------------------------------------------
void SiStripMonitorTrack::beginRun(const edm::Run& run,const edm::EventSetup& es)
{
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);

  //get geom 
  es.get<TrackerDigiGeometryRecord>().get( tkgeom );
  LogDebug("SiStripMonitorTrack") << "[SiStripMonitorTrack::beginRun] There are "<<tkgeom->detUnits().size() <<" detectors instantiated in the geometry" << std::endl;  
  es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );

  book(tTopo);

  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() )genTriggerEventFlag_->initRun( run, es );
}

//------------------------------------------------------------------------
void SiStripMonitorTrack::endJob(void)
{
  if(conf_.getParameter<bool>("OutputMEsInRootFile")){
    dbe->showDirStructure();
    dbe->save(conf_.getParameter<std::string>("OutputFileName"));
  }
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
  runNb   = e.id().run();
  eventNb = e.id().event();
  vPSiStripCluster.clear();
  
  iOrbitSec = e.orbitNumber()/11223.0;

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
 
  for (std::map<std::string, SubDetMEs>::iterator iSubDet = SubDetMEsMap.begin();
       iSubDet != SubDetMEsMap.end(); iSubDet++) {
    SubDetMEs subdet_mes = iSubDet->second;
    fillME(subdet_mes.nClustersOnTrack, subdet_mes.totNClustersOnTrack);
    fillME(subdet_mes.nClustersOffTrack, subdet_mes.totNClustersOffTrack);
    if (Trend_On_) {
      fillME(subdet_mes.nClustersTrendOnTrack,iOrbitSec,subdet_mes.totNClustersOnTrack);
      fillME(subdet_mes.nClustersTrendOffTrack,iOrbitSec,subdet_mes.totNClustersOffTrack);
    }
  }  
}

//------------------------------------------------------------------------  
void SiStripMonitorTrack::book(edm::ESHandle<TrackerTopology>& tTopo)
{
  
  SiStripFolderOrganizer folder_organizer;
  //******** TkHistoMaps
  if (TkHistoMap_On_) {
    tkhisto_StoNCorrOnTrack = new TkHistoMap("SiStrip/TkHisto" ,"TkHMap_StoNCorrOnTrack",0.0,1); 
    tkhisto_NumOnTrack  = new TkHistoMap("SiStrip/TkHisto", "TkHMap_NumberOfOnTrackCluster",0.0,1);
    tkhisto_NumOffTrack = new TkHistoMap("SiStrip/TkHisto", "TkHMap_NumberOfOfffTrackCluster",0.0,1);
  }
  //******** TkHistoMaps

  std::vector<uint32_t> vdetId_;
  SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
  //Histos for each detector, layer and module
  for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){  //loop on all the active detid
    uint32_t detid = *detid_iter;
    
    if (detid < 1){
      edm::LogError("SiStripMonitorTrack")<< "[" <<__PRETTY_FUNCTION__ << "] invalid detid " << detid<< std::endl;
      continue;
    }

    
    std::string name;    
    
    // book Layer and RING plots
    std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid,tTopo,flag_ring);
    
    SiStripHistoId hidmanager;
    std::string layer_id = hidmanager.getSubdetid(detid, tTopo, flag_ring);
    std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(layer_id);
    if(iLayerME==LayerMEsMap.end()){
      folder_organizer.setLayerFolder(detid, tTopo, det_layer_pair.second, flag_ring);
      bookLayerMEs(detid, layer_id);
    }
    // book sub-detector plots
    std::pair<std::string,std::string> sdet_pair = folder_organizer.getSubDetFolderAndTag(detid, tTopo);
    if (SubDetMEsMap.find(sdet_pair.second) == SubDetMEsMap.end()){
      dbe->setCurrentFolder(sdet_pair.first);
      bookSubDetMEs(sdet_pair.second);        
    }
    // book module plots
    if(Mod_On_) {
      folder_organizer.setDetectorFolder(detid,tTopo);
      bookModMEs(*detid_iter);
    } 
  }//end loop on detectors detid
}
  
//--------------------------------------------------------------------------------
void SiStripMonitorTrack::bookModMEs(const uint32_t & id)//Histograms at MODULE level
{
  std::string name = "det";
  SiStripHistoId hidmanager;
  std::string hid = hidmanager.createHistoId("",name,id);
  std::map<std::string, ModMEs>::iterator iModME  = ModMEsMap.find(hid);
  if(iModME==ModMEsMap.end()){
    ModMEs theModMEs; 
    theModMEs.ClusterStoNCorr   = 0;
    theModMEs.ClusterCharge     = 0;
    theModMEs.ClusterChargeCorr = 0;
    theModMEs.ClusterWidth      = 0;
    theModMEs.ClusterPos        = 0;
    theModMEs.ClusterPGV        = 0;

    // Cluster Width
    theModMEs.ClusterWidth=bookME1D("TH1ClusterWidth", hidmanager.createHistoId("ClusterWidth_OnTrack",name,id).c_str()); 
    dbe->tag(theModMEs.ClusterWidth,id); 
    // Cluster Charge
    theModMEs.ClusterCharge=bookME1D("TH1ClusterCharge", hidmanager.createHistoId("ClusterCharge_OnTrack",name,id).c_str());
    dbe->tag(theModMEs.ClusterCharge,id); 
    // Cluster Charge Corrected
    theModMEs.ClusterChargeCorr=bookME1D("TH1ClusterChargeCorr", hidmanager.createHistoId("ClusterChargeCorr_OnTrack",name,id).c_str());
    dbe->tag(theModMEs.ClusterChargeCorr,id); 
    // Cluster StoN Corrected
    theModMEs.ClusterStoNCorr=bookME1D("TH1ClusterStoNCorrMod", hidmanager.createHistoId("ClusterStoNCorr_OnTrack",name,id).c_str());
    dbe->tag(theModMEs.ClusterStoNCorr,id); 
    // Cluster Position
    short total_nr_strips = SiStripDetCabling_->nApvPairs(id) * 2 * 128;
    theModMEs.ClusterPos=dbe->book1D(hidmanager.createHistoId("ClusterPosition_OnTrack",name,id).c_str(),hidmanager.createHistoId("ClusterPosition_OnTrack",name,id).c_str(),total_nr_strips,0.5,total_nr_strips+0.5);
    dbe->tag(theModMEs.ClusterPos,id); 
    // Cluster PGV
    theModMEs.ClusterPGV=bookMEProfile("TProfileClusterPGV", hidmanager.createHistoId("PGV_OnTrack",name,id).c_str()); 
    dbe->tag(theModMEs.ClusterPGV,id); 

    ModMEsMap[hid]=theModMEs;
  }
}
//
// -- Book Layer Level Histograms and Trend plots
//
void SiStripMonitorTrack::bookLayerMEs(const uint32_t& mod_id, std::string& layer_id)
{
  std::string name = "layer";  
  std::string hname;
  SiStripHistoId hidmanager;

  LayerMEs theLayerMEs; 
  theLayerMEs.ClusterStoNCorrOnTrack   = 0;
  theLayerMEs.ClusterChargeCorrOnTrack = 0;
  theLayerMEs.ClusterChargeOnTrack     = 0;
  theLayerMEs.ClusterChargeOffTrack    = 0;
  theLayerMEs.ClusterNoiseOnTrack      = 0;
  theLayerMEs.ClusterNoiseOffTrack     = 0;
  theLayerMEs.ClusterWidthOnTrack      = 0;
  theLayerMEs.ClusterWidthOffTrack     = 0;
  theLayerMEs.ClusterPosOnTrack        = 0;
  theLayerMEs.ClusterPosOffTrack       = 0;
  
  // Cluster StoN Corrected
  if (layerstoncorrontrack){
    hname = hidmanager.createHistoLayer("Summary_ClusterStoNCorr",name,layer_id,"OnTrack");
    theLayerMEs.ClusterStoNCorrOnTrack = bookME1D("TH1ClusterStoNCorr", hname.c_str());
  }

  // Cluster Charge Corrected
  if (layerchargecorr){
    hname = hidmanager.createHistoLayer("Summary_ClusterChargeCorr",name,layer_id,"OnTrack");
    theLayerMEs.ClusterChargeCorrOnTrack = bookME1D("TH1ClusterChargeCorr", hname.c_str());
  }

  // Cluster Charge (On and Off Track)
  if (layercharge){
    hname = hidmanager.createHistoLayer("Summary_ClusterCharge",name,layer_id,"OnTrack");
    theLayerMEs.ClusterChargeOnTrack = bookME1D("TH1ClusterCharge", hname.c_str());
  
    hname = hidmanager.createHistoLayer("Summary_ClusterCharge",name,layer_id,"OffTrack");
    theLayerMEs.ClusterChargeOffTrack = bookME1D("TH1ClusterCharge", hname.c_str());
  }

  // Cluster Noise (On and Off Track)
  hname = hidmanager.createHistoLayer("Summary_ClusterNoise",name,layer_id,"OnTrack");
  theLayerMEs.ClusterNoiseOnTrack = bookME1D("TH1ClusterNoise", hname.c_str()); 

  hname = hidmanager.createHistoLayer("Summary_ClusterNoise",name,layer_id,"OffTrack");
  theLayerMEs.ClusterNoiseOffTrack = bookME1D("TH1ClusterNoise", hname.c_str()); 

  if (layernoise){
    hname = hidmanager.createHistoLayer("Summary_ClusterNoise",name,layer_id,"OnTrack");
    theLayerMEs.ClusterNoiseOnTrack = bookME1D("TH1ClusterNoise", hname.c_str()); 
    
    hname = hidmanager.createHistoLayer("Summary_ClusterNoise",name,layer_id,"OffTrack");
    theLayerMEs.ClusterNoiseOffTrack = bookME1D("TH1ClusterNoise", hname.c_str()); 
  }
  // Cluster Width (On and Off Track)
  if (layerwidth){
    hname = hidmanager.createHistoLayer("Summary_ClusterWidth",name,layer_id,"OnTrack");
    theLayerMEs.ClusterWidthOnTrack = bookME1D("TH1ClusterWidth", hname.c_str()); 
    
    hname = hidmanager.createHistoLayer("Summary_ClusterWidth",name,layer_id,"OffTrack");
    theLayerMEs.ClusterWidthOffTrack = bookME1D("TH1ClusterWidth", hname.c_str()); 
  }

  //Cluster Position
  short total_nr_strips = SiStripDetCabling_->nApvPairs(mod_id) * 2 * 128; 
  if (layer_id.find("TEC") != std::string::npos && !flag_ring)  total_nr_strips = 3 * 2 * 128;
  
  hname = hidmanager.createHistoLayer("Summary_ClusterPosition",name,layer_id,"OnTrack");
  theLayerMEs.ClusterPosOnTrack = dbe->book1D(hname, hname, total_nr_strips, 0.5,total_nr_strips+0.5);
  
  hname = hidmanager.createHistoLayer("Summary_ClusterPosition",name,layer_id,"OffTrack");
  theLayerMEs.ClusterPosOffTrack = dbe->book1D(hname, hname, total_nr_strips, 0.5,total_nr_strips+0.5);
  
  //bookeeping
  LayerMEsMap[layer_id]=theLayerMEs;
}
//
// -- Book Histograms at Sub-Detector Level
//
void SiStripMonitorTrack::bookSubDetMEs(std::string& name){

  std::string subdet_tag;
  subdet_tag = "__" + name;
  std::string completeName;

  SubDetMEs theSubDetMEs;
  theSubDetMEs.totNClustersOnTrack    = 0;
  theSubDetMEs.totNClustersOffTrack   = 0;
  theSubDetMEs.nClustersOnTrack       = 0;
  theSubDetMEs.nClustersTrendOnTrack  = 0;
  theSubDetMEs.nClustersOffTrack      = 0;
  theSubDetMEs.nClustersTrendOffTrack = 0;
  theSubDetMEs.ClusterStoNCorrOnTrack = 0;
  theSubDetMEs.ClusterChargeOffTrack  = 0;
  theSubDetMEs.ClusterStoNOffTrack    = 0;

  // TotalNumber of Cluster OnTrack
  completeName = "Summary_TotalNumberOfClusters_OnTrack" + subdet_tag;
  theSubDetMEs.nClustersOnTrack = bookME1D("TH1nClustersOn", completeName.c_str());
  theSubDetMEs.nClustersOnTrack->getTH1()->StatOverflows(kTRUE);
  
  // TotalNumber of Cluster OffTrack
  completeName = "Summary_TotalNumberOfClusters_OffTrack" + subdet_tag;
  theSubDetMEs.nClustersOffTrack = bookME1D("TH1nClustersOff", completeName.c_str());
  theSubDetMEs.nClustersOffTrack->getTH1()->StatOverflows(kTRUE);
  
  // Cluster StoN On Track
  completeName = "Summary_ClusterStoNCorr_OnTrack"  + subdet_tag;
  theSubDetMEs.ClusterStoNCorrOnTrack = bookME1D("TH1ClusterStoNCorr", completeName.c_str());
  
  // Cluster Charge Off Track
  completeName = "Summary_ClusterCharge_OffTrack" + subdet_tag;
  theSubDetMEs.ClusterChargeOffTrack=bookME1D("TH1ClusterCharge", completeName.c_str());
  
  // Cluster Charge StoN Off Track
  completeName = "Summary_ClusterStoN_OffTrack"  + subdet_tag;
  theSubDetMEs.ClusterStoNOffTrack = bookME1D("TH1ClusterStoN", completeName.c_str());
  
  if(Trend_On_){
    // TotalNumber of Cluster 
    completeName = "Trend_TotalNumberOfClusters_OnTrack"  + subdet_tag;
    theSubDetMEs.nClustersTrendOnTrack = bookMETrend("TH1nClustersOn", completeName.c_str());
    completeName = "Trend_TotalNumberOfClusters_OffTrack"  + subdet_tag;
    theSubDetMEs.nClustersTrendOffTrack = bookMETrend("TH1nClustersOff", completeName.c_str());
  }
  //bookeeping
  SubDetMEsMap[name]=theSubDetMEs;
}
//--------------------------------------------------------------------------------

MonitorElement* SiStripMonitorTrack::bookME1D(const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return dbe->book1D(HistoName,HistoName,
		         Parameters.getParameter<int32_t>("Nbinx"),
		         Parameters.getParameter<double>("xmin"),
		         Parameters.getParameter<double>("xmax")
		    );
}

//--------------------------------------------------------------------------------
MonitorElement* SiStripMonitorTrack::bookME2D(const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return dbe->book2D(HistoName,HistoName,
		     Parameters.getParameter<int32_t>("Nbinx"),
		     Parameters.getParameter<double>("xmin"),
		     Parameters.getParameter<double>("xmax"),
		     Parameters.getParameter<int32_t>("Nbiny"),
		     Parameters.getParameter<double>("ymin"),
		     Parameters.getParameter<double>("ymax")
		     );
}

//--------------------------------------------------------------------------------
MonitorElement* SiStripMonitorTrack::bookME3D(const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return dbe->book3D(HistoName,HistoName,
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
MonitorElement* SiStripMonitorTrack::bookMEProfile(const char* ParameterSetLabel, const char* HistoName)
{
    Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
    return dbe->bookProfile(HistoName,HistoName,
                            Parameters.getParameter<int32_t>("Nbinx"),
                            Parameters.getParameter<double>("xmin"),
                            Parameters.getParameter<double>("xmax"),
                            Parameters.getParameter<int32_t>("Nbiny"),
                            Parameters.getParameter<double>("ymin"),
                            Parameters.getParameter<double>("ymax"),
                            "" );
}

//--------------------------------------------------------------------------------
MonitorElement* SiStripMonitorTrack::bookMETrend(const char* ParameterSetLabel, const char* HistoName)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  edm::ParameterSet ParametersTrend =  conf_.getParameter<edm::ParameterSet>("Trending");
  MonitorElement* me = dbe->bookProfile(HistoName,HistoName,
					ParametersTrend.getParameter<int32_t>("Nbins"),
					0,
					ParametersTrend.getParameter<int32_t>("Nbins"),
					100, //that parameter should not be there !?
					Parameters.getParameter<double>("xmin"),
					Parameters.getParameter<double>("xmax"),
					"" );
  if (me->kind() == MonitorElement::DQM_KIND_TPROFILE) me->getTH1()->SetBit(TH1::kCanRebin);

  if(!me) return me;
  me->setAxisTitle("Event Time in Seconds",1);
  return me;
}

//------------------------------------------------------------------------------------------
 void SiStripMonitorTrack::trackStudy(const edm::Event& ev, const edm::EventSetup& es){

  // track input  
 
  edm::Handle<reco::TrackCollection > trackCollectionHandle;
  ev.getByLabel(TrackProducer_, TrackLabel_, trackCollectionHandle);//takes the track collection
  if (!trackCollectionHandle.isValid()){
    edm::LogError("SiStripMonitorTrack")<<" Track Collection is not valid !! " << TrackLabel_<<std::endl;
    return;
  }
  
  // trajectory input
  edm::Handle<TrajTrackAssociationCollection> TItkAssociatorCollection;
  ev.getByLabel(TrackProducer_, TrackLabel_, TItkAssociatorCollection);
  if( !TItkAssociatorCollection.isValid()){
    edm::LogError("SiStripMonitorTrack")<<"Association not found "<<std::endl;
    return;
  }
  
  //Perform track study
  int i=0;
  for(TrajTrackAssociationCollection::const_iterator it =  TItkAssociatorCollection->begin();it !=  TItkAssociatorCollection->end(); ++it){
    const edm::Ref<std::vector<Trajectory> > traj_iterator = it->key;  
    // Trajectory Map, extract Trajectory for this track
    reco::TrackRef trackref = it->val;
    LogDebug("SiStripMonitorTrack")
      << "Track number "<< i+1 
      << "\n\tmomentum: " << trackref->momentum()
      << "\n\tPT: " << trackref->pt()
      << "\n\tvertex: " << trackref->vertex()
      << "\n\timpact parameter: " << trackref->d0()
      << "\n\tcharge: " << trackref->charge()
      << "\n\tnormalizedChi2: " << trackref->normalizedChi2() 
      <<"\n\tFrom EXTRA : "
      <<"\n\t\touter PT "<< trackref->outerPt()<<std::endl;
    i++;

    const std::vector<TrajectoryMeasurement> & measurements = traj_iterator->measurements();
    std::vector<TrajectoryMeasurement>::const_iterator traj_mes_iterator;
    int nhit=0;
    for(std::vector<TrajectoryMeasurement>::const_iterator traj_mes_iterator= measurements.begin();traj_mes_iterator!=measurements.end();traj_mes_iterator++){//loop on measurements
      //trajectory local direction and position on detector
      LocalPoint  stateposition;
      LocalVector statedirection;
      
      TrajectoryStateOnSurface  updatedtsos=traj_mes_iterator->updatedState();
      ConstRecHitPointer ttrh=traj_mes_iterator->recHit();
      if (!ttrh->isValid()) {continue;}
      
      nhit++;
      
      const ProjectedSiStripRecHit2D* phit     = dynamic_cast<const ProjectedSiStripRecHit2D*>( ttrh->hit() );
      const SiStripMatchedRecHit2D* matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>( ttrh->hit() );
      const SiStripRecHit2D* hit2D             = dynamic_cast<const SiStripRecHit2D*>( ttrh->hit() );	
      const SiStripRecHit1D* hit1D             = dynamic_cast<const SiStripRecHit1D*>( ttrh->hit() );	
      
      //      RecHitType type=Single;

      if(matchedhit){
	LogTrace("SiStripMonitorTrack")<<"\nMatched recHit found"<< std::endl;
	//	type=Matched;
	
	GluedGeomDet * gdet=(GluedGeomDet *)tkgeom->idToDet(matchedhit->geographicalId());
	GlobalVector gtrkdirup=gdet->toGlobal(updatedtsos.localMomentum());	    
	//mono side
	const GeomDetUnit * monodet=gdet->monoDet();
	statedirection=monodet->toLocal(gtrkdirup);
        SiStripRecHit2D m = matchedhit->monoHit();
	if(statedirection.mag() != 0)	  RecHitInfo<SiStripRecHit2D>(&m,statedirection,trackref,es);
	//stereo side
	const GeomDetUnit * stereodet=gdet->stereoDet();
	statedirection=stereodet->toLocal(gtrkdirup);
        SiStripRecHit2D s = matchedhit->stereoHit();
	if(statedirection.mag() != 0)	  RecHitInfo<SiStripRecHit2D>(&s,statedirection,trackref,es);
      }
      else if(phit){
	LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found"<< std::endl;
	//	type=Projected;
	GluedGeomDet * gdet=(GluedGeomDet *)tkgeom->idToDet(phit->geographicalId());
	
	GlobalVector gtrkdirup=gdet->toGlobal(updatedtsos.localMomentum());
	const SiStripRecHit2D&  originalhit=phit->originalHit();
	const GeomDetUnit * det;
	if(!StripSubdetector(originalhit.geographicalId().rawId()).stereo()){
	  //mono side
	  LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found  MONO"<< std::endl;
	  det=gdet->monoDet();
	  statedirection=det->toLocal(gtrkdirup);
	  if(statedirection.mag() != 0) RecHitInfo<SiStripRecHit2D>(&(phit->originalHit()),statedirection,trackref,es);
	}
	else{
	  LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found STEREO"<< std::endl;
	  //stereo side
	  det=gdet->stereoDet();
	  statedirection=det->toLocal(gtrkdirup);
	  if(statedirection.mag() != 0) RecHitInfo<SiStripRecHit2D>(&(phit->originalHit()),statedirection,trackref,es);
	}
      }else if (hit2D){
	statedirection=updatedtsos.localMomentum();
	if(statedirection.mag() != 0) RecHitInfo<SiStripRecHit2D>(hit2D,statedirection,trackref,es);
      } else if (hit1D) {
	statedirection=updatedtsos.localMomentum();
	if(statedirection.mag() != 0) RecHitInfo<SiStripRecHit1D>(hit1D,statedirection,trackref,es);
      } else {
	LogDebug ("SiStrioMonitorTrack") 
                 << " LocalMomentum: "<<statedirection
		 << "\nLocal x-z plane angle: "<<atan2(statedirection.x(),statedirection.z());
      }
    }
  }
}

template <class T> void SiStripMonitorTrack::RecHitInfo(const T* tkrecHit, LocalVector LV,reco::TrackRef track_ref, const edm::EventSetup& es){
    
    if(!tkrecHit->isValid()){
      LogTrace("SiStripMonitorTrack") <<"\t\t Invalid Hit " << std::endl;
      return;  
    }
    
    const uint32_t& detid = tkrecHit->geographicalId().rawId();
    if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()){
      LogTrace("SiStripMonitorTrack") << "Modules Excluded" << std::endl;
      return;
    }
    
    LogTrace("SiStripMonitorTrack")
      <<"\n\t\tRecHit on det "<<tkrecHit->geographicalId().rawId()
      <<"\n\t\tRecHit in LP "<<tkrecHit->localPosition()
      <<"\n\t\tRecHit in GP "<<tkgeom->idToDet(tkrecHit->geographicalId())->surface().toGlobal(tkrecHit->localPosition()) 
      <<"\n\t\tRecHit trackLocal vector "<<LV.x() << " " << LV.y() << " " << LV.z() <<std::endl; 


    edm::ESHandle<TrackerTopology> tTopo;
    es.get<IdealGeometryRecord>().get(tTopo);
    
    //Get SiStripCluster from SiStripRecHit
    if ( tkrecHit != NULL ){
      const SiStripCluster* SiStripCluster_ = &*(tkrecHit->cluster());
      SiStripClusterInfo SiStripClusterInfo_(*SiStripCluster_,es,detid);
            
      if ( clusterInfos(&SiStripClusterInfo_,detid, tTopo, OnTrack, LV ) ) {
	vPSiStripCluster.push_back(SiStripCluster_);
      }
    }else{
     edm::LogError("SiStripMonitorTrack") << "NULL hit" << std::endl;
    }	  
  }

//------------------------------------------------------------------------

void SiStripMonitorTrack::AllClusters(const edm::Event& ev, const edm::EventSetup& es) 
{
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);
    
  edm::Handle< edmNew::DetSetVector<SiStripCluster> > siStripClusterHandle;
  ev.getByLabel( Cluster_src_, siStripClusterHandle); 
  if (!siStripClusterHandle.isValid()){
    edm::LogError("SiStripMonitorTrack")<< "ClusterCollection is not valid!!" << std::endl;
    return;
  }
  //Loop on Dets
  for ( edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=siStripClusterHandle->begin(); DSViter!=siStripClusterHandle->end();DSViter++){
    uint32_t detid=DSViter->id();
    if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()) continue;
    //Loop on Clusters
    LogDebug("SiStripMonitorTrack") << "on detid "<< detid << " N Cluster= " << DSViter->size();
    edmNew::DetSet<SiStripCluster>::const_iterator ClusIter = DSViter->begin();
    for(; ClusIter!=DSViter->end(); ClusIter++) {
      if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	SiStripClusterInfo SiStripClusterInfo_(*ClusIter,es,detid);
	clusterInfos(&SiStripClusterInfo_,detid,tTopo,OffTrack,LV);
      }
    }
  }
}

//------------------------------------------------------------------------
bool SiStripMonitorTrack::clusterInfos(SiStripClusterInfo* cluster, const uint32_t& detid, edm::ESHandle<TrackerTopology>& tTopo, enum ClusterFlags flag, const LocalVector LV)
{
  if (cluster==0) return false;
  // if one imposes a cut on the clusters, apply it
  if( (applyClusterQuality_) &&
      (cluster->signalOverNoise() < sToNLowerLimit_ ||
       cluster->signalOverNoise() > sToNUpperLimit_ ||
       cluster->width() < widthLowerLimit_ ||
       cluster->width() > widthUpperLimit_) ) return false;
  // start of the analysis
  
  std::pair<std::string,std::string> sdet_pair = folderOrganizer_.getSubDetFolderAndTag(detid,tTopo);
  std::map<std::string, SubDetMEs>::iterator iSubdet  = SubDetMEsMap.find(sdet_pair.second);
  if(iSubdet != SubDetMEsMap.end()){ 
    if (flag == OnTrack) iSubdet->second.totNClustersOnTrack++;
    else if (flag == OffTrack) iSubdet->second.totNClustersOffTrack++;
  }
  
  float cosRZ = -2;
  LogDebug("SiStripMonitorTrack")<< "\n\tLV " << LV.x() << " " << LV.y() << " " << LV.z() << " " << LV.mag() << std::endl;
  if (LV.mag()!=0){
    cosRZ= fabs(LV.z())/LV.mag();
    LogDebug("SiStripMonitorTrack")<< "\n\t cosRZ " << cosRZ << std::endl;
  }
  std::string name;
  
  // Filling SubDet/Layer Plots (on Track + off Track)
  fillMEs(cluster,detid,tTopo,cosRZ,flag);
  
  
  //******** TkHistoMaps
  if (TkHistoMap_On_) {
    uint32_t adet=cluster->detId();
    float noise = cluster->noiseRescaledByGain();
    if(flag==OnTrack){
      tkhisto_NumOnTrack->add(adet,1.);
      if(noise > 0.0) tkhisto_StoNCorrOnTrack->fill(adet,cluster->signalOverNoise()*cosRZ);
      if(noise == 0.0) 
	LogDebug("SiStripMonitorTrack") << "Module " << detid << " in Event " << eventNb << " noise " << noise << std::endl;
    }
    else if(flag==OffTrack){
      tkhisto_NumOffTrack->add(adet,1.);
      if(cluster->charge() > 250){
	LogDebug("SiStripMonitorTrack") << "Module firing " << detid << " in Event " << eventNb << std::endl;
      }
    }
  }

  // Module plots filled only for onTrack Clusters
  if(Mod_On_){
    if(flag==OnTrack){
      SiStripHistoId hidmanager2;
      name =hidmanager2.createHistoId("","det",detid);
      fillModMEs(cluster,name,cosRZ); 
    }
  }
  return true;
}

//--------------------------------------------------------------------------------
void SiStripMonitorTrack::fillModMEs(SiStripClusterInfo* cluster,std::string name,float cos)
{
  std::map<std::string, ModMEs>::iterator iModME  = ModMEsMap.find(name);
  if(iModME!=ModMEsMap.end()){

    float    StoN     = cluster->signalOverNoise();
    uint16_t charge   = cluster->charge();
    uint16_t width    = cluster->width();
    float    position = cluster->baryStrip(); 

    float noise = cluster->noiseRescaledByGain();
    if(noise > 0.0) fillME(iModME->second.ClusterStoNCorr ,StoN*cos);
    if(noise == 0.0) LogDebug("SiStripMonitorTrack") << "Module " << name << " in Event " << eventNb << " noise " << noise << std::endl;
    fillME(iModME->second.ClusterCharge,charge);

    fillME(iModME->second.ClusterChargeCorr,charge*cos);

    fillME(iModME->second.ClusterWidth ,width);
    fillME(iModME->second.ClusterPos   ,position);
    
    //fill the PGV histo
    float PGVmax = cluster->maxCharge();
    int PGVposCounter = cluster->maxIndex();
    for (int i= int(conf_.getParameter<edm::ParameterSet>("TProfileClusterPGV").getParameter<double>("xmin"));i<PGVposCounter;++i)
      fillME(iModME->second.ClusterPGV, i,0.);
    for (std::vector<uint8_t>::const_iterator it=cluster->stripCharges().begin();it<cluster->stripCharges().end();++it) {
      fillME(iModME->second.ClusterPGV, PGVposCounter++,(*it)/PGVmax);
    }
    for (int i= PGVposCounter;i<int(conf_.getParameter<edm::ParameterSet>("TProfileClusterPGV").getParameter<double>("xmax"));++i)
      fillME(iModME->second.ClusterPGV, i,0.);
    //end fill the PGV histo
  }
}

//------------------------------------------------------------------------
void SiStripMonitorTrack::fillMEs(SiStripClusterInfo* cluster,uint32_t detid, edm::ESHandle<TrackerTopology>& tTopo, float cos, enum ClusterFlags flag)
{ 
  std::pair<std::string,int32_t> SubDetAndLayer = folderOrganizer_.GetSubDetAndLayer(detid,tTopo,flag_ring);
  SiStripHistoId hidmanager1;
  std::string layer_id = hidmanager1.getSubdetid(detid,tTopo,flag_ring); 
  
  std::pair<std::string,std::string> sdet_pair = folderOrganizer_.getSubDetFolderAndTag(detid,tTopo);
  float    StoN     = cluster->signalOverNoise();
  float    noise    = cluster->noiseRescaledByGain();
  uint16_t charge   = cluster->charge();
  uint16_t width    = cluster->width();
  float    position = cluster->baryStrip(); 
   
  std::map<std::string, LayerMEs>::iterator iLayer  = LayerMEsMap.find(layer_id);
  if (iLayer != LayerMEsMap.end()) {
    if(flag==OnTrack){
      if(noise > 0.0 && layerstoncorrontrack) fillME(iLayer->second.ClusterStoNCorrOnTrack, StoN*cos);
      if(noise == 0.0) LogDebug("SiStripMonitorTrack") << "Module " << detid << " in Event " << eventNb << " noise " << cluster->noiseRescaledByGain() << std::endl;
      if(layerchargecorr) fillME(iLayer->second.ClusterChargeCorrOnTrack, charge*cos);
      if (layercharge) fillME(iLayer->second.ClusterChargeOnTrack, charge);
      if (layernoise) fillME(iLayer->second.ClusterNoiseOnTrack, noise);
      if (layerwidth) fillME(iLayer->second.ClusterWidthOnTrack, width);
      fillME(iLayer->second.ClusterPosOnTrack, position);
    } else {
      if (layercharge) fillME(iLayer->second.ClusterChargeOffTrack, charge);
      if (layernoise) fillME(iLayer->second.ClusterNoiseOffTrack, noise);
      if (layerwidth) fillME(iLayer->second.ClusterWidthOffTrack, width);
      fillME(iLayer->second.ClusterPosOffTrack, position);
    }
  }
  std::map<std::string, SubDetMEs>::iterator iSubdet  = SubDetMEsMap.find(sdet_pair.second);
  if(iSubdet != SubDetMEsMap.end() ){
    if(flag==OnTrack){
      if(noise > 0.0) fillME(iSubdet->second.ClusterStoNCorrOnTrack,StoN*cos);
    } else {
      fillME(iSubdet->second.ClusterChargeOffTrack,charge);
      if(noise > 0.0) fillME(iSubdet->second.ClusterStoNOffTrack,StoN);
    }
  }
}
//
// -- Get Subdetector Tag from the Folder name
//
void SiStripMonitorTrack::getSubDetTag(std::string& folder_name, std::string& tag){

  tag =  folder_name.substr(folder_name.find("MechanicalView")+15);
  if (tag.find("side_") != std::string::npos) {
    tag.replace(tag.find_last_of("/"),1,"_");
  }
}   
