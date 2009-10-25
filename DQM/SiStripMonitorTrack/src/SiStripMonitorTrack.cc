#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "DQM/SiStripMonitorTrack/interface/SiStripMonitorTrack.h"

#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "TMath.h"

static const uint16_t _NUM_SISTRIP_SUBDET_ = 4;
static TString SubDet[_NUM_SISTRIP_SUBDET_]={"TIB","TID","TOB","TEC"};
static std::string flags[2] = {"OnTrack","OffTrack"};

SiStripMonitorTrack::SiStripMonitorTrack(const edm::ParameterSet& conf): 
  dbe(edm::Service<DQMStore>().operator->()),
  conf_(conf),
  folder_organizer(), 
  tracksCollection_in_EventTree(true),
  firstEvent(-1)
{
  Cluster_src_   = conf.getParameter<edm::InputTag>("Cluster_src");
  Mod_On_        = conf.getParameter<bool>("Mod_On");
  Trend_On_      = conf.getParameter<bool>("Trend_On");
  OffHisto_On_   = conf.getParameter<bool>("OffHisto_On");
  HistoFlag_On_  = conf.getParameter<bool>("HistoFlag_On");
  flag_ring      = conf.getParameter<bool>("RingFlag_On");
  TkHistoMap_On_ = conf.getParameter<bool>("TkHistoMap_On");
  //  RawDigis_On_   = conf.getParameter<bool>("RawDigis_On");
  //  CCAnalysis_On_ = conf.getParameter<bool>("CCAnalysis_On");

  for(int i=0;i<4;++i) for(int j=0;j<2;++j) NClus[i][j]=0;
  if(OffHisto_On_){
    off_Flag = 2;
  }else{
    off_Flag = 1;
  }
}

//------------------------------------------------------------------------
SiStripMonitorTrack::~SiStripMonitorTrack() { }

//------------------------------------------------------------------------
//void SiStripMonitorTrack::beginJob(const edm::EventSetup& es)

void SiStripMonitorTrack::beginRun(const edm::Run& run,const edm::EventSetup& es)
{
  //get geom 
  es.get<TrackerDigiGeometryRecord>().get( tkgeom );
  edm::LogInfo("SiStripMonitorTrack") << "[SiStripMonitorTrack::beginRun] There are "<<tkgeom->detUnits().size() <<" detectors instantiated in the geometry" << std::endl;  
  es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );

  book();
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
  
  tracksCollection_in_EventTree=true;
  trackAssociatorCollection_in_EventTree=true;
  
  //initialization of global quantities
  edm::LogInfo("SiStripMonitorTrack") << "[SiStripMonitorTrack::analyse]  " << "Run " << e.id().run() << " Event " << e.id().event() << std::endl;
  runNb   = e.id().run();
  eventNb = e.id().event();
  vPSiStripCluster.clear();
  countOn=0;
  countOff=0;
  
  iOrbitSec = e.orbitNumber()/11223.0;
  e.getByLabel( Cluster_src_, dsv_SiStripCluster); 
  
  // track input  
  std::string TrackProducer = conf_.getParameter<std::string>("TrackProducer");
  std::string TrackLabel = conf_.getParameter<std::string>("TrackLabel");
  
  e.getByLabel(TrackProducer, TrackLabel, trackCollection);//takes the track collection
 
  if (trackCollection.isValid()){
  }else{
    edm::LogError("SiStripMonitorTrack")<<" Track Collection is not valid !! " << TrackLabel<<std::endl;
    tracksCollection_in_EventTree=false;
  }
  
  // trajectory input
  e.getByLabel(TrackProducer, TrackLabel, TrajectoryCollection);
  e.getByLabel(TrackProducer, TrackLabel, TItkAssociatorCollection);
  if( TItkAssociatorCollection.isValid()){
  }else{
    edm::LogError("SiStripMonitorTrack")<<"Association not found "<<std::endl;
    trackAssociatorCollection_in_EventTree=false;
  }
  
  //Perform track study
  if (tracksCollection_in_EventTree && trackAssociatorCollection_in_EventTree) trackStudy(es);
  
  //Perform Cluster Study (irrespectively to tracks)

  if(OffHisto_On_){
    if (dsv_SiStripCluster.isValid()){
      AllClusters(es);//analyzes the off Track Clusters
    }else{
      edm::LogError("SiStripMonitorTrack")<< "ClusterCollection is not valid!!" << std::endl;
    }
  }

  //Summary Counts of clusters
  std::map<TString, MonitorElement*>::iterator iME;
  std::map<TString, ModMEs>::iterator          iModME ;
  std::map<TString, LayerMEs>::iterator        iLayerME;
  
  for (int j=0;j<off_Flag;++j){ // loop over ontrack, offtrack
    int nTot=0;
    for (int i=0;i<4;++i){ // loop over TIB, TID, TOB, TEC
      name=flags[j]+"_in_"+SubDet[i];      
      iLayerME = LayerMEsMap.find(name);
      if(iLayerME!=LayerMEsMap.end()) {
	if(flags[j]=="OnTrack" && NClus[i][j]){
	  fillME(   iLayerME->second.nClusters,      NClus[i][j]);
	}else if(flags[j]=="OffTrack"){
	  fillME(   iLayerME->second.nClusters,      NClus[i][j]);
	}
	if(Trend_On_){
	  fillME(iLayerME->second.nClustersTrend,iOrbitSec,NClus[i][j]);
	}
      }
      nTot+=NClus[i][j];
      NClus[i][j]=0;
    } // loop over TIB, TID, TOB, TEC
    
    name=flags[j]+"_TotalNumberOfClusters";
    iME = MEMap.find(name);
    if(iME!=MEMap.end() && nTot) iME->second->Fill(nTot);
    if(Trend_On_){
      iME = MEMap.find(name+"Trend");
      if(iME!=MEMap.end()){
	fillME(iME->second,iOrbitSec,nTot);
      }
    }
  } // loop over ontrack, offtrack
  
}

//------------------------------------------------------------------------  
void SiStripMonitorTrack::book() 
{
  
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

    // set the DQM directory
    std::string MEFolderName = conf_.getParameter<std::string>("FolderName");    
    dbe->setCurrentFolder(MEFolderName);
    
    for (int j=0;j<off_Flag;j++) { // Loop on onTrack, offTrack
      name=flags[j]+"_TotalNumberOfClusters";
      if(MEMap.find(name)==MEMap.end()) {
	if(flags[j] == "OnTrack"){
	  MEMap[name]=bookME1D("TH1nClustersOn", name.Data());
	}else{
	  MEMap[name]=bookME1D("TH1nClustersOff", name.Data());
	}
	if(Trend_On_){
	  name+="Trend";
	  if(flags[j] == "OnTrack"){
	    MEMap[name]=bookMETrend("TH1nClustersOn", name.Data());
	  }else{
	    MEMap[name]=bookMETrend("TH1nClustersOff", name.Data());
	  }
	}      
      }
    }// End Loop on onTrack, offTrack
    
    // 	LogTrace("SiStripMonitorTrack") << " Detid " << *detid 
    //					<< " SubDet " << GetSubDetAndLayer(*detid).first 
    //					<< " Layer "  << GetSubDetAndLayer(*detid).second;

     
    // book Layer and RING plots      
    if (DetectedLayers.find(folder_organizer.GetSubDetAndLayer(detid,flag_ring)) == DetectedLayers.end()){
      
      DetectedLayers[folder_organizer.GetSubDetAndLayer(detid,flag_ring)]=true;
      for (int j=0;j<off_Flag;j++){     
	folder_organizer.setLayerFolder(*detid_iter,folder_organizer.GetSubDetAndLayer(*detid_iter,flag_ring).second,flag_ring);
	bookTrendMEs("layer",(folder_organizer.GetSubDetAndLayer(*detid_iter,flag_ring)).second,*detid_iter,flags[j]);
      }
    }
    
    if(Mod_On_){
      //    book module plots
      folder_organizer.setDetectorFolder(*detid_iter);
      bookModMEs("det",*detid_iter);
    }

    DetectedLayers[folder_organizer.GetSubDetAndLayer(*detid_iter,flag_ring)] |= (DetectedLayers.find(folder_organizer.GetSubDetAndLayer(*detid_iter,flag_ring)) == DetectedLayers.end());
  }//end loop on detectors detid
  
  //  book SubDet plots
  for (std::map<std::pair<std::string,int32_t>,bool>::const_iterator iter=DetectedLayers.begin(); iter!=DetectedLayers.end();iter++){
    for (int j=0;j<off_Flag;j++){ // Loop on onTrack, offTrack
      folder_organizer.setDetectorFolder(0);
      dbe->cd("SiStrip/MechanicalView/"+iter->first.first);
      name=flags[j]+"_in_"+iter->first.first; 
      bookSubDetMEs(name,flags[j]); //for subdets
    }//end loop on onTrack,offTrack
  }  

}
  
//--------------------------------------------------------------------------------
void SiStripMonitorTrack::bookModMEs(TString name, uint32_t id)//Histograms at MODULE level
{
  SiStripHistoId hidmanager;
  std::string hid = hidmanager.createHistoId("",name.Data(),id);
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(TString(hid));
  if(iModME==ModMEsMap.end()){
    ModMEs theModMEs; 
    // Cluster Width
    theModMEs.ClusterWidth=bookME1D("TH1ClusterWidth", hidmanager.createHistoId("ClusterWidth_OnTrack",name.Data(),id).c_str()); 
    dbe->tag(theModMEs.ClusterWidth,id); 
    // Cluster Charge
    theModMEs.ClusterCharge=bookME1D("TH1ClusterCharge", hidmanager.createHistoId("ClusterCharge_OnTrack",name.Data(),id).c_str());
    dbe->tag(theModMEs.ClusterCharge,id); 
    // Cluster StoN
    if(HistoFlag_On_){
      theModMEs.ClusterStoN=bookME1D("TH1ClusterStoN", hidmanager.createHistoId("ClusterStoN_OnTrack",name.Data(),id).c_str());
      dbe->tag(theModMEs.ClusterStoN,id); 
    // Cluster Charge Corrected
      theModMEs.ClusterChargeCorr=bookME1D("TH1ClusterChargeCorr", hidmanager.createHistoId("ClusterChargeCorr_OnTrack",name.Data(),id).c_str());
      dbe->tag(theModMEs.ClusterChargeCorr,id); 
    }
    // Cluster StoN Corrected
    theModMEs.ClusterStoNCorr=bookME1D("TH1ClusterStoNCorr", hidmanager.createHistoId("ClusterStoNCorr_OnTrack",name.Data(),id).c_str());
    dbe->tag(theModMEs.ClusterStoNCorr,id); 
    // Cluster Position
    short total_nr_strips = SiStripDetCabling_->nApvPairs(id) * 2 * 128;
    theModMEs.ClusterPos=dbe->book1D(hidmanager.createHistoId("ClusterPosition_OnTrack",name.Data(),id).c_str(),hidmanager.createHistoId("ClusterPosition_OnTrack",name.Data(),id).c_str(),total_nr_strips,0.5,total_nr_strips+0.5);
    dbe->tag(theModMEs.ClusterPos,id); 
    // Cluster PGV
    theModMEs.ClusterPGV=bookMEProfile("TProfileClusterPGV", hidmanager.createHistoId("PGV_OnTrack",name.Data(),id).c_str()); 
    dbe->tag(theModMEs.ClusterPGV,id); 

    ModMEsMap[hid]=theModMEs;
  }
}

void SiStripMonitorTrack::bookTrendMEs(TString name,int32_t layer,uint32_t id,std::string flag)//Histograms and Trends at LAYER LEVEL
{

  SiStripHistoId hidmanager;
  std::string rest = hidmanager.getSubdetid(id,flag_ring);
  std::string hid = hidmanager.createHistoLayer("",name.Data(),rest,flag);
  std::map<TString, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(TString(hid));
  if(iLayerME==LayerMEsMap.end()){
    LayerMEs theLayerMEs; 
 
    // Cluster Width
    theLayerMEs.ClusterWidth=bookME1D("TH1ClusterWidth", hidmanager.createHistoLayer("Summary_ClusterWidth",name.Data(),rest,flag).c_str()); 
    dbe->tag(theLayerMEs.ClusterWidth,layer); 
    
    // Cluster Noise
    theLayerMEs.ClusterNoise=bookME1D("TH1ClusterNoise", hidmanager.createHistoLayer("Summary_ClusterNoise",name.Data(),rest,flag).c_str()); 
    dbe->tag(theLayerMEs.ClusterNoise,layer); 
    
    // Cluster Charge
    theLayerMEs.ClusterCharge=bookME1D("TH1ClusterCharge", hidmanager.createHistoLayer("Summary_ClusterCharge",name.Data(),rest,flag).c_str());
    dbe->tag(theLayerMEs.ClusterCharge,layer);
    
    // Cluster StoN
    theLayerMEs.ClusterStoN=bookME1D("TH1ClusterStoN", hidmanager.createHistoLayer("Summary_ClusterStoN",name.Data(),rest,flag).c_str());
    dbe->tag(theLayerMEs.ClusterStoN,layer); 

    // Trends
    if(Trend_On_){
      // Cluster Width
      theLayerMEs.ClusterWidthTrend=bookMETrend("TH1ClusterWidth", hidmanager.createHistoLayer("Trend_ClusterWidth",name.Data(),rest,flag).c_str()); 
      dbe->tag(theLayerMEs.ClusterWidthTrend,layer); 
      // Cluster Noise
      theLayerMEs.ClusterNoiseTrend=bookMETrend("TH1ClusterNoise", hidmanager.createHistoLayer("Trend_ClusterNoise",name.Data(),rest,flag).c_str()); 
      dbe->tag(theLayerMEs.ClusterNoiseTrend,layer); 
      // Cluster Charge
      theLayerMEs.ClusterChargeTrend=bookMETrend("TH1ClusterCharge", hidmanager.createHistoLayer("Trend_ClusterCharge",name.Data(),rest,flag).c_str());
      dbe->tag(theLayerMEs.ClusterChargeTrend,layer); 
      // Cluster StoN
      theLayerMEs.ClusterStoNTrend=bookMETrend("TH1ClusterStoN", hidmanager.createHistoLayer("Trend_ClusterStoN",name.Data(),rest,flag).c_str());
      dbe->tag(theLayerMEs.ClusterStoNTrend,layer); 
    }
    
    if(flag=="OnTrack"){
      // Cluster Charge Corrected
      theLayerMEs.ClusterChargeCorr=bookME1D("TH1ClusterChargeCorr", hidmanager.createHistoLayer("Summary_ClusterChargeCorr",name.Data(),rest,flag).c_str());
      dbe->tag(theLayerMEs.ClusterChargeCorr,layer); 
      // Cluster StoN Corrected
      theLayerMEs.ClusterStoNCorr=bookME1D("TH1ClusterStoNCorr", hidmanager.createHistoLayer("Summary_ClusterStoNCorr",name.Data(),rest,flag).c_str());
      dbe->tag(theLayerMEs.ClusterStoNCorr,layer); 
      
      if(Trend_On_){
	// Cluster Charge Corrected
	theLayerMEs.ClusterChargeCorrTrend=bookMETrend("TH1ClusterChargeCorr", hidmanager.createHistoLayer("Trend_ClusterChargeCorr",name.Data(),rest,flag).c_str());
	dbe->tag(theLayerMEs.ClusterChargeCorrTrend,layer); 
	
	// Cluster StoN Corrected
	theLayerMEs.ClusterStoNCorrTrend=bookMETrend("TH1ClusterStoNCorr", hidmanager.createHistoLayer("Trend_ClusterStoNCorr",name.Data(),rest,flag).c_str());
	dbe->tag(theLayerMEs.ClusterStoNCorrTrend,layer); 
	
      }
      
    }
    
    //Cluster Position
    short total_nr_strips = SiStripDetCabling_->nApvPairs(id) * 2 * 128; 
    theLayerMEs.ClusterPos= dbe->book1D(hidmanager.createHistoLayer("Summary_ClusterPosition",name.Data(),rest,flag).c_str(),hidmanager.createHistoLayer("Summary_ClusterPosition",name.Data(),rest,flag).c_str(),total_nr_strips, 0.5,total_nr_strips+0.5);
    dbe->tag(theLayerMEs.ClusterPos,layer); 
    
    //bookeeping
    LayerMEsMap[hid]=theLayerMEs;
  }
  
}

void SiStripMonitorTrack::bookSubDetMEs(TString name,TString flag)//Histograms at SubDet level
{
  std::map<TString, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(name);
  char completeName[1024];
  if(iLayerME==LayerMEsMap.end()){
    LayerMEs theLayerMEs;
    
    // TotalNumber of Cluster 

    if (flag=="OnTrack"){
      sprintf(completeName,"Summary_TotalNumberOfClusters_%s",name.Data());
      theLayerMEs.nClusters=bookME1D("TH1nClustersOn", completeName);
      theLayerMEs.nClusters->getTH1()->StatOverflows(kTRUE);
    }else{
      sprintf(completeName,"Summary_TotalNumberOfClusters_%s",name.Data());
      theLayerMEs.nClusters=bookME1D("TH1nClustersOff", completeName);
      theLayerMEs.nClusters->getTH1()->StatOverflows(kTRUE);
    }
    
    // Cluster Width
    sprintf(completeName,"Summary_ClusterWidth_%s",name.Data());
    theLayerMEs.ClusterWidth=bookME1D("TH1ClusterWidth", completeName);
    
    // Cluster Noise
    sprintf(completeName,"Summary_ClusterNoise_%s",name.Data());
    theLayerMEs.ClusterNoise=bookME1D("TH1ClusterNoise", completeName);
    
    // Cluster Charge
    sprintf(completeName,"Summary_ClusterCharge_%s",name.Data());
    theLayerMEs.ClusterCharge=bookME1D("TH1ClusterCharge", completeName);
    
    // Cluster StoN
    sprintf(completeName,"Summary_ClusterStoN_%s",name.Data());
    theLayerMEs.ClusterStoN=bookME1D("TH1ClusterStoN", completeName);


    if(Trend_On_){
      if (flag=="OnTrack"){
	// TotalNumber of Cluster 
	sprintf(completeName,"Trend_TotalNumberOfClusters_%s",name.Data());
	theLayerMEs.nClustersTrend=bookMETrend("TH1nClustersOn", completeName);
      }else{
	sprintf(completeName,"Trend_TotalNumberOfClusters_%s",name.Data());
	theLayerMEs.nClustersTrend=bookMETrend("TH1nClustersOff", completeName);
      }
      // Cluster Width
      sprintf(completeName,"Trend_ClusterWidth_%s",name.Data());
      theLayerMEs.ClusterWidthTrend=bookMETrend("TH1ClusterWidth", completeName);
      // Cluster Noise
      sprintf(completeName,"Trend_ClusterNoise_%s",name.Data());
      theLayerMEs.ClusterNoiseTrend=bookMETrend("TH1ClusterNoise", completeName);
      // Cluster Charge
      sprintf(completeName,"Trend_ClusterCharge_%s",name.Data());
      theLayerMEs.ClusterChargeTrend=bookMETrend("TH1ClusterCharge", completeName);
      // Cluster StoN
      sprintf(completeName,"Trend_ClusterStoN_%s",name.Data());
      theLayerMEs.ClusterStoNTrend=bookMETrend("TH1ClusterStoN", completeName); 
    }

    if (flag=="OnTrack"){
      //Cluster StoNCorr
      sprintf(completeName,"Summary_ClusterStoNCorr_%s",name.Data());
      theLayerMEs.ClusterStoNCorr=bookME1D("TH1ClusterStoNCorr", completeName);
      
      // Cluster ChargeCorr
      sprintf(completeName,"Summary_ClusterChargeCorr_%s",name.Data());
      theLayerMEs.ClusterChargeCorr=bookME1D("TH1ClusterChargeCorr", completeName);
      
      if(Trend_On_){ 
	// Cluster StoNCorr
	sprintf(completeName,"Trend_ClusterStoNCorr_%s",name.Data());
	theLayerMEs.ClusterStoNCorrTrend=bookMETrend("TH1ClusterStoNCorr", completeName);     
	// Cluster ChargeCorr
	sprintf(completeName,"Trend_ClusterChargeCorr_%s",name.Data());
	theLayerMEs.ClusterChargeCorrTrend=bookMETrend("TH1ClusterChargeCorr", completeName);

      }
    }
    
    //bookeeping
    LayerMEsMap[name]=theLayerMEs;
  }
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
void SiStripMonitorTrack::trackStudy(const edm::EventSetup& es)
{

  const reco::TrackCollection tC = *(trackCollection.product());
  int i=0;
  std::vector<TrajectoryMeasurement> measurements;
  for(TrajTrackAssociationCollection::const_iterator it =  TItkAssociatorCollection->begin();it !=  TItkAssociatorCollection->end(); ++it){
    const edm::Ref<std::vector<Trajectory> > traj_iterator = it->key;  
    // Trajectory Map, extract Trajectory for this track
    reco::TrackRef trackref = it->val;
    LogTrace("SiStripMonitorTrack")
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

    measurements =traj_iterator->measurements();
    std::vector<TrajectoryMeasurement>::iterator traj_mes_iterator;
    int nhit=0;
    for(traj_mes_iterator=measurements.begin();traj_mes_iterator!=measurements.end();traj_mes_iterator++){//loop on measurements
      //trajectory local direction and position on detector
      LocalPoint  stateposition;
      LocalVector statedirection;
      
      TrajectoryStateOnSurface  updatedtsos=traj_mes_iterator->updatedState();
      ConstRecHitPointer ttrh=traj_mes_iterator->recHit();
      if (!ttrh->isValid()) {continue;}
      
      std::stringstream ss;
      
      nhit++;
      
      const ProjectedSiStripRecHit2D* phit     = dynamic_cast<const ProjectedSiStripRecHit2D*>( ttrh->hit() );
      const SiStripMatchedRecHit2D* matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>( ttrh->hit() );
      const SiStripRecHit2D* hit2D             = dynamic_cast<const SiStripRecHit2D*>( ttrh->hit() );	
      const SiStripRecHit1D* hit1D             = dynamic_cast<const SiStripRecHit1D*>( ttrh->hit() );	
      
      RecHitType type=Single;

      if(matchedhit){
	LogTrace("SiStripMonitorTrack")<<"\nMatched recHit found"<< std::endl;
	type=Matched;
	
	GluedGeomDet * gdet=(GluedGeomDet *)tkgeom->idToDet(matchedhit->geographicalId());
	GlobalVector gtrkdirup=gdet->toGlobal(updatedtsos.localMomentum());	    
	//mono side
	const GeomDetUnit * monodet=gdet->monoDet();
	statedirection=monodet->toLocal(gtrkdirup);
	if(statedirection.mag() != 0)	  RecHitInfo<SiStripRecHit2D>(matchedhit->monoHit(),statedirection,trackref,es);
	//stereo side
	const GeomDetUnit * stereodet=gdet->stereoDet();
	statedirection=stereodet->toLocal(gtrkdirup);
	if(statedirection.mag() != 0)	  RecHitInfo<SiStripRecHit2D>(matchedhit->stereoHit(),statedirection,trackref,es);
	ss<<"\nLocalMomentum (stereo): " <<  statedirection;
      }
      else if(phit){
	LogTrace("SiStripMonitorTrack")<<"\nProjected recHit found"<< std::endl;
	type=Projected;
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
	ss<<"\nSingle recHit2D found"<< std::endl;	  
	statedirection=updatedtsos.localMomentum();
	if(statedirection.mag() != 0) RecHitInfo<SiStripRecHit2D>(hit2D,statedirection,trackref,es);
      } else if (hit1D) {
	ss<<"\nSingle recHit1D found"<< std::endl;	  
	statedirection=updatedtsos.localMomentum();
	if(statedirection.mag() != 0) RecHitInfo<SiStripRecHit1D>(hit1D,statedirection,trackref,es);
      } else {
	ss <<"LocalMomentum: "<<statedirection
	   << "\nLocal x-z plane angle: "<<atan2(statedirection.x(),statedirection.z());	      
	LogTrace("SiStripMonitorTrack") <<ss.str() << std::endl;
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
    
    //Get SiStripCluster from SiStripRecHit
    if ( tkrecHit != NULL ){
      LogTrace("SiStripMonitorTrack") << "GOOD hit" << std::endl;
      const SiStripCluster* SiStripCluster_ = &*(tkrecHit->cluster());
      SiStripClusterInfo* SiStripClusterInfo_ = new SiStripClusterInfo(*SiStripCluster_,es);
            
      if ( clusterInfos(SiStripClusterInfo_,detid,"OnTrack", LV ) ) {
	vPSiStripCluster.push_back(SiStripCluster_);
	countOn++;
      }
      delete SiStripClusterInfo_; 
      //}
    }else{
     edm::LogError("SiStripMonitorTrack") << "NULL hit" << std::endl;
    }	  
  }

//------------------------------------------------------------------------

void SiStripMonitorTrack::AllClusters( const edm::EventSetup& es)
{

  //Loop on Dets
  for ( edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=dsv_SiStripCluster->begin(); DSViter!=dsv_SiStripCluster->end();DSViter++){
    uint32_t detid=DSViter->id();
    if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()) continue;
    //Loop on Clusters
    edm::LogInfo("SiStripMonitorTrack") << "on detid "<< detid << " N Cluster= " << DSViter->size();
    edmNew::DetSet<SiStripCluster>::const_iterator ClusIter = DSViter->begin();
    for(; ClusIter!=DSViter->end(); ClusIter++) {
      SiStripClusterInfo* SiStripClusterInfo_= new SiStripClusterInfo(*ClusIter,es);
	LogDebug("SiStripMonitorTrack") << "ClusIter " << &*ClusIter << "\t " 
	                                << std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter)-vPSiStripCluster.begin();
	if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	  if ( clusterInfos(SiStripClusterInfo_,detid,"OffTrack",LV) ) {
	    countOff++;
	  }
	}
	delete SiStripClusterInfo_; 
    }
  }
}

//------------------------------------------------------------------------
bool SiStripMonitorTrack::clusterInfos(SiStripClusterInfo* cluster, const uint32_t& detid,std::string flag, const LocalVector LV)
{
  LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
  //folder_organizer.setDetectorFolder(0);
  if (cluster==0) return false;
  // if one imposes a cut on the clusters, apply it
  const  edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
  if( ps.getParameter<bool>("On") &&
      (cluster->signalOverNoise() < ps.getParameter<double>("minStoN") ||
       cluster->signalOverNoise() > ps.getParameter<double>("maxStoN") ||
       cluster->width() < ps.getParameter<double>("minWidth") ||
       cluster->width() > ps.getParameter<double>("maxWidth")                    )) return false;
  // start of the analysis
  
  int SubDet_enum = StripSubdetector(detid).subdetId()-3;
  int iflag =0;
  if      (flag=="OnTrack")  iflag=0;
  else if (flag=="OffTrack") iflag=1;
  NClus[SubDet_enum][iflag]++;
  std::stringstream ss;
  //  const_cast<SiStripClusterInfo*>(cluster)->print(ss);
  LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]\n" << ss.str() << std::endl;
  
  float cosRZ = -2;
  LogTrace("SiStripMonitorTrack")<< "\n\tLV " << LV.x() << " " << LV.y() << " " << LV.z() << " " << LV.mag() << std::endl;
  if (LV.mag()!=0){
    cosRZ= fabs(LV.z())/LV.mag();
    LogTrace("SiStripMonitorTrack")<< "\n\t cosRZ " << cosRZ << std::endl;
  }
  std::string name;
  
  // Filling SubDet Plots (on Track + off Track)
  std::pair<std::string,int32_t> SubDetAndLayer = folder_organizer.GetSubDetAndLayer(detid,flag_ring);
  name=flag+"_in_"+SubDetAndLayer.first;
  fillMEs(cluster,name,cosRZ,flag);
  
  // Filling Layer Plots
  SiStripHistoId hidmanager1;
  std::string rest = hidmanager1.getSubdetid(detid,flag_ring);   
  name= hidmanager1.createHistoLayer("","layer",rest,flag);
  fillMEs(cluster,name,cosRZ,flag);
  
  
  //******** TkHistoMaps
  if (TkHistoMap_On_) {
    uint32_t adet=cluster->detId();
    if(flag=="OnTrack"){
      tkhisto_NumOnTrack->add(adet,1.);
      tkhisto_StoNCorrOnTrack->fill(adet,cluster->signalOverNoise()*cosRZ);
    }
    else if(flag=="OffTrack"){
      tkhisto_NumOffTrack->add(adet,1.);
      if(cluster->charge() > 250){
	edm::LogInfo("SiStripMonitorTrack") << "Module firing " << detid << " in Event " << eventNb << std::endl;
      }
    }
  }

  // Module plots filled only for onTrack Clusters
  if(Mod_On_){
    if(flag=="OnTrack"){
      SiStripHistoId hidmanager2;
      name =hidmanager2.createHistoId("","det",detid);
      fillModMEs(cluster,name,cosRZ); 
    }
  }
  return true;
}

//--------------------------------------------------------------------------------
void SiStripMonitorTrack::fillModMEs(SiStripClusterInfo* cluster,TString name,float cos)
{
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(name);
  if(iModME!=ModMEsMap.end()){
    fillME(iModME->second.ClusterStoNCorr ,cluster->signalOverNoise()*cos);
    fillME(iModME->second.ClusterCharge,cluster->charge());

    if(HistoFlag_On_){
      fillME(iModME->second.ClusterStoN  ,cluster->signalOverNoise());
      fillME(iModME->second.ClusterChargeCorr,cluster->charge()*cos);
    }
    fillME(iModME->second.ClusterWidth ,cluster->width());
    fillME(iModME->second.ClusterPos   ,cluster->baryStrip());
    
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
void SiStripMonitorTrack::fillMEs(SiStripClusterInfo* cluster,std::string name,float cos, std::string flag)
{ 
  std::map<TString, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(name);
  if(iLayerME!=LayerMEsMap.end()){
    if(flag=="OnTrack"){
      fillME(iLayerME->second.ClusterStoNCorr,(cluster->signalOverNoise())*cos);
      fillME(iLayerME->second.ClusterStoNCorrTrend,iOrbitSec,(cluster->signalOverNoise())*cos);
      fillME(iLayerME->second.ClusterChargeCorr,cluster->charge()*cos);
      fillME(iLayerME->second.ClusterChargeCorrTrend,iOrbitSec,cluster->charge()*cos);
    }
    fillME(iLayerME->second.ClusterStoN  ,cluster->signalOverNoise());
    fillME(iLayerME->second.ClusterStoNTrend,iOrbitSec,cluster->signalOverNoise());
    fillME(iLayerME->second.ClusterCharge,cluster->charge());
    fillME(iLayerME->second.ClusterChargeTrend,iOrbitSec,cluster->charge());
    fillME(iLayerME->second.ClusterNoise ,cluster->noiseRescaledByGain());
    fillME(iLayerME->second.ClusterNoiseTrend,iOrbitSec,cluster->noiseRescaledByGain());
    fillME(iLayerME->second.ClusterWidth ,cluster->width());
    fillME(iLayerME->second.ClusterWidthTrend,iOrbitSec,cluster->width());
    fillME(iLayerME->second.ClusterPos   ,cluster->baryStrip());
  }
}

