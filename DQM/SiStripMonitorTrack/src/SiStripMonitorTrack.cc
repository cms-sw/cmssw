#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "AnalysisDataFormats/TrackInfo/src/TrackInfo.cc"

#include "DQM/SiStripMonitorTrack/interface/SiStripMonitorTrack.h"

#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "TMath.h"

static const uint16_t _NUM_SISTRIP_SUBDET_ = 4;
static TString SubDet[_NUM_SISTRIP_SUBDET_]={"TIB","TID","TOB","TEC"};
static std::string flags[2] = {"OnTrack","OffTrack"};

SiStripMonitorTrack::SiStripMonitorTrack(const edm::ParameterSet& conf):
  dbe(edm::Service<DaqMonitorBEInterface>().operator->()),
  conf_(conf),
  Track_src_( conf.getParameter<edm::InputTag>( "Track_src" ) ),
  ClusterInfo_src_( conf.getParameter<edm::InputTag>( "ClusterInfo_src" ) ),
  Cluster_src_( conf.getParameter<edm::InputTag>( "Cluster_src" ) ),
  folder_organizer(), tracksCollection_in_EventTree(true),
  firstEvent(-1)
{
  for(int i=0;i<4;++i) for(int j=0;j<2;++j) NClus[i][j]=0;
}

//------------------------------------------------------------------------
SiStripMonitorTrack::~SiStripMonitorTrack() { }

//------------------------------------------------------------------------
void SiStripMonitorTrack::beginRun(const edm::Run& run,const edm::EventSetup& es)
{
  //get geom    
  es.get<TrackerDigiGeometryRecord>().get( tkgeom );
  edm::LogInfo("SiStripMonitorTrack") << "[SiStripMonitorTrack::beginJob] There are "<<tkgeom->detUnits().size() <<" detectors instantiated in the geometry" << std::endl;  
  es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );
  book();
}

//------------------------------------------------------------------------
void SiStripMonitorTrack::endJob(void)
{
  dbe->showDirStructure();
  if(conf_.getParameter<bool>("OutputMEsInRootFile"))
    dbe->save(conf_.getParameter<std::string>("OutputFileName"));
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
  
  //cluster input
  e.getByLabel( ClusterInfo_src_, dsv_SiStripClusterInfo);
  e.getByLabel( Cluster_src_,     dsv_SiStripCluster);    

  e.getByLabel(Track_src_, trackCollection);
  if (trackCollection.isValid()){
  }else{
    edm::LogError("SiStripMonitorTrack")<<" Track Collection is not valid !!" <<std::endl;
    tracksCollection_in_EventTree=false;
  }
  
  // track input
  edm::InputTag TkiTag = conf_.getParameter<edm::InputTag>( "TrackInfo" );
  e.getByLabel( TkiTag, TItkAssociatorCollection );
  if( TItkAssociatorCollection.isValid()){
  }else{
    edm::LogError("SiStripMonitorTrack")<<"trackInfo not found "<<std::endl;
    trackAssociatorCollection_in_EventTree=false;
  }
 
 //Perform track study
  if (tracksCollection_in_EventTree || trackAssociatorCollection_in_EventTree) trackStudy();
  
  //Perform Cluster Study (irrespectively to tracks)
  AllClusters();//analyzes the off Track Clusters

  //Summary Counts of clusters
  std::map<TString, MonitorElement*>::iterator iME;
  std::map<TString, ModMEs>::iterator          iModME ;
  for (int j=0;j<2;++j){ // loop over ontrack, offtrack
    int nTot=0;
    for (int i=0;i<4;++i){ // loop over TIB, TID, TOB, TEC
      name=flags[j]+"_in_"+SubDet[i];      
      iModME = ModMEsMap.find(name);
      if(iModME!=ModMEsMap.end()) {
       	fillME(   iModME->second.nClusters,      NClus[i][j]);
	fillTrend(iModME->second.nClustersTrend, NClus[i][j]);
      }
      nTot+=NClus[i][j];
      NClus[i][j]=0;
    }// loop over TIB, TID, TOB, TEC
  
    name=flags[j]+"_NumberOfClusters";
    iME = MEMap.find(name);
    if(iME!=MEMap.end()) iME->second->Fill(nTot);
    iME = MEMap.find(name+"Trend");
    if(iME!=MEMap.end()) fillTrend(iME->second,nTot);

  } // loop over ontrack, offtrack
  
}

//------------------------------------------------------------------------  
void SiStripMonitorTrack::book() 
{
  
  std::vector<uint32_t> vdetId_;
  SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
  //Histos for each detector, layer and module
  for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){  //loop on all the active detid
    uint32_t detid = *detid_iter;
    
    if (detid < 1){
      edm::LogError("SiStripMonitorTrack")<< "[" <<__PRETTY_FUNCTION__ << "] invalid detid " << detid<< std::endl;
      continue;
    }
    if (DetectedLayers.find(GetSubDetAndLayer(detid)) == DetectedLayers.end()){
      
      DetectedLayers[GetSubDetAndLayer(detid)]=true;
    }    

    // set the DQM directory
    dbe->setCurrentFolder("Track/GlobalParameters");
    
    for (int j=0;j<2;j++) { // Loop on onTrack, offTrack
      name=flags[j]+"_NumberOfClusters";
      if(MEMap.find(name)==MEMap.end()) {
	MEMap[name]=bookME1D("TH1nClusters", name.Data()); 
	name+="Trend";
	MEMap[name]=bookMETrend("TH1nClusters", name.Data());
      }
    }// End Loop on onTrack, offTrack
    
    // 	LogTrace("SiStripMonitorTrack") << " Detid " << *detid 
    //					<< " SubDet " << GetSubDetAndLayer(*detid).first 
    //					<< " Layer "  << GetSubDetAndLayer(*detid).second;

    const edm::ParameterSet mod = conf_.getParameter<edm::ParameterSet>("Layers");
    if (mod.getParameter<bool>("Lay_On")) {
    // book Layer plots      
      for (int j=0;j<2;j++){ 
	folder_organizer.setLayerFolder(*detid_iter,GetSubDetAndLayer(*detid_iter).second); 
	bookTrendMEs("layer",GetSubDetAndLayer(*detid_iter).second,*detid_iter,flags[j]);
      }
    }else{
      //    book module plots
      folder_organizer.setDetectorFolder(*detid_iter);
      bookModMEs("det",*detid_iter);
    }
    DetectedLayers[GetSubDetAndLayer(*detid_iter)] |= (DetectedLayers.find(GetSubDetAndLayer(*detid_iter)) == DetectedLayers.end());
    //      }
  }//end loop on detector
  
  // book SubDet plots
      for (std::map<std::pair<std::string,int32_t>,bool>::const_iterator iter=DetectedLayers.begin(); iter!=DetectedLayers.end();iter++){
        for (int j=0;j<2;j++){ // Loop on onTrack, offTrack
  	folder_organizer.setDetectorFolder(0);
  	dbe->cd(iter->first.first);
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
    //Cluster Width
    theModMEs.ClusterWidth=bookME1D("TH1ClusterWidth", hidmanager.createHistoId("cWidth",name.Data(),id).c_str()); 
    dbe->tag(theModMEs.ClusterWidth,id); 
    //Cluster Charge
    theModMEs.ClusterCharge=bookME1D("TH1ClusterCharge", hidmanager.createHistoId("cCharge",name.Data(),id).c_str());
    dbe->tag(theModMEs.ClusterCharge,id); 
    //Cluster StoN
    theModMEs.ClusterStoN=bookME1D("TH1ClusterStoN", hidmanager.createHistoId("cStoN",name.Data(),id).c_str());
    dbe->tag(theModMEs.ClusterStoN,id); 
    //Cluster Charge Corrected
    theModMEs.ClusterChargeCorr=bookME1D("TH1ClusterChargeCorr", hidmanager.createHistoId("cChargeCorr",name.Data(),id).c_str());
    dbe->tag(theModMEs.ClusterChargeCorr,id); 
    //Cluster StoN Corrected
    theModMEs.ClusterStoNCorr=bookME1D("TH1ClusterStoNCorr", hidmanager.createHistoId("cStoNCorr",name.Data(),id).c_str());
    dbe->tag(theModMEs.ClusterStoNCorr,id); 
    //Cluster Position
    theModMEs.ClusterPos=bookME1D("TH1ClusterPos", hidmanager.createHistoId("cPos",name.Data(),id).c_str());  
    dbe->tag(theModMEs.ClusterPos,id); 
    //Cluster PGV
    theModMEs.ClusterPGV=bookMEProfile("TProfileClusterPGV", hidmanager.createHistoId("cPGV",name.Data(),id).c_str()); 
    dbe->tag(theModMEs.ClusterPGV,id); 
    //bookeeping
    ModMEsMap[hid]=theModMEs;
  }
}

void SiStripMonitorTrack::bookTrendMEs(TString name,int32_t layer,uint32_t id,std::string flag)//Histograms and Trends at LAYER LEVEL
{
  char rest[1024];
  int subdetid = ((id>>25)&0x7);
  if(       subdetid==3 ){
  // ---------------------------  TIB  --------------------------- //
    TIBDetId tib1 = TIBDetId(id);
    sprintf(rest,"TIB__layer__%d",tib1.layer());
  }else if( subdetid==4){
  // ---------------------------  TID  --------------------------- //
    TIDDetId tid1 = TIDDetId(id);
    sprintf(rest,"TID__side__%d__wheel__%d",tid1.side(),tid1.wheel());
  }else if( subdetid==5){
  // ---------------------------  TOB  --------------------------- //
    TOBDetId tob1 = TOBDetId(id);
    sprintf(rest,"TOB__layer__%d",tob1.layer());
  }else if( subdetid==6){
  // ---------------------------  TEC  --------------------------- //
    TECDetId tec1 = TECDetId(id);
    sprintf(rest,"TEC__side__%d__wheel__%d",tec1.side(),tec1.wheel());
  }else{
  // ---------------------------  ???  --------------------------- //
    edm::LogError("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<subdetid<<" no folder set!"<<std::endl;
    return;
  }

  SiStripHistoId hidmanager;
  std::string hid = hidmanager.createHistoLayer("",name.Data(),rest,flag);
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(TString(hid));
  if(iModME==ModMEsMap.end()){
    ModMEs theModMEs; 
    //Cluster Width
    theModMEs.ClusterWidth=bookME1D("TH1ClusterWidth", hidmanager.createHistoLayer("cWidth",name.Data(),rest,flag).c_str()); 
    LogTrace("SiStripMonitorTrack") << "booking histogram: "<<  hidmanager.createHistoLayer("cWidth",name.Data(),rest,flag).c_str() << std::endl;
    dbe->tag(theModMEs.ClusterWidth,layer); 
    theModMEs.ClusterWidthTrend=bookMETrend("TH1ClusterWidth", hidmanager.createHistoLayer("Trend_cWidth",name.Data(),rest,flag).c_str()); 
    dbe->tag(theModMEs.ClusterWidthTrend,layer); 
    LogTrace("SiStripMonitorTrack") << "booking histogram: "<<  hidmanager.createHistoLayer("Trend_cWidth",name.Data(),rest,flag).c_str() << std::endl;

    //Cluster Noise
    theModMEs.ClusterNoise=bookME1D("TH1ClusterNoise", hidmanager.createHistoLayer("cNoise",name.Data(),rest,flag).c_str()); 
    dbe->tag(theModMEs.ClusterNoise,layer); 
    theModMEs.ClusterNoiseTrend=bookMETrend("TH1ClusterNoise", hidmanager.createHistoLayer("Trend_cNoise",name.Data(),rest,flag).c_str()); 
    dbe->tag(theModMEs.ClusterNoiseTrend,layer); 
    //Cluster Charge
    theModMEs.ClusterCharge=bookME1D("TH1ClusterCharge", hidmanager.createHistoLayer("cCharge",name.Data(),rest,flag).c_str());
    dbe->tag(theModMEs.ClusterCharge,layer);
    theModMEs.ClusterChargeTrend=bookMETrend("TH1ClusterCharge", hidmanager.createHistoLayer("Trend_cCharge",name.Data(),rest,flag).c_str());
    dbe->tag(theModMEs.ClusterChargeTrend,layer); 
    //Cluster StoN
    theModMEs.ClusterStoN=bookME1D("TH1ClusterStoN", hidmanager.createHistoLayer("cStoN",name.Data(),rest,flag).c_str());
    dbe->tag(theModMEs.ClusterStoN,layer); 
    theModMEs.ClusterStoNTrend=bookMETrend("TH1ClusterStoN", hidmanager.createHistoLayer("Trend_cStoN",name.Data(),rest,flag).c_str());
    dbe->tag(theModMEs.ClusterStoNTrend,layer); 
    if(flag=="OnTrack"){
      //Cluster Charge Corrected
      theModMEs.ClusterChargeCorr=bookME1D("TH1ClusterChargeCorr", hidmanager.createHistoLayer("cChargeCorr",name.Data(),rest,flag).c_str());
      dbe->tag(theModMEs.ClusterChargeCorr,layer); 
      theModMEs.ClusterChargeCorrTrend=bookMETrend("TH1ClusterChargeCorr", hidmanager.createHistoLayer("Trend_cChargeCorr",name.Data(),rest,flag).c_str());
      dbe->tag(theModMEs.ClusterChargeCorrTrend,layer); 
      //Cluster StoN Corrected
      theModMEs.ClusterStoNCorr=bookME1D("TH1ClusterStoNCorr", hidmanager.createHistoLayer("cStoNCorr",name.Data(),rest,flag).c_str());
      dbe->tag(theModMEs.ClusterStoNCorr,layer); 
      theModMEs.ClusterStoNCorrTrend=bookMETrend("TH1ClusterStoNCorr", hidmanager.createHistoLayer("Trend_cStoNCorr",name.Data(),rest,flag).c_str());
      //      dbe->tag(theModMEs.ClusterStoNCorrTrend,layer); 
    }
    //Cluster Position
    theModMEs.ClusterPos=bookME1D("TH1ClusterPos", hidmanager.createHistoLayer("cPos",name.Data(),rest,flag).c_str());  
    dbe->tag(theModMEs.ClusterPos,layer); 
    //bookeeping
    ModMEsMap[hid]=theModMEs;
  }

}

void SiStripMonitorTrack::bookSubDetMEs(TString name,TString flag)//Histograms at SubDet level
{
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(name);
  char completeName[1024];
  if(iModME==ModMEsMap.end()){
    ModMEs theModMEs; 
    //Number of Cluster 
    sprintf(completeName,"Summary_Trend_NumberOfClusters_%s",name.Data());
    theModMEs.nClustersTrend=bookMETrend("TH1nClusters", completeName);
    sprintf(completeName,"Summary_NumberOfClusters_%s",name.Data());
    theModMEs.nClusters=bookME1D("TH1nClusters", completeName);
    //Cluster Width
    sprintf(completeName,"Summary_Trend_cWidth_%s",name.Data());
    theModMEs.ClusterWidthTrend=bookMETrend("TH1ClusterWidth", completeName);
    sprintf(completeName,"Summary_cWidth_%s",name.Data());
    theModMEs.ClusterWidth=bookME1D("TH1ClusterWidth", completeName);
    //Cluster Noise
    sprintf(completeName,"Summary_Trend_cNoise_%s",name.Data());
    theModMEs.ClusterNoiseTrend=bookMETrend("TH1ClusterNoise", completeName);
    sprintf(completeName,"Summary_cNoise_%s",name.Data());
    theModMEs.ClusterNoise=bookME1D("TH1ClusterNoise", completeName);
    //Cluster Charge
    sprintf(completeName,"Summary_Trend_cCharge_%s",name.Data());
    theModMEs.ClusterChargeTrend=bookMETrend("TH1ClusterCharge", completeName);
    sprintf(completeName,"Summary_cCharge_%s",name.Data());
    theModMEs.ClusterCharge=bookME1D("TH1ClusterCharge", completeName);
    //Cluster StoN
    sprintf(completeName,"Summary_Trend_cStoN_%s",name.Data());
    theModMEs.ClusterStoNTrend=bookMETrend("TH1ClusterStoN", completeName);
    sprintf(completeName,"Summary_cStoN_%s",name.Data());
    theModMEs.ClusterStoN=bookME1D("TH1ClusterStoN", completeName);
    if (flag=="OnTrack"){    //Cluster StoNCorr
      sprintf(completeName,"Summary_Trend_cStoNCorr_%s",name.Data());
      theModMEs.ClusterStoNCorrTrend=bookMETrend("TH1ClusterStoNCorr", completeName);
      sprintf(completeName,"Summary_cStoNCorr_%s",name.Data());
      theModMEs.ClusterStoNCorr=bookME1D("TH1ClusterStoNCorr", completeName);
      
      //Cluster ChargeCorr
      sprintf(completeName,"Summary_Trend_cChargeCorr_%s",name.Data());
      theModMEs.ClusterChargeCorrTrend=bookMETrend("TH1ClusterChargeCorr", completeName);
      sprintf(completeName,"Summary_cChargeCorr_%s",name.Data());
      theModMEs.ClusterChargeCorr=bookME1D("TH1ClusterChargeCorr", completeName);
}
    //bookeeping
    ModMEsMap[name]=theModMEs;
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
  if(!me) return me;
  char buffer[256];
  sprintf(buffer,"EventId/%d",ParametersTrend.getParameter<int32_t>("Steps"));
  me->setAxisTitle(std::string(buffer),1);
  return me;
}

//------------------------------------------------------------------------------------------
void SiStripMonitorTrack::trackStudy()
{
  LogDebug("SiStripMonitorTrack") << "Start" << __PRETTY_FUNCTION__ ;
  const reco::TrackCollection tC = *(trackCollection.product());
  int i=0;
  for (unsigned int track=0;track<trackCollection->size();++track){
    // TrackInfo Map, extract TrackInfo for this track
    reco::TrackRef trackref = reco::TrackRef(trackCollection, track); 
    reco::TrackInfoRef trackinforef=(*TItkAssociatorCollection.product())[trackref];
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

//     std::vector<std::pair<const TrackingRecHit*, float> > hitangle;
//     hitangle = SeparateHits(trackinforef);

    reco::TrackInfo::TrajectoryInfo::const_iterator iter;
    
    for(iter=trackinforef->trajStateMap().begin();iter!=trackinforef->trajStateMap().end();iter++){
      
      //trajectory local direction and position on detector
      LocalVector statedirection=(trackinforef->stateOnDet(Updated,(*iter).first)->parameters()).momentum();
      LocalPoint  stateposition=(trackinforef->stateOnDet(Updated,(*iter).first)->parameters()).position();
      
      std::stringstream ss;
      ss <<"LocalMomentum: "<<statedirection
	 <<"\nLocalPosition: "<<stateposition
	 << "\nLocal x-z plane angle: "<<atan2(statedirection.x(),statedirection.z());
      
      if(trackinforef->type((*iter).first)==Matched){ // get the direction for the components
	
	const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(iter)->first));
	if (matchedhit!=0){
	  ss<<"\nMatched recHit found"<< std::endl;	  
	  //mono side
	  statedirection= trackinforef->localTrackMomentumOnMono(Updated,(*iter).first);
	  if(statedirection.mag() != 0)	  RecHitInfo(matchedhit->monoHit(),statedirection,trackref);
	  //stereo side
	  statedirection= trackinforef->localTrackMomentumOnStereo(Updated,(*iter).first);
	  if(statedirection.mag() != 0)	  RecHitInfo(matchedhit->stereoHit(),statedirection,trackref);
	  ss<<"\nLocalMomentum (stereo): "<<trackinforef->localTrackMomentumOnStereo(Updated,(*iter).first);
	}
      }
      else if (trackinforef->type((*iter).first)==Projected){//one should be 0
	ss<<"\nProjected recHit found"<< std::endl;
	const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(&(*(iter)->first));
	if(phit!=0){
	  //mono side
	  statedirection= trackinforef->localTrackMomentumOnMono(Updated,(*iter).first);
	  if(statedirection.mag() != 0) RecHitInfo(&(phit->originalHit()),statedirection,trackref);
	  //stereo side
	  statedirection= trackinforef->localTrackMomentumOnStereo(Updated,(*iter).first);
	  if(statedirection.mag() != 0)  RecHitInfo(&(phit->originalHit()),statedirection,trackref);
	}	
	
      }
      else {
	const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(&(*(iter)->first));
	if(hit!=0){
	  ss<<"\nSingle recHit found"<< std::endl;	  
	  statedirection=(trackinforef->stateOnDet(Updated,(*iter).first)->parameters()).momentum();
	  if(statedirection.mag() != 0) RecHitInfo(hit,statedirection,trackref);
	  
	}
      }
      LogTrace("TrackInfoAnalyzerExample") <<ss.str() << std::endl;
    }
  }
}

  void SiStripMonitorTrack::RecHitInfo(const SiStripRecHit2D* tkrecHit, LocalVector LV,reco::TrackRef track_ref ){
    
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
      const SiStripClusterInfo* SiStripClusterInfo_ = MatchClusterInfo(SiStripCluster_,detid);
      
      if ( clusterInfos(SiStripClusterInfo_,detid,"OnTrack", LV ) ) {
	vPSiStripCluster.push_back(SiStripCluster_);
	countOn++;
      }
      //}
    }else{
      LogTrace("SiStripMonitorTrack") << "NULL hit" << std::endl;
    }	  
  }

//------------------------------------------------------------------------

void SiStripMonitorTrack::AllClusters()
{

  LogDebug("SiStripMonitorTrack") << "Start cluster analysis" ;
  //Loop on Dets
  edm::DetSetVector<SiStripCluster>::const_iterator DSViter=dsv_SiStripCluster->begin();
  for (; DSViter!=dsv_SiStripCluster->end();DSViter++){
    uint32_t detid=DSViter->id;
    if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()) continue;
    //Loop on Clusters
    LogDebug("SiStripMonitorTrack") << "on detid "<< detid << " N Cluster= " << DSViter->data.size();
    edm::DetSet<SiStripCluster>::const_iterator ClusIter = DSViter->data.begin();
    for(; ClusIter!=DSViter->data.end(); ClusIter++) {
      const SiStripClusterInfo* SiStripClusterInfo_=MatchClusterInfo(&*ClusIter,detid);
	LogDebug("SiStripMonitorTrack") << "ClusIter " << &*ClusIter << "\t " 
	                                << std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter)-vPSiStripCluster.begin();
	if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	  if ( clusterInfos(SiStripClusterInfo_,detid,"OffTrack",LV) ) {
	    countOff++;
	  }
	}
    }
  }
}


//------------------------------------------------------------------------
const SiStripClusterInfo* SiStripMonitorTrack::MatchClusterInfo(const SiStripCluster* cluster, const uint32_t& detid)
{
  LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
  edm::DetSetVector<SiStripClusterInfo>::const_iterator DSViter = dsv_SiStripClusterInfo->find(detid);
  edm::DetSet<SiStripClusterInfo>::const_iterator ClusIter = DSViter->data.begin();
  for(; ClusIter!=DSViter->data.end(); ClusIter++) {
    if((ClusIter->firstStrip() == cluster->firstStrip()) &&
       (ClusIter->stripAmplitudes().size() == cluster->amplitudes().size()))
      return &(*ClusIter);
  }
  edm::LogError("SiStripMonitorTrack") << "Matching of SiStripCluster and SiStripClusterInfo is failed for cluster on detid "
				       << detid << "\n\tReturning NULL pointer";
  return 0;
}

//------------------------------------------------------------------------
bool SiStripMonitorTrack::clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,std::string flag, const LocalVector LV)
{
  LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
  //folder_organizer.setDetectorFolder(0);
  if (cluster==0) return false;
  // if one imposes a cut on the clusters, apply it
  const  edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
  if( ps.getParameter<bool>("On") &&
      (cluster->charge()/cluster->noise() < ps.getParameter<double>("minStoN") ||
       cluster->charge()/cluster->noise() > ps.getParameter<double>("maxStoN") ||
       cluster->width() < ps.getParameter<double>("minWidth") ||
       cluster->width() > ps.getParameter<double>("maxWidth")                    )) return false;
  // start of the analysis
  
  int SubDet_enum = StripSubdetector(detid).subdetId()-3;
  int iflag;
  if      (flag=="OnTrack")  iflag=0;
  else if (flag=="OffTrack") iflag=1;
  NClus[SubDet_enum][iflag]++;
  std::stringstream ss;
  const_cast<SiStripClusterInfo*>(cluster)->print(ss);
  LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]\n" << ss.str() << std::endl;
  
  float cosRZ = -2;
  LogTrace("SiStripMonitorTrack")<< "\n\tLV " << LV.x() << " " << LV.y() << " " << LV.z() << " " << LV.mag() << std::endl;
  if (LV.mag()!=0){
    cosRZ= fabs(LV.z())/LV.mag();
    LogTrace("SiStripMonitorTrack")<< "\n\t cosRZ " << cosRZ << std::endl;
  }
  std::string name;
  
   //Filling SubDet Plots (on Track + off Track)
   std::pair<std::string,int32_t> SubDetAndLayer = GetSubDetAndLayer(detid);
   name=flag+"_in_"+SubDetAndLayer.first;
   fillTrendMEs(cluster,name,cosRZ,flag);

   char rest[1024];
   int subdetid = ((detid>>25)&0x7);
   if(       subdetid==3 ){
     // ---------------------------  TIB  --------------------------- //
     TIBDetId tib1 = TIBDetId(detid);
     sprintf(rest,"TIB__layer__%d",tib1.layer());
   }else if( subdetid==4){
     // ---------------------------  TID  --------------------------- //
     TIDDetId tid1 = TIDDetId(detid);
     sprintf(rest,"TID__side__%d__wheel__%d",tid1.side(),tid1.wheel());
   }else if( subdetid==5){
     // ---------------------------  TOB  --------------------------- //
     TOBDetId tob1 = TOBDetId(detid);
     sprintf(rest,"TOB__layer__%d",tob1.layer());
   }else if( subdetid==6){
     // ---------------------------  TEC  --------------------------- //
     TECDetId tec1 = TECDetId(detid);
     sprintf(rest,"TEC__side__%d__wheel__%d",tec1.side(),tec1.wheel());
   }else{
     // ---------------------------  ???  --------------------------- //
     edm::LogError("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<subdetid<<" no folder set!"<<std::endl;
     return 0;
   }
   
   SiStripHistoId hidmanager1;
   
   //Filling Layer Plots
   const edm::ParameterSet _mod = conf_.getParameter<edm::ParameterSet>("Layers");
   if (_mod.getParameter<bool>("Lay_On")) { 
     name=hidmanager1.createHistoLayer("","layer",rest,flag);
     LogTrace("SiStripMonitorTrack") << "fill " << name << std::endl;
     fillTrendMEs(cluster,name,cosRZ,flag);
   }else{
     //Module plots filled only for onTrack Clusters
     if(flag=="OnTrack"){
       SiStripHistoId hidmanager2;
       name = hidmanager2.createHistoId("","det",detid);
       fillModMEs(cluster,name,cosRZ); 
     }
   }
   return true;
}

//--------------------------------------------------------------------------------
std::pair<std::string,int32_t> SiStripMonitorTrack::GetSubDetAndLayer(const uint32_t& detid)
{
  std::string cSubDet;
  int32_t layer=0;
  switch(StripSubdetector::SubDetector(StripSubdetector(detid).subdetId()))
    {
    case StripSubdetector::TIB:
      cSubDet="TIB";
      layer=TIBDetId(detid).layer();
      break;
    case StripSubdetector::TOB:
      cSubDet="TOB";
      layer=TOBDetId(detid).layer();
      break;
    case StripSubdetector::TID:
      cSubDet="TID";
      layer=TIDDetId(detid).wheel() * ( TIDDetId(detid).side()==1 ? -1 : +1);
      break;
    case StripSubdetector::TEC:
      cSubDet="TEC";
      layer=TECDetId(detid).wheel() * ( TECDetId(detid).side()==1 ? -1 : +1);
      break;
    default:
      edm::LogWarning("SiStripMonitorTrack") << "WARNING!!! this detid does not belong to tracker" << std::endl;
    }
  return std::make_pair(cSubDet,layer);
}

//--------------------------------------------------------------------------------
void SiStripMonitorTrack::fillTrend(MonitorElement* me ,float value)
{
  if(!me) return;
  //check the origin and check options
  int option = conf_.getParameter<edm::ParameterSet>("Trending").getParameter<int32_t>("UpdateMode");
  if(firstEvent==-1) firstEvent = eventNb;
  int CurrentStep = atoi(me->getAxisTitle(1).c_str()+8);
  int firstEventUsed = firstEvent;
  int presentOverflow = (int)me->getBinEntries(me->getNbinsX()+1);
  if(option==2) firstEventUsed += CurrentStep * int(me->getBinEntries(me->getNbinsX()+1));
  else if(option==3) firstEventUsed += CurrentStep * int(me->getBinEntries(me->getNbinsX()+1)) * me->getNbinsX();
  //fill
  me->Fill((eventNb-firstEventUsed)/CurrentStep,value);
  if(eventNb-firstEvent<1) return;
  // check if we reached the end
  if(presentOverflow == me->getBinEntries(me->getNbinsX()+1)) return;
  switch(option) {
  case 1:
    {
      // mode 1: rebin and change X scale
      int NbinsX = me->getNbinsX();
      float entries = 0.;
      float content = 0.;
      float error = 0.;
      int bin = 1;
      int totEntries = int(me->getEntries());
      for(;bin<=NbinsX/2;++bin) {
	content = (me->getBinContent(2*bin-1) + me->getBinContent(2*bin))/2.; 
	error   = pow((me->getBinError(2*bin-1)*me->getBinError(2*bin-1)) + (me->getBinError(2*bin)*me->getBinError(2*bin)),0.5)/2.; 
	entries = me->getBinEntries(2*bin-1) + me->getBinEntries(2*bin);
	me->setBinContent(bin,content*entries);
	me->setBinError(bin,error);
	me->setBinEntries(bin,entries);
      }
      for(;bin<=NbinsX+1;++bin) {
	me->setBinContent(bin,0);
	me->setBinError(bin,0);
	me->setBinEntries(bin,0); 
      }
      me->setEntries(totEntries);
      char buffer[256];
      sprintf(buffer,"EventId/%d",CurrentStep*2);
      me->setAxisTitle(std::string(buffer),1);
      break;
    }
  case 2:
    {
      // mode 2: slide
      int bin=1;
      int NbinsX = me->getNbinsX();
      for(;bin<=NbinsX;++bin) {
	me->setBinContent(bin,me->getBinContent(bin+1)*me->getBinEntries(bin+1));
	me->setBinError(bin,me->getBinError(bin+1));
	me->setBinEntries(bin,me->getBinEntries(bin+1));
      }
      break;
    }
  case 3:
    {
      // mode 3: reset
      int NbinsX = me->getNbinsX();
      for(int bin=0;bin<=NbinsX;++bin) {
	me->setBinContent(bin,0);
	me->setBinError(bin,0);
	me->setBinEntries(bin,0); 
      }
      break;
    }
  }
}

//--------------------------------------------------------------------------------
void SiStripMonitorTrack::fillModMEs(const SiStripClusterInfo* cluster,TString name,float cos)
{
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(name);
  if(iModME!=ModMEsMap.end()){
    fillME(iModME->second.ClusterStoN  ,cluster->charge()/cluster->noise());
    fillME(iModME->second.ClusterStoNCorr ,cluster->charge()*cos/cluster->noise());
    fillME(iModME->second.ClusterCharge,cluster->charge());
    fillME(iModME->second.ClusterChargeCorr,cluster->charge()*cos);
    fillME(iModME->second.ClusterWidth ,cluster->width());
    fillME(iModME->second.ClusterPos   ,cluster->position());

    //fill the PGV histo
    float PGVmax = cluster->maxCharge();
    int PGVposCounter = cluster->firstStrip() - cluster->rawdigiAmplitudesL().size() - cluster->maxPos();
    for (int i= int(conf_.getParameter<edm::ParameterSet>("TProfileClusterPGV").getParameter<double>("xmin"));i<PGVposCounter;++i)
      fillME(iModME->second.ClusterPGV, i,0.);
    for (std::vector<float>::const_iterator it=cluster->rawdigiAmplitudesL().begin();it<cluster->rawdigiAmplitudesL().end();++it) {
      fillME(iModME->second.ClusterPGV, PGVposCounter++,(*it)/PGVmax);
    }
    for (std::vector<uint16_t>::const_iterator it=cluster->stripAmplitudes().begin();it<cluster->stripAmplitudes().end();++it) {
      fillME(iModME->second.ClusterPGV, PGVposCounter++,(*it)/PGVmax);
    }
    for (std::vector<float>::const_iterator it=cluster->rawdigiAmplitudesR().begin();it<cluster->rawdigiAmplitudesR().end();++it) {
      fillME(iModME->second.ClusterPGV, PGVposCounter++,(*it)/PGVmax);
    }
    for (int i= PGVposCounter;i<int(conf_.getParameter<edm::ParameterSet>("TProfileClusterPGV").getParameter<double>("xmax"));++i)
      fillME(iModME->second.ClusterPGV, i,0.);
    //end fill the PGV histo
  }
}

void SiStripMonitorTrack::fillTrendMEs(const SiStripClusterInfo* cluster,std::string name,float cos, std::string flag)
{ 
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(name);
  if(iModME!=ModMEsMap.end()){
    if(flag=="OnTrack"){
      fillME(iModME->second.ClusterStoNCorr,(cluster->charge()/cluster->noise())*cos);
      fillTrend(iModME->second.ClusterStoNCorrTrend,(cluster->charge()/cluster->noise())*cos);
      fillME(iModME->second.ClusterChargeCorr,cluster->charge()*cos);
      fillTrend(iModME->second.ClusterChargeCorrTrend,cluster->charge()*cos);
    }
    fillME(iModME->second.ClusterStoN  ,cluster->charge()/cluster->noise());
    fillTrend(iModME->second.ClusterStoNTrend,cluster->charge()/cluster->noise());
    fillME(iModME->second.ClusterCharge,cluster->charge());
    fillTrend(iModME->second.ClusterChargeTrend,cluster->charge());
    fillME(iModME->second.ClusterNoise ,cluster->noise());
    fillTrend(iModME->second.ClusterNoiseTrend,cluster->noise());
    fillME(iModME->second.ClusterWidth ,cluster->width());
    fillTrend(iModME->second.ClusterWidthTrend,cluster->width());
    fillME(iModME->second.ClusterPos   ,cluster->position());
  }
}
DEFINE_FWK_MODULE(SiStripMonitorTrack);
