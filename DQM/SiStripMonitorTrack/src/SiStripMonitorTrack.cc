#include <string>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DQM/SiStripMonitorTrack/interface/SiStripMonitorTrack.h"

static const uint16_t _NUM_SISTRIP_SUBDET_ = 4;
static TString SubDet[_NUM_SISTRIP_SUBDET_]={"TIB","TOB","TID","TEC"};
static std::string flags[3] = {"OnTrack","OffTrack","All"};
static TString width_flags[5] = {"","_width_1","_width_2","_width_3","_width_ge_4"};

SiStripMonitorTrack::SiStripMonitorTrack(const edm::ParameterSet& conf):
  dbe(edm::Service<DaqMonitorBEInterface>().operator->()),
  conf_(conf),
  firstEvent(-1),
  SiStripNoiseService_(conf),
  SiStripPedestalsService_(conf),
  Track_src_( conf.getParameter<edm::InputTag>( "Track_src" ) ),
  ClusterInfo_src_( conf.getParameter<edm::InputTag>( "ClusterInfo_src" ) ),
  Cluster_src_( conf.getParameter<edm::InputTag>( "Cluster_src" ) ),
  EtaAlgo_(conf.getParameter<int32_t>("EtaAlgo")),
  NeighStrips_(conf.getParameter<int32_t>("NeighStrips")),
  tracksCollection_in_EventTree(true)
{
 for(int i=0;i<4;++i) for(int j=0;j<3;++j) NClus[i][j]=0;
}
//------------------------------------------------------------------------
SiStripMonitorTrack::~SiStripMonitorTrack()
{
}
//------------------------------------------------------------------------
void SiStripMonitorTrack::beginJob(edm::EventSetup const& es)
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
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe->save(outputFileName);
  }
}
// ------------ method called to produce the data  ------------
void SiStripMonitorTrack::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  edm::LogInfo("SiStripMonitorTrack") << "[SiStripMonitorTrack::analyse]  " << "Run " << e.id().run() << " Event " << e.id().event() << std::endl;
  runNb   = e.id().run();
  eventNb = e.id().event();
  
  //Get input
  e.getByLabel( ClusterInfo_src_, dsv_SiStripClusterInfo);
  e.getByLabel( Cluster_src_, dsv_SiStripCluster);    
  try{
    e.getByLabel(Track_src_, trackCollection);
  } catch ( cms::Exception& er ) {
    LogTrace("SiStripMonitorTrack")<<"caught std::exception "<<er.what()<<std::endl;
    tracksCollection_in_EventTree=false;
  } catch ( ... ) {
    LogTrace("SiStripMonitorTrack")<<" funny error " <<std::endl;
    tracksCollection_in_EventTree=false;
  }
  // TrackInfoAssociator Collection
  edm::InputTag TkiTag = conf_.getParameter<edm::InputTag>( "TrackInfoLabel" );
  e.getByLabel( TkiTag, TItkAssociatorCollection );
  vPSiStripCluster.clear();
  countOn=0;
  countOff=0;
  countAll=0;
    
  // get geometry to evaluate local angles
  edm::ESHandle<TrackerGeometry> estracker;
  es.get<TrackerDigiGeometryRecord>().get(estracker);
  _tracker=&(* estracker);
  
  //Perform track study
  if (tracksCollection_in_EventTree) trackStudy();
  
  //Perform Cluster Study (irrespectively to tracks)
  AllClusters();

  //Summary Counts
  if (countAll != countOn+countOff)
    edm::LogWarning("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"] Counts (on, off, all) do not match" << countOn << " " << countOff << " " << countAll; 

  std::map<TString, MonitorElement*>::iterator iME;
  std::map<TString, ModMEs>::iterator iModME ;
  for (int j=0;j<3;j++){
    int nTot=0;
    for (int i=0;i<4;i++){
      name=flags[j]+"_"+SubDet[i];      
      iModME  = ModMEsMap.find(name);
      if(iModME!=ModMEsMap.end()){
	fillME(iModME->second.nClusters, NClus[i][j]);
        fillTrend(iModME->second.nClustersTrend, NClus[i][j]);
      }
      nTot+=NClus[i][j];
      NClus[i][j]=0;
    }

    name=flags[j]+"_nClusters";
    iME  = MEMap.find(name);
    if(iME!=MEMap.end()) iME->second->Fill(nTot);
    iME  = MEMap.find(name+"Trend");
    if(iME!=MEMap.end()) fillTrend(iME->second,nTot);
  }  
}
//------------------------------------------------------------------------  
void SiStripMonitorTrack::book() {
  dbe->setCurrentFolder("Track/GlobalParameters");

  // get list of active detectors from SiStripDetCabling 
  std::vector<uint32_t> vdetId_;
  SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
  for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){ //loop on detector
    uint32_t detid = *detid_iter;
    if (detid < 1){
      edm::LogError("SiStripMonitorTrack") << "invalid detid " << detid<< std::endl;
      continue;
    }
    edm::LogInfo("SiStripMonitorTrack") << " Detid " << detid << " SubDet " << GetSubDetAndLayer(detid).first << " Layer " << GetSubDetAndLayer(detid).second << std::endl;   
    if (DetectedLayers.find(GetSubDetAndLayer(detid)) == DetectedLayers.end()){
      DetectedLayers[GetSubDetAndLayer(detid)]=true;
    }
  }//end loop on detector

  // book global histograms
  NumberOfTracks = bookME1D("TH1nTracks", "nTracks"); 
  NumberOfTracksTrend = bookMETrend("TH1nTracks", "nTracksTrend"); 
  NumberOfRecHitsPerTrack = bookME1D("TH1nRecHits", "nRecHitsPerTrack"); 
  NumberOfRecHitsPerTrackTrend = bookMETrend("TH1nRecHits", "nRecHitsPerTrackTrend");
  for (int j=0;j<3;j++){ // Loop on onTrack, offTrack, All
    name=flags[j]+"_nClusters";
    std::map<TString, MonitorElement*>::iterator iME  = MEMap.find(name);
    if(iME==MEMap.end())
      MEMap[name]=bookME1D("TH1nClusters", name.Data()); 
      MEMap[name+"Trend"]=bookMETrend("TH1nClusters", name.Data());
  } // Loop on onTrack, offTrack, All

  // book layer plots
  for (std::map<std::pair<std::string,uint32_t>,bool>::const_iterator iter=DetectedLayers.begin(); iter!=DetectedLayers.end();iter++){
    for (int j=0;j<3;j++){ // Loop on onTrack, offTrack, All
      name=flags[j]+"_"+iter->first.first;
      bookModMEs(name);
    }//end loop on onTrack,offTrack,all
    TString flag="All";
    char cApp[64];
    sprintf(cApp,"_Layer_%d",iter->first.second);
    bookModMEs(flag+"_"+iter->first.first+cApp);
  } 
}
//------------------------------------------------------------------------------------------
void SiStripMonitorTrack::trackStudy(){
  LogDebug("SiStripMonitorTrack") << "Start trackStudy";
  const reco::TrackCollection tC = *(trackCollection.product());
  int nTracks=tC.size();
  edm::LogInfo("SiStripMonitorTrack") << "Reconstructed "<< nTracks << " tracks" << std::endl ;
  NumberOfTracks->Fill(nTracks);
  fillTrend(NumberOfTracksTrend,nTracks);
  int i=0;
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){
    LogTrace("SiStripMonitorTrack")
      << "Track number "<< i+1 
      << "\n\tmomentum: " << track->momentum()
      << "\n\tPT: " << track->pt()
      << "\n\tvertex: " << track->vertex()
      << "\n\timpact parameter: " << track->d0()
      << "\n\tcharge: " << track->charge()
      << "\n\tnormalizedChi2: " << track->normalizedChi2() 
      <<"\n\tFrom EXTRA : "
      <<"\n\t\touter PT "<< track->outerPt()<<std::endl;

    // TrackInfo Map, extract TrackInfo for this track
    reco::TrackRef trackref = reco::TrackRef(trackCollection, i);
    reco::TrackInfoRef trackinforef=(*TItkAssociatorCollection.product())[trackref];
    std::vector<std::pair<const TrackingRecHit*, float> > hitangle;
    hitangle = SeparateHits(trackinforef);
    i++;
    // try and access Hits
    int recHitsSize=track->recHitsSize();
    edm::LogInfo("SiStripMonitorTrack") <<"\t\tNumber of RecHits "<<recHitsSize<<std::endl;
    NumberOfRecHitsPerTrack->Fill(recHitsSize);
    fillTrend(NumberOfRecHitsPerTrackTrend,recHitsSize);
    // Loop directly on the vector
    // We are using clusters now, so no matched hits
    std::vector<std::pair<const TrackingRecHit*, float> >::const_iterator tkangle_iter;
    for ( tkangle_iter = hitangle.begin(); tkangle_iter != hitangle.end(); ++tkangle_iter ) {
      const TrackingRecHit* trh = tkangle_iter->first;
      const uint32_t& detid = trh->geographicalId().rawId();
      if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end())
	continue;
      if (trh->isValid()){
	LogTrace("SiStripMonitorTrack")
	  <<"\n\t\tRecHit on det "<<trh->geographicalId().rawId()
	  <<"\n\t\tRecHit in LP "<<trh->localPosition()
	  <<"\n\t\tRecHit track angle "<<tkangle_iter->second
	  <<"\n\t\tRecHit in GP "<<tkgeom->idToDet(trh->geographicalId())->surface().toGlobal(trh->localPosition()) <<std::endl;
	//Get SiStripCluster from SiStripRecHit
	const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(trh);
	if ( hit != NULL ){
	  LogTrace("SiStripMonitorTrack") << "GOOD hit" << std::endl;
	  const SiStripCluster* SiStripCluster_ = &*(hit->cluster());
	  const SiStripClusterInfo* SiStripClusterInfo_ = MatchClusterInfo(SiStripCluster_,detid);
	  if ( clusterInfos(SiStripClusterInfo_,detid,"OnTrack", tkangle_iter->second ) ) {
	    vPSiStripCluster.push_back(SiStripCluster_);
	    countOn++;
	  }
	}else{
	  LogTrace("SiStripMonitorTrack") << "NULL hit" << std::endl;
	}
      }else{
	LogTrace("SiStripMonitorTrack") <<"\t\t Invalid Hit On "<<detid<<std::endl;
      }
    }
  }
}

//------------------------------------------------------------------------

void SiStripMonitorTrack::AllClusters(){
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
      if ( clusterInfos(SiStripClusterInfo_, detid,"All") ){
	countAll++;
	LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"] ClusIter " << &*ClusIter << 
	  "\t " << std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter)-vPSiStripCluster.begin() << std::endl;
	if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	  if ( clusterInfos(SiStripClusterInfo_,detid,"OffTrack") )
	    countOff++;
	}
      }
    }       
  }
}
  
//------------------------------------------------------------------------

const SiStripClusterInfo* SiStripMonitorTrack::MatchClusterInfo(const SiStripCluster* cluster, const uint32_t& detid){
  LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
  edm::DetSetVector<SiStripClusterInfo>::const_iterator DSViter = dsv_SiStripClusterInfo->find(detid);
  edm::DetSet<SiStripClusterInfo>::const_iterator ClusIter = DSViter->data.begin();
  for(; ClusIter!=DSViter->data.end(); ClusIter++) {
    if ( 
	(ClusIter->firstStrip() == cluster->firstStrip())
	&&
	(ClusIter->stripAmplitudes().size() == cluster->amplitudes().size())
	)
      return &(*ClusIter);
  }
  edm::LogError("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]\n\t" << "Matching of SiStripCluster and SiStripClusterInfo is failed for cluster on detid "<< detid << "\n\tReturning NULL pointer" <<std::endl;
  return 0;
}

//------------------------------------------------------------------------

bool SiStripMonitorTrack::clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,TString flag , float angle){
  LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
  if (cluster==0) return false;
  const  edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
  if  ( ps.getParameter<bool>("On") 
	&&
	( 
	 cluster->charge()/cluster->noise() < ps.getParameter<double>("minStoN") 
	 ||
	 cluster->charge()/cluster->noise() > ps.getParameter<double>("maxStoN") 
	 ||
	 cluster->width() < ps.getParameter<double>("minWidth") 
	 ||
	 cluster->width() > ps.getParameter<double>("maxWidth") 
	 )
	)
    return false;

  std::pair<std::string,uint32_t> SubDetAndLayer=GetSubDetAndLayer(detid);

  int SubDet_enum;
  if (SubDetAndLayer.first=="TIB")
    SubDet_enum=0;
  else if(SubDetAndLayer.first=="TOB")
    SubDet_enum=1;
  else if(SubDetAndLayer.first=="TID")
    SubDet_enum=2;
  else if (SubDetAndLayer.first=="TEC")
    SubDet_enum=3;
  else{
    edm::LogError("SiStripMonitorTrack")<< "[" <<__PRETTY_FUNCTION__ << "] invalid SubDet corresponding to detid " << detid<< std::endl;
    return false;
  }

  //Count Entries for single SubDet 
  int iflag;
  if      (flag=="onTrack")  iflag=0;
  else if (flag=="offTrack") iflag=1;
  else                       iflag=2;
  NClus[SubDet_enum][iflag]++;
  std::stringstream ss;
  const_cast<SiStripClusterInfo*>(cluster)->print(ss);
  LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]\n" << ss.str() << std::endl;

  //Cumulative Plots
  name=flag+"_"+SubDetAndLayer.first;
  fillModMEs(cluster,name,false);
  if(flag=="All"){
    // Layer Detail Plots
    char cApp[64];
    sprintf(cApp,"_Layer_%d",SubDetAndLayer.second);
    name=flag+"_"+SubDetAndLayer.first+cApp;
    fillModMEs(cluster,name,false);
  }      
  return true;
}
  
//--------------------------------------------------------------------------------
std::pair<std::string,uint32_t> SiStripMonitorTrack::GetSubDetAndLayer(const uint32_t& detid){
    
  std::string cSubDet;
  uint32_t layer=0;
  const StripGeomDetUnit* _StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
  switch(_StripGeomDetUnit->specificType().subDetector())
    {
    case GeomDetEnumerators::TIB:
      cSubDet="TIB";
      layer=TIBDetId(detid).layer();
      break;
    case GeomDetEnumerators::TOB:
      cSubDet="TOB";
      layer=TOBDetId(detid).layer();
      break;
    case GeomDetEnumerators::TID:
      cSubDet="TID";
      layer=TIDDetId(detid).wheel();
      break;
    case GeomDetEnumerators::TEC:
      cSubDet="TEC";
      layer=TECDetId(detid).wheel();
      break;
    default:
      edm::LogWarning("SiStripMonitorTrack") << "WARNING!!! this detid does not belong to tracker" << std::endl;
    }
  return std::make_pair(cSubDet,layer);
}
//--------------------------------------------------------------------------------
float SiStripMonitorTrack::clusEta(const SiStripClusterInfo* cluster){
  if (cluster->rawdigiAmplitudesL().size()!=0 ||  cluster->rawdigiAmplitudesR().size()!=0){
      float Ql=0;
      float Qr=0;
      float Qt=0;
      if (EtaAlgo_==1){
        Ql=cluster->chargeL();
        Qr=cluster->chargeR();
        for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesL().begin(); it !=cluster->rawdigiAmplitudesL().end() && it-cluster->rawdigiAmplitudesL().begin()<NeighStrips_; it ++)
	  { Ql += (*it);}
        for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesR().begin(); it !=cluster->rawdigiAmplitudesR().end() && it-cluster->rawdigiAmplitudesR().begin()<NeighStrips_; it ++)
	  { Qr += (*it);}
        Qt=Ql+Qr+cluster->maxCharge();
      }
      else{
        int Nstrip=cluster->stripAmplitudes().size();
        float pos=cluster->position()-0.5;
        for(int is=0;is<Nstrip && cluster->firstStrip()+is<=pos;is++)
  	Ql+=cluster->stripAmplitudes()[is];
        Qr=cluster->charge()-Ql;
        for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesL().begin(); it !=cluster->rawdigiAmplitudesL().end() && it-cluster->rawdigiAmplitudesL().begin()<NeighStrips_; it ++)
  	{ Ql += (*it);}
        for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesR().begin(); it !=cluster->rawdigiAmplitudesR().end() && it-cluster->rawdigiAmplitudesR().begin()<NeighStrips_; it ++)
  	{ Qr += (*it);}
        Qt=Ql+Qr;
      }
  return Ql/Qt;
  }
  return -999;
}
//--------------------------------------------------------------------------------
void SiStripMonitorTrack::bookModMEs(TString name){
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(name);
  if(iModME==ModMEsMap.end()){
    ModMEs theModMEs; 
    //Number of Cluster 
    theModMEs.nClusters=bookME1D("TH1nClusters", name+"_nClusters");
    theModMEs.nClustersTrend=bookMETrend("TH1nClusters", name+"_nClustersTrend");
    //Cluster Width
    theModMEs.ClusterWidth=bookME1D("TH1ClusterWidth", name+"_cWidth"); 
    theModMEs.ClusterWidthTrend=bookMETrend("TH1ClusterWidth", name+"_cWidthTrend");
    //Cluster Noise
    theModMEs.ClusterNoise=bookME1D("TH1ClusterNoise", name+"_cNoise"); 
    theModMEs.ClusterNoiseTrend=bookMETrend("TH1ClusterNoise", name+"_cNoiseTrend");
    //Cluster Signal
    theModMEs.ClusterSignal=bookME1D("TH1ClusterSignal", name+"_cSignal");
    theModMEs.ClusterSignalTrend=bookMETrend("TH1ClusterSignal", name+"_cSignalTrend");
    //Cluster StoN
    theModMEs.ClusterStoN=bookME1D("TH1ClusterStoN", name+"_cStoN");
    theModMEs.ClusterStoNTrend=bookMETrend("TH1ClusterStoN", name+"_cStoNTrend");
    //Cluster Position
    theModMEs.ClusterPos=bookME1D("TH1ClusterPos", name+"_cPos");  
    //Cluster PGV
    theModMEs.ClusterPGV=bookMEProfile("TProfileClusterPGV", name+"_cPGV"); 
    //Cluster Charge Division (only for study on Raw Data Runs)
    //theModMEs.ClusterEta=bookME("TH1ClusterEta", name+"cEta");
    ModMEsMap[name]=theModMEs;
  }
}

//--------------------------------------------------------------------------------

MonitorElement* SiStripMonitorTrack::bookME1D(const char* ParameterSetLabel, const char* HistoName){
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return dbe->book1D(HistoName,HistoName,
		         Parameters.getParameter<int32_t>("Nbinx"),
		         Parameters.getParameter<double>("xmin"),
		         Parameters.getParameter<double>("xmax")
		        );
}

MonitorElement* SiStripMonitorTrack::bookME2D(const char* ParameterSetLabel, const char* HistoName){
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

MonitorElement* SiStripMonitorTrack::bookME3D(const char* ParameterSetLabel, const char* HistoName){
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

MonitorElement* SiStripMonitorTrack::bookMEProfile(const char* ParameterSetLabel, const char* HistoName){
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

MonitorElement* SiStripMonitorTrack::bookMETrend(const char* ParameterSetLabel, const char* HistoName){
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

//--------------------------------------------------------------------------------

void SiStripMonitorTrack::fillTrend(MonitorElement* me ,float value){
  if(!me) return;
  //check the origin and check options
  int option = conf_.getParameter<edm::ParameterSet>("Trending").getParameter<int32_t>("UpdateMode");
  if(firstEvent==-1) firstEvent = eventNb;
  int CurrentStep = atoi(me->getAxisTitle(1).c_str()+8);
  int firstEventUsed = firstEvent;
  int presentOverflow = (int)me->getBinEntries(me->getNbinsX()+1);
  if(option==2) firstEventUsed += CurrentStep * me->getBinEntries(me->getNbinsX()+1);
  else if(option==3) firstEventUsed += CurrentStep * me->getBinEntries(me->getNbinsX()+1) * me->getNbinsX();
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
       float entries = 0;
       float content = 0.;
       float error = 0.;
       int bin = 1;
       int totEntries = me->getEntries();
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

void SiStripMonitorTrack::fillModMEs(const SiStripClusterInfo* cluster,TString basename,bool widthFlag){
  for (int iw=0;iw<5;iw++){
    float cwidth=cluster->width();
    if ( iw==0 || (iw==4 && cwidth>3) || ( iw>0 && iw<4 && cwidth==iw) ){     
      TString name=basename+width_flags[iw];
      std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(name);
      if(iModME!=ModMEsMap.end()){
        fillME(iModME->second.ClusterStoN  ,cluster->charge()/cluster->noise());
        fillTrend(iModME->second.ClusterStoNTrend,cluster->charge()/cluster->noise());
        fillME(iModME->second.ClusterSignal,cluster->charge());
        fillTrend(iModME->second.ClusterSignalTrend,cluster->charge());
        fillME(iModME->second.ClusterNoise ,cluster->noise());
        fillTrend(iModME->second.ClusterNoiseTrend,cluster->noise());
        fillME(iModME->second.ClusterWidth ,cluster->width());
        fillTrend(iModME->second.ClusterWidthTrend,cluster->width());
        fillME(iModME->second.ClusterPos   ,cluster->position());
        //fill the PGV histo
        float PGVmax =  cluster->maxCharge();
        int PGVposCounter = cluster->firstStrip() - cluster->rawdigiAmplitudesL().size() - cluster->maxPos();
        for (int i= int(conf_.getParameter<edm::ParameterSet>("TProfileClusterPGV").getParameter<double>("xmin"));i<PGVposCounter;++i)
          fillME(iModME->second.ClusterPGV, i,0.);
        for (std::vector<short>::const_iterator it=cluster->rawdigiAmplitudesL().begin();it<cluster->rawdigiAmplitudesL().end();++it) {
          fillME(iModME->second.ClusterPGV, PGVposCounter++,(*it)/PGVmax);
          LogTrace("PGV") << "debugging PGValgo: " << PGVposCounter-1 << " : " << (*it)/PGVmax << std::endl;
        }
        for (std::vector<uint16_t>::const_iterator it=cluster->stripAmplitudes().begin();it<cluster->stripAmplitudes().end();++it) {
          fillME(iModME->second.ClusterPGV, PGVposCounter++,(*it)/PGVmax);
          LogTrace("PGV") << "debugging PGValgo: " << PGVposCounter-1 << " : " << (*it)/PGVmax << std::endl;
        }
        for (std::vector<short>::const_iterator it=cluster->rawdigiAmplitudesR().begin();it<cluster->rawdigiAmplitudesR().end();++it) {
          fillME(iModME->second.ClusterPGV, PGVposCounter++,(*it)/PGVmax);
          LogTrace("PGV") << "debugging PGValgo: " << PGVposCounter-1 << " : " << (*it)/PGVmax << std::endl;
        }
        for (int i= PGVposCounter;i<int(conf_.getParameter<edm::ParameterSet>("TProfileClusterPGV").getParameter<double>("xmax"));++i)
          fillME(iModME->second.ClusterPGV, i,0.);
        //end fill the PGV histo
      }
    }
    if (!widthFlag)
      break;
  }
}

//--------------------------------------------------------------------------------
// Method to separate matched rechits in single clusters and to take
// the cluster from projected rechits and evaluate the track angle 

std::vector<std::pair<const TrackingRecHit*,float> > SiStripMonitorTrack::SeparateHits(reco::TrackInfoRef & trackinforef) {
  std::vector<std::pair<const TrackingRecHit*,float> >hitangleassociation;
  for(_tkinfoiter=trackinforef->trajStateMap().begin();_tkinfoiter!=trackinforef->trajStateMap().end();++_tkinfoiter) {
    const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(&(*(_tkinfoiter->first)));
    const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(_tkinfoiter->first)));
    const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(&(*(_tkinfoiter->first)));
    LocalVector trackdirection=(_tkinfoiter->second.stateOnDet().parameters()).momentum();
    if (phit) {
      //phit = POINTER TO THE PROJECTED RECHIT
      hit=&(phit->originalHit());
      std::cout << "ProjectedHit found" << std::endl;
    }
    if(matchedhit){//if matched hit...
      GluedGeomDet * gdet=(GluedGeomDet *)_tracker->idToDet(matchedhit->geographicalId());
      GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
      std::cout<<"Track direction trasformed in global direction"<<std::endl;
      //cluster and trackdirection on mono det
      // THIS THE POINTER TO THE MONO HIT OF A MATCHED HIT 
      const SiStripRecHit2D *monohit=matchedhit->monoHit();
      const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > monocluster=monohit->cluster();
      const GeomDetUnit * monodet=gdet->monoDet();
      LocalVector monotkdir=monodet->toLocal(gtrkdir);
      if(monotkdir.z()!=0){
	// THE LOCAL ANGLE (MONO)
	float angle = atan2(monotkdir.x(), monotkdir.z())*180/TMath::Pi();
	hitangleassociation.push_back(std::make_pair(monohit, angle)); 
	oXZHitAngle.push_back( std::make_pair( monohit, atan2( monotkdir.x(), monotkdir.z())));
	oYZHitAngle.push_back( std::make_pair( monohit, atan2( monotkdir.y(), monotkdir.z())));
	oLocalDir.push_back( std::make_pair( monohit, monotkdir));
	oGlobalDir.push_back( std::make_pair( monohit, gtrkdir));
	//cluster and trackdirection on stereo det
	// THIS THE POINTER TO THE STEREO HIT OF A MATCHED HIT 
	const SiStripRecHit2D *stereohit=matchedhit->stereoHit();
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > stereocluster=stereohit->cluster();
	const GeomDetUnit * stereodet=gdet->stereoDet(); 
	LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	if(stereotkdir.z()!=0){
	  // THE LOCAL ANGLE (STEREO)
	  float angle = atan2(stereotkdir.x(), stereotkdir.z())*180/TMath::Pi();
	  hitangleassociation.push_back(std::make_pair(stereohit, angle)); 
	  oXZHitAngle.push_back( std::make_pair( stereohit, atan2( stereotkdir.x(), stereotkdir.z())));
	  oYZHitAngle.push_back( std::make_pair( stereohit, atan2( stereotkdir.y(), stereotkdir.z())));
	  oLocalDir.push_back( std::make_pair( stereohit, stereotkdir));
	  oGlobalDir.push_back( std::make_pair( stereohit, gtrkdir));
	}
      }
    }
    else if(hit) {
      //  hit= POINTER TO THE RECHIT
      const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
      GeomDet * gdet=(GeomDet *)_tracker->idToDet(hit->geographicalId());
      if(trackdirection.z()!=0){
	// THE LOCAL ANGLE (STEREO)
	float angle = atan2(trackdirection.x(), trackdirection.z())*180/TMath::Pi();
	hitangleassociation.push_back(std::make_pair(hit, angle)); 
	oXZHitAngle.push_back( std::make_pair( hit, atan2( trackdirection.x(), trackdirection.z())));
	oYZHitAngle.push_back( std::make_pair( hit, atan2( trackdirection.y(), trackdirection.z())));
	oLocalDir.push_back( std::make_pair( hit, trackdirection));
	GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
	oGlobalDir.push_back( std::make_pair( hit, gtrkdir));
      }
    }
    else {
      std::cout << "not matched, mono or projected rechit" << std::endl;
    }
  } // end loop on rechits
  return (hitangleassociation);
}
