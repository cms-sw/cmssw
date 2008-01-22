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

#include "DQM/SiStripMonitorTrack/interface/SiStripMonitorTrack.h"

#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "TMath.h"

static const uint16_t _NUM_SISTRIP_SUBDET_ = 4;
static TString SubDet[_NUM_SISTRIP_SUBDET_]={"TIB","TID","TOB","TEC"};
static std::string flags[3] = {"OnTrack","OffTrack","All"};

SiStripMonitorTrack::SiStripMonitorTrack(const edm::ParameterSet& conf):
  dbe(edm::Service<DaqMonitorBEInterface>().operator->()),
  conf_(conf),
  Track_src_( conf.getParameter<edm::InputTag>( "Track_src" ) ),
  ClusterInfo_src_( conf.getParameter<edm::InputTag>( "ClusterInfo_src" ) ),
  Cluster_src_( conf.getParameter<edm::InputTag>( "Cluster_src" ) ),
   folder_organizer(), tracksCollection_in_EventTree(true),
  firstEvent(-1)
{
  for(int i=0;i<4;++i) for(int j=0;j<3;++j) NClus[i][j]=0;
}

//------------------------------------------------------------------------
SiStripMonitorTrack::~SiStripMonitorTrack() { }

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
  if(conf_.getParameter<bool>("OutputMEsInRootFile"))
    dbe->save(conf_.getParameter<std::string>("OutputFileName"));
}

// ------------ method called to produce the data  ------------
void SiStripMonitorTrack::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  //initialization of global quantities
  edm::LogInfo("SiStripMonitorTrack") << "[SiStripMonitorTrack::analyse]  " << "Run " << e.id().run() << " Event " << e.id().event() << std::endl;
  runNb   = e.id().run();
  eventNb = e.id().event();
  vPSiStripCluster.clear();
  countOn=0;
  countOff=0;
  countAll=0;
  
  //cluster input
  e.getByLabel( ClusterInfo_src_, dsv_SiStripClusterInfo);
  e.getByLabel( Cluster_src_,     dsv_SiStripCluster);    
  e.getByLabel(Track_src_, trackCollection);
  if (trackCollection.isValid()){
    LogTrace("SiStripMonitorTrack")<<" Track Collection is not valid !!" <<std::endl;
    tracksCollection_in_EventTree=false;
  }
  
  // track input
  edm::InputTag TkiTag = conf_.getParameter<edm::InputTag>( "TrackInfoLabel" );
  e.getByLabel( TkiTag, TItkAssociatorCollection );
    
  //Perform track study
  if (tracksCollection_in_EventTree) trackStudy();
  
  //Perform Cluster Study (irrespectively to tracks)
  AllClusters();

  //Summary Counts of clusters
  if (countAll != countOn+countOff)
    edm::LogWarning("SiStripMonitorTrack") << "Counts (on, off, all) do not match" << countOn << " " << countOff << " " << countAll; 
  std::map<TString, MonitorElement*>::iterator iME;
  std::map<TString, ModMEs>::iterator          iModME ;
  for (int j=0;j<3;++j){ // loop over ontrack, offtrack, all
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
  
    name=flags[j]+"_nClusters";
    iME = MEMap.find(name);
    if(iME!=MEMap.end()) iME->second->Fill(nTot);
    iME = MEMap.find(name+"Trend");
    if(iME!=MEMap.end()) fillTrend(iME->second,nTot);

  } // loop over ontrack, offtrack, all
  
}

//------------------------------------------------------------------------  
void SiStripMonitorTrack::book() 
{
  // set the DQM directory
  dbe->setCurrentFolder("Track/GlobalParameters");

  // book global histograms
  NumberOfTracks               = bookME1D("TH1nTracks",     "nTracks"); 
  NumberOfTracksTrend          = bookMETrend("TH1nTracks",  "nTracksTrend"); 
  NumberOfRecHitsPerTrack      = bookME1D("TH1nRecHits",    "nRecHitsPerTrack"); 
  NumberOfRecHitsPerTrackTrend = bookMETrend("TH1nRecHits", "nRecHitsPerTrackTrend");
  //LocalAngle                   = bookME1D("TH1localAngle",    "localAngle"); 

  for (int j=0;j<3;j++) { // Loop on onTrack, offTrack, All
    name=flags[j]+"_nClusters";
    if(MEMap.find(name)==MEMap.end()) {
      MEMap[name]=bookME1D("TH1nClusters", name.Data()); 
      name+="Trend";
      MEMap[name]=bookMETrend("TH1nClusters", name.Data());
    }
  } // Loop on onTrack, offTrack, All

  //loop on active detector ids : build "DetectedLayers"
  std::vector<uint32_t> vdetId_;
  SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
  for (std::vector<uint32_t>::const_iterator detid=vdetId_.begin();detid!=vdetId_.end();++detid) {
    if (*detid < 1) {
      edm::LogError("SiStripMonitorTrack") << "invalid detid " << (*detid);
      continue;
    }
    edm::LogInfo("SiStripMonitorTrack") << " Detid " << (*detid) 
                                        << " SubDet " << GetSubDetAndLayer(*detid).first 
                                        << " Layer "  << GetSubDetAndLayer(*detid).second;
    DetectedLayers[GetSubDetAndLayer(*detid)] |= (DetectedLayers.find(GetSubDetAndLayer(*detid)) == DetectedLayers.end());
  }//end loop on detector

  // create SiStripFolderOrganizer
  //SiStripFolderOrganizer folder_organizer;

  // book layer plots
  for (std::map<std::pair<std::string,int32_t>,bool>::const_iterator iter=DetectedLayers.begin(); iter!=DetectedLayers.end();iter++){
    for (int j=0;j<3;j++){ // Loop on onTrack, offTrack, All
      folder_organizer.setDetectorFolder(0);
      dbe->cd(iter->first.first);
      name=flags[j]+"_in_"+iter->first.first; 
      bookTrendMEs(name); //for subdets
    }//end loop on onTrack,offTrack,all
    TString flag="All";
    char cApp[64];
    sprintf(cApp,"_Layer_%d",iter->first.second); 
    folder_organizer.setDetectorFolder(0);
    dbe->cd(iter->first.first);
    if(iter->first.first=="TOB" || iter->first.first=="TIB") { 
      char layer[16]; sprintf(layer,"layer_%d",iter->first.second); dbe->cd(layer); 
    } else { 
      char layer[64]; sprintf(layer,"side_%d/wheel_%d",iter->first.second<0 ? 1 : 2,abs(iter->first.second)); dbe->cd(layer); 
    } 
    bookTrendMEs(flag+"_in_"+iter->first.first+cApp); // for layers
  } 

  // book module plots
  for (std::vector<uint32_t>::const_iterator detid=vdetId_.begin();detid!=vdetId_.end();++detid) {
    // set appropriate folder using SiStripFolderOrganizer
    folder_organizer.setDetectorFolder(*detid);
    bookModMEs("det",*detid);
  }
}

//--------------------------------------------------------------------------------
void SiStripMonitorTrack::bookModMEs(TString name, uint32_t id)
{
  SiStripHistoId hidmanager;
  std::string hid = hidmanager.createHistoId("",name.Data(),id);
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(TString(hid));
  if(iModME==ModMEsMap.end()){
    ModMEs theModMEs; 
    //Number of Cluster 
    theModMEs.nClusters=bookME1D("TH1nClusters", hidmanager.createHistoId("nClusters",name.Data(),id).c_str());
    dbe->tag(theModMEs.nClusters,id); 
    //Cluster Width
    theModMEs.ClusterWidth=bookME1D("TH1ClusterWidth", hidmanager.createHistoId("cWidth",name.Data(),id).c_str()); 
    dbe->tag(theModMEs.ClusterWidth,id); 
    //Cluster Noise
    theModMEs.ClusterNoise=bookME1D("TH1ClusterNoise", hidmanager.createHistoId("cNoise",name.Data(),id).c_str()); 
    dbe->tag(theModMEs.ClusterNoise,id); 
    //Cluster Signal
    theModMEs.ClusterSignal=bookME1D("TH1ClusterSignal", hidmanager.createHistoId("cSignal",name.Data(),id).c_str());
    dbe->tag(theModMEs.ClusterSignal,id); 
    //Cluster StoN
    theModMEs.ClusterStoN=bookME1D("TH1ClusterStoN", hidmanager.createHistoId("cStoN",name.Data(),id).c_str());
    dbe->tag(theModMEs.ClusterStoN,id); 
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

void SiStripMonitorTrack::bookTrendMEs(TString name)
{
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(name);
  char completeName[1024];
  if(iModME==ModMEsMap.end()){
    ModMEs theModMEs; 
    //Number of Cluster 
    sprintf(completeName,"Trend_nClusters_%s",name.Data());
    theModMEs.nClustersTrend=bookMETrend("TH1nClusters", completeName);
    //Cluster Width
    sprintf(completeName,"Trend_cWidth_%s",name.Data());
    theModMEs.ClusterWidthTrend=bookMETrend("TH1ClusterWidth", completeName);
    //Cluster Noise
    sprintf(completeName,"Trend_cNoise_%s",name.Data());
    theModMEs.ClusterNoiseTrend=bookMETrend("TH1ClusterNoise", completeName);
    //Cluster Signal
    sprintf(completeName,"Trend_cSignal_%s",name.Data());
    theModMEs.ClusterSignalTrend=bookMETrend("TH1ClusterSignal", completeName);
    //Cluster StoN
    sprintf(completeName,"Trend_cStoN_%s",name.Data());
    theModMEs.ClusterStoNTrend=bookMETrend("TH1ClusterStoN", completeName);
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
  int nTracks=tC.size();
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
    NumberOfRecHitsPerTrack->Fill(recHitsSize);
    fillTrend(NumberOfRecHitsPerTrackTrend,recHitsSize);
    std::vector<std::pair<const TrackingRecHit*, float> >::const_iterator tkangle_iter;
    for ( tkangle_iter = hitangle.begin(); tkangle_iter != hitangle.end(); ++tkangle_iter ) {
      const TrackingRecHit* trh = tkangle_iter->first;
      const uint32_t& detid = trh->geographicalId().rawId();
      if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()) continue;
      if (trh->isValid()) {
	LogTrace("SiStripMonitorTrack")
	  <<"\n\t\tRecHit on det "<<trh->geographicalId().rawId()
	  <<"\n\t\tRecHit in LP "<<trh->localPosition()
	  <<"\n\t\tRecHit track angle "<<tkangle_iter->second
	  <<"\n\t\tRecHit in GP "<<tkgeom->idToDet(trh->geographicalId())->surface().toGlobal(trh->localPosition()) <<std::endl;
	const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(trh);

	if (hit){
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
      if ( clusterInfos(SiStripClusterInfo_, detid,"All") ) {
	countAll++;
	LogDebug("SiStripMonitorTrack") << "ClusIter " << &*ClusIter << "\t " 
	                                << std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter)-vPSiStripCluster.begin();
	if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	  if ( clusterInfos(SiStripClusterInfo_,detid,"OffTrack") ) {
	    countOff++;
	  }
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
bool SiStripMonitorTrack::clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,TString flag , float angle)
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
  
  //Count Entries for single SubDet -> used later on for global histograms
  int SubDet_enum = StripSubdetector(detid).subdetId()-3;
  int iflag;
  if      (flag=="OnTrack")  iflag=0;
  else if (flag=="OffTrack") iflag=1;
  else                       iflag=2;
  NClus[SubDet_enum][iflag]++;
  std::stringstream ss;
  const_cast<SiStripClusterInfo*>(cluster)->print(ss);
  LogTrace("SiStripMonitorTrack") << "\n["<<__PRETTY_FUNCTION__<<"]\n" << ss.str() << std::endl;
  
  //TODO: correct for the local angle
  //if(flag == "OnTrack") {
  //LocalAngle->Fill(angle);
  //}
  //Cumulative Plots
  std::pair<std::string,int32_t> SubDetAndLayer = GetSubDetAndLayer(detid);
  name=flag+"_in_"+SubDetAndLayer.first;
  fillTrendMEs(cluster,name);  // subdet plots
  if(flag=="All"){
    char cApp[64];
    sprintf(cApp,"_Layer_%d",SubDetAndLayer.second);
    name=flag+"_in_"+SubDetAndLayer.first+cApp;
    fillTrendMEs(cluster,name); // layer plots
  }

  SiStripHistoId hidmanager;
  name = hidmanager.createHistoId("","det",detid);
  fillModMEs(cluster,name);
  
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
void SiStripMonitorTrack::fillModMEs(const SiStripClusterInfo* cluster,TString name)
{
  std::map<TString, ModMEs>::iterator iModME  = ModMEsMap.find(name);
  if(iModME!=ModMEsMap.end()){
    fillME(iModME->second.ClusterStoN  ,cluster->charge()/cluster->noise());
    fillME(iModME->second.ClusterSignal,cluster->charge());
    fillME(iModME->second.ClusterNoise ,cluster->noise());
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

void SiStripMonitorTrack::fillTrendMEs(const SiStripClusterInfo* cluster,TString name)
{
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
  }
}
//--------------------------------------------------------------------------------
// Method to separate matched rechits in single clusters and to take
// the cluster from projected rechits and evaluate the track angle 
std::vector<std::pair<const TrackingRecHit*,float> > SiStripMonitorTrack::SeparateHits(reco::TrackInfoRef & trackinforef) 
{
  std::vector<std::pair<const TrackingRecHit*,float> >hitangleassociation;
  reco::TrackInfo::TrajectoryInfo::const_iterator _tkinfoiter;
  for(_tkinfoiter=trackinforef->trajStateMap().begin();_tkinfoiter!=trackinforef->trajStateMap().end();++_tkinfoiter) {
    const ProjectedSiStripRecHit2D* phit       = dynamic_cast<const ProjectedSiStripRecHit2D*>(&(*(_tkinfoiter->first)));
    const SiStripMatchedRecHit2D*   matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(_tkinfoiter->first)));
    const SiStripRecHit2D*          hit        = dynamic_cast<const SiStripRecHit2D*>(&(*(_tkinfoiter->first)));
    LocalVector trackdirection = (_tkinfoiter->second.stateOnDet(reco::Updated)->parameters()).momentum();
    if(phit) {
      // obtain the hit from the projected hit... association common to single monohit part
      hit=&(phit->originalHit());
      LogDebug("SiStripMonitorTrack") << "ProjectedHit found";
    }
    if(matchedhit) {
      GluedGeomDet* gdet   = (GluedGeomDet *)tkgeom->idToDet(matchedhit->geographicalId());
      GlobalVector gtrkdir = gdet->toGlobal(trackdirection);
      LogDebug("SiStripMonitorTrack") << "Track direction transformed in global direction";
      // asssociation for the mono hit
      const SiStripRecHit2D* monohit = matchedhit->monoHit();
      LocalVector monotkdir = gdet->monoDet()->toLocal(gtrkdir);
      if(monotkdir.z()!=0){
	float angle = atan2(monotkdir.x(), monotkdir.z())*180/TMath::Pi();
	hitangleassociation.push_back(std::make_pair(monohit, angle)); 
      }
      // asssociation for the stereo hit
      const SiStripRecHit2D* stereohit = matchedhit->stereoHit();
      LocalVector stereotkdir = gdet->stereoDet()->toLocal(gtrkdir);
      if(stereotkdir.z()!=0){
        float angle = atan2(stereotkdir.x(), stereotkdir.z())*180/TMath::Pi();
        hitangleassociation.push_back(std::make_pair(stereohit, angle)); 
      }
    }
    else if(hit) {
      // asssociation for the single mono hit
      if(trackdirection.z()!=0){
	float angle = atan2(trackdirection.x(), trackdirection.z())*180/TMath::Pi();
	hitangleassociation.push_back(std::make_pair(hit, angle)); 
      }
    }
    else {
      LogDebug("SiStripMonitorTrack") << "not matched, mono or projected rechit";
    }
  } // end loop on rechits
  return (hitangleassociation);
}

DEFINE_FWK_MODULE(SiStripMonitorTrack);
