
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngleAlgorithm.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"

using namespace std;

#include <functional>

 
class DetIdLess 
  : public std::binary_function< const SiStripRecHit2D*,const SiStripRecHit2D*,bool> {
public:
  
  DetIdLess() {}
  
  bool operator()( const SiStripRecHit2D* a, const SiStripRecHit2D* b) const {
    return *(a->cluster())<*(b->cluster());
  }
};
  

  //Constructor

SiStripLorentzAngleAlgorithm::SiStripLorentzAngleAlgorithm(edm::ParameterSet const& conf) : 
  conf_(conf)
{
}

  //BeginJob

void SiStripLorentzAngleAlgorithm::init(const edm::EventSetup& c){

  hFile = new TFile (conf_.getParameter<std::string>("fileName").c_str(), "RECREATE" );
  
  edm::ESHandle<MagneticField> esmagfield;
  c.get<IdealMagneticFieldRecord>().get(esmagfield);
  magfield=&(*esmagfield);
  
  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker); 
  
  //Get Ids;
  
  const TrackerGeometry::DetIdContainer& Id = estracker->detIds();
  
  TrackerGeometry::DetIdContainer::const_iterator Iditer;
  
  for(Iditer=Id.begin();Iditer!=Id.end();Iditer++){ //loop on detids
    
    if((Iditer->subdetId() != int(PixelSubdetector::PixelBarrel)) && (Iditer->subdetId() != int(PixelSubdetector::PixelEndcap))){
      
      StripSubdetector subid(*Iditer);
      
      //Mono single sided detectors
      LocalPoint p;
      const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(subid);
      if(stripdet==0)continue;
      const StripTopology& topol=(StripTopology&)stripdet->topology();
      float thickness=stripdet->specificSurface().bounds().thickness();		
      TProfile * profile=new TProfile(makename(*Iditer,false,false).c_str(),makename(*Iditer,true,false).c_str(),30,-0.6,0.6);
      detparameters *param=new detparameters;
      histos[Iditer->rawId()] = profile;
      detmap[Iditer->rawId()] = param;
      param->thickness = thickness*10000;
      param->pitch = topol.localPitch(p)*10000;
      profile->GetXaxis()->SetTitle("tan(#theta_{t})");
      profile->GetYaxis()->SetTitle("Cluster size");
      int layer=0;
      if(subid.subdetId() == int (StripSubdetector::TIB)){
	TIBDetId TIBid=TIBDetId(subid.rawId());
	layer = TIBid.layer();
      }
      
      else if(subid.subdetId() == int (StripSubdetector::TID)){
	TIDDetId TIDid=TIDDetId(subid.rawId());
	layer = TIDid.ring();
      }
      
      else if(subid.subdetId() == int (StripSubdetector::TOB)){
	TOBDetId TOBid=TOBDetId(subid.rawId());
	layer = TOBid.layer();
      }
      
      else if(subid.subdetId() == int (StripSubdetector::TEC)){
	TECDetId TECid=TECDetId(subid.rawId());
	layer = TECid.ring();
      }
      if(summaryhisto.find(subid.subdetId()*10+layer)==(summaryhisto.end())){
	TProfile * summaryprofile=new TProfile(makename(*Iditer,false,true).c_str(),makename(*Iditer,true,true).c_str(),30,-0.6,0.6);
	detparameters *summaryparam=new detparameters;
	summaryhisto[subid.subdetId()*10+layer] = summaryprofile;
	summarydetmap[subid.subdetId()*10+layer] = summaryparam;
	summaryparam->thickness = thickness*10000;
	summaryparam->pitch = topol.localPitch(p)*10000;
	summaryprofile->GetXaxis()->SetTitle("tan(#theta_{t})");
	summaryprofile->GetYaxis()->SetTitle("Cluster size");
      }
    } 
  } 
  //Directory hierarchy  
  
  histograms = new TDirectory("Histograms", "Histograms", "");
  summary = new TDirectory("LorentzAngleSummary", "LorentzAngleSummary", "");

  TIB = histograms->mkdir("TIB");
  TOB = histograms->mkdir("TOB");
  TID = histograms->mkdir("TID");
  TEC = histograms->mkdir("TEC");
  //  }
  //Forward-Backward
  
  TIBfw = TIB->mkdir("TIB forward");
  TIDfw = TID->mkdir("TID forward");
  TOBfw = TOB->mkdir("TOB forward");
  TECfw = TEC->mkdir("TEC forward");
  
  TIBbw = TIB->mkdir("TIB backward");
  TIDbw = TID->mkdir("TID backward");
  TOBbw = TOB->mkdir("TOB backward");
  TECbw = TEC->mkdir("TEC backward"); 
  
  //TIB directories
  for(int i=0;i<4;i++){
    TIBfwl[i] = TIBfw->mkdir(Form("TIB forward layer %d", i+1));
    TIBbwl[i] = TIBbw->mkdir(Form("TIB backward layer %d",i+1));
  }
  //TID directories
    for(int i=0;i<3;i++){
    TIDfwr[i] = TIDfw->mkdir(Form("TID forward disk %d",i+1));
    TIDbwr[i] = TIDbw->mkdir(Form("TID backward disk %d",i+1)); 
  }  
  //TOB directories
  for(int i=0;i<6;i++){
    TOBfwl[i] = TOBfw->mkdir(Form("TOB forward layer %d", i+1));
    TOBbwl[i] = TOBbw->mkdir(Form("TOB backward layer %d",i+1));
  }
  //TEC directories
  for(int i=0;i<7;i++){
    TECfwr[i] = TECfw->mkdir(Form("TEC forward ring %d",i+1));
    TECbwr[i] = TECbw->mkdir(Form("TEC backward ring %d",i+1));
  }
  eventcounter = 0;
  trackcounter = 0;
  hitcounter = 0;
  runnr = 0;
  runcounter = 0;
  
  for(int m=0;m<1000;m++) runvector[m]=0;
  
} 

// Virtual destructor needed.

SiStripLorentzAngleAlgorithm::~SiStripLorentzAngleAlgorithm() {  
  detparmap::iterator detpariter;
  for( detpariter=detmap.begin(); detpariter!=detmap.end();++detpariter)delete detpariter->second;
  for( detpariter=summarydetmap.begin(); detpariter!=summarydetmap.end();++detpariter)delete detpariter->second;
  fitmap::iterator  fitpar;
  for( fitpar=summaryfits.begin(); fitpar!=summaryfits.end();++fitpar)delete fitpar->second;
  delete hFile;

}  

// Analyzer: Functions that gets called by framework every event

void SiStripLorentzAngleAlgorithm::run(const edm::Event& e, const edm::EventSetup& es)
{
  
  if(e.id().run() != runnr){
    runvector[runcounter]=e.id().run();
    runcounter++;
  }
  
  runnr       = e.id().run();
  eventnr     = e.id().event();
       
  cout<<"Run number = "<<runnr<<endl;
  cout<<"Event number = "<<eventnr<<endl;
  
  eventcounter++;
  
  using namespace edm;
  
  //Analysis of Trajectory-RecHits
        
  edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("Tracks");
  
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByLabel(TkTag,trackCollection);
  
  edm::Handle<TrackingRecHitCollection> trackerchitCollection;
  e.getByLabel(TkTag,trackerchitCollection);
  
  edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
  e.getByLabel(TkTag,TrajectoryCollection);
  
  const reco::TrackCollection *tracks=trackCollection.product();
 
  std::map<const SiStripRecHit2D*,std::pair<float,float>,DetIdLess> hitangleassociation;
  trackcollsize = 0;
  trajsize = 0;
  
  trackcollsize=tracks->size();
  trajsize=TrajectoryCollection->size();
  
  edm::LogInfo("SiStripLorentzAngleAlgorithm::analyze") <<" Number of tracks in event = "<<trackcollsize<<"\n";
  edm::LogInfo("SiStripLorentzAngleAlgorithm::analyze") <<" Number of trajectories in event = "<<trajsize<<"\n";
  std::vector<Trajectory>::const_iterator theTraj;
  for(theTraj = TrajectoryCollection->begin(); theTraj!= TrajectoryCollection->end();theTraj++){
    
    std::vector<TrajectoryMeasurement> TMeas=theTraj->measurements();
    std::vector<TrajectoryMeasurement>::iterator itm;
    
    LogDebug("SiStripLorentzAngleAlgorithm::analyze")<<"Loop on rechit and TSOS";
    for (itm=TMeas.begin();itm!=TMeas.end();itm++){
      TrajectoryStateOnSurface tsos=itm->updatedState();
      const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
      const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>((*thit).hit());
      const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>((*thit).hit());
      const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>((*thit).hit());
      if(phit) hit=&(phit->originalHit());
      LocalVector trackdirection=tsos.localDirection();
      
      if(matchedhit){//if matched hit...
	
	GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(matchedhit->geographicalId());
	
	GlobalVector gtrkdir=gdet->toGlobal(trackdirection);	
	
	LogDebug("SiStripLorentzAngleAlgorithm::analyze") <<"Matched hits used";
	
	//cluster and trackdirection on mono det
	
	// THIS THE POINTER TO THE MONO HIT OF A MATCHED HIT 
	const SiStripRecHit2D *monohit=matchedhit->monoHit();
	
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > monocluster=monohit->cluster();
	const GeomDetUnit * monodet=gdet->monoDet();
	
	LocalVector monotkdir=monodet->toLocal(gtrkdir);
	//size=(monocluster->amplitudes()).size();
	if(monotkdir.z()!=0){
	  
	  // THE LOCAL ANGLE (MONO)
	  float tanangle = monotkdir.x()/monotkdir.z();
	  std::map<const SiStripRecHit2D *,std::pair<float,float>,DetIdLess>::iterator alreadystored=hitangleassociation.find(monohit);
	  if(alreadystored != hitangleassociation.end()){//decide which hit take
	    if(itm->estimate() <  alreadystored->second.second) hitangleassociation.insert(make_pair(monohit, std::make_pair(itm->estimate(),tanangle)));
	  }
	  else hitangleassociation.insert(make_pair(monohit, std::make_pair(itm->estimate(),tanangle))); 
	  
	  //cluster and trackdirection on stereo det
	  
	  // THIS THE POINTER TO THE STEREO HIT OF A MATCHED HIT 
	  const SiStripRecHit2D *stereohit=matchedhit->stereoHit();
	  const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > stereocluster=stereohit->cluster();
	  const GeomDetUnit * stereodet=gdet->stereoDet(); 
	  LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	  
	  if(stereotkdir.z()!=0){
	    
	    // THE LOCAL ANGLE (STEREO)
	    float tanangle = stereotkdir.x()/stereotkdir.z();
	    std::map<const SiStripRecHit2D *,std::pair<float,float>,DetIdLess>::iterator alreadystored=hitangleassociation.find(stereohit);
	    if(alreadystored != hitangleassociation.end()){//decide which hit take
	      if(itm->estimate() <  alreadystored->second.second) hitangleassociation.insert(make_pair(stereohit, std::make_pair(itm->estimate(),tanangle)));
	    }
	    else hitangleassociation.insert(std::make_pair(stereohit, std::make_pair(itm->estimate(),tanangle))); 		  
	  }
	}
      }
      else if(hit){
	//  hit= POINTER TO THE RECHIT
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
	
	if(trackdirection.z()!=0){
	  
	  // THE LOCAL ANGLE (STEREO)
	  float tanangle = trackdirection.x()/trackdirection.z();
	  std::map<const SiStripRecHit2D *,std::pair<float,float>, DetIdLess>::iterator alreadystored=hitangleassociation.find(hit);
	  if(alreadystored != hitangleassociation.end()){//decide which hit take
	    if(itm->estimate() <  alreadystored->second.second) hitangleassociation.insert(make_pair(hit, std::make_pair(itm->estimate(),tanangle)));
	  }
	  else hitangleassociation.insert(std::make_pair(hit,std::make_pair(itm->estimate(), tanangle) ) ); 
	}
      }
    }
  }
  
  std::map<const SiStripRecHit2D *,pair<float,float>,DetIdLess>::iterator hitsiter;
    
  for(hitsiter=hitangleassociation.begin();hitsiter!=hitangleassociation.end();hitsiter++){
    
    const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(hitsiter->first);
    const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();

    int size=(cluster->amplitudes()).size();
    
	
    StripSubdetector detid=(StripSubdetector)hit->geographicalId();
    //   
    // TrackLocalAngle=hitsiter->second.second;
	  
    float tangent = hitsiter->second.second;
	  
    //Sign and XZ plane projection correction applied in TrackLocalAngle (TIB|TOB layers)
      
    const GeomDet *geomdet = tracker->idToDet(hit->geographicalId());
    LocalPoint localp(0,0,0);
    const GlobalPoint globalp = (geomdet->surface()).toGlobal(localp);
    GlobalVector globalmagdir = magfield->inTesla(globalp);
    LocalVector localmagdir = (geomdet->surface()).toLocal(globalmagdir);
    float localmagfield = localmagdir.mag();
    
    if(localmagfield != 0.){
	    
      if((detid.subdetId() == int (StripSubdetector::TIB)) || (detid.subdetId() == int (StripSubdetector::TOB))){
	    
	LocalVector ylocal(0,1,0);
	    
	float normprojection = (localmagdir * ylocal)/(localmagfield);
            
	if(normprojection == 0.)LogDebug("SiStripLorentzAngleAlgorithm::analyze")<<"Error: TIB|TOB YBprojection = 0";
	    
	else{
	  float signprojcorrection = 1/normprojection;
	  tangent*=signprojcorrection;
	  //  TrackLocalAngle = atan(tangent)*180/TMath::Pi();
	}
      }
    }
	  
    float thickness = detmap[detid.rawId()]->thickness;
    float pitch = detmap[detid.rawId()]->pitch;
	  
    //    trackproj=(tangent*thickness)/pitch;
    
    //Filling histograms
    histomap::iterator thehisto=histos.find(detid.rawId());
    if(thehisto==histos.end())edm::LogError("SiStripLorentzAngleAlgorithm::analyze")<<"Error: the profile associated to"<<detid.rawId()<<"does not exist! ";
    else thehisto->second->Fill(tangent,size);

    //Summary histograms
    int layer;
    if(detid.subdetId() == int (StripSubdetector::TIB)){
      TIBDetId TIBid=TIBDetId(hit->geographicalId());
      layer = TIBid.layer();
    }

    else if(detid.subdetId() == int (StripSubdetector::TID)){
      TIDDetId TIDid=TIDDetId(hit->geographicalId());
      layer = TIDid.ring();
    }

    else if(detid.subdetId() == int (StripSubdetector::TOB)){
      TOBDetId TOBid=TOBDetId(hit->geographicalId());
      layer = TOBid.layer();
    }

    else if(detid.subdetId() == int (StripSubdetector::TEC)){
      TECDetId TECid=TECDetId(hit->geographicalId());
      layer = TECid.ring();
    }

    histomap::iterator thesummaryhisto=summaryhisto.find(detid.subdetId()*10+layer);
    if(thesummaryhisto==summaryhisto.end())edm::LogError("SiStripLorentzAngleAlgorithm::analyze")<<"Error: the profile associated to subdet "<<detid.subdetId()<<" layer "<<layer<<"does not exist! ";
    else thesummaryhisto->second->Fill(tangent,size);

  }
}

 
//Makename function
 
std::string  SiStripLorentzAngleAlgorithm::makename(const DetId & detid,bool description,bool summary){
  
  std::string name;
  std::string backward,forward;
  std::string internal,external;
  std::string mono,stereo;
  std::string negative,positive;
  std::string layerstring,wheelstring,stringstring,rodstring,ringstring,petalstring;
  if(description){
    name="Cluster width vs tan(track local angle) (";
    backward="backward, ";
    forward="forward, ";
    internal=", internal ";
    external=", external ";
    mono=", mono, ";
    stereo=", stereo, ";
    negative=", negative ";
    positive=", positive ";
    layerstring="Layer n.";
    stringstring="string n.";
    wheelstring="wheel n.";
    ringstring="ring n.";
    rodstring="rod n.";
    petalstring="petal n.";
  }
  else{
    name="";
    backward="bw";
    forward="fw";
    internal="int";
    external="ext";
    mono="mono";
    stereo="stereo";
    negative="neg";
    positive="pos";
    layerstring="L";
    stringstring="string";
    wheelstring="W";
    ringstring="ring";
    rodstring="rod";
    petalstring="petal";
  }
  stringstream idnum;
  stringstream layernum;
  stringstream wheelnum;
  stringstream stringnum;
  stringstream rodnum;
  stringstream ringnum;
  stringstream petalnum;
  
  idnum << detid.rawId();
  
  //TIB
  
  if(detid.subdetId() == int (StripSubdetector::TIB)){
    name+="TIB";
    if(description)name+=" ";
    TIBDetId TIBid=TIBDetId(detid.rawId());
    
    if(!summary){
      if(TIBid.string()[0] == 1) name+=backward;
      else if(TIBid.string()[0] == 2) name+=forward;
    }
    name+=layerstring;
    int layer = TIBid.layer();    
    layernum << layer;
    name+=layernum.str();
    if(summary) return name.c_str();
    
    if(TIBid.string()[1] == 1)name+=internal;
    else if(TIBid.string()[1] == 2)name+=external;
    
     name+=stringstring;
     int string = TIBid.string()[2];
     stringnum << string;
     name+=stringnum.str();
     
     if(TIBid.stereo() == 0)name+=mono;
     else if(TIBid.stereo() == 1)name+=stereo;    
  }
  
  //TID
  
  else if(detid.subdetId() == int (StripSubdetector::TID)){
    name+="TID";
    
    TIDDetId TIDid=TIDDetId(detid.rawId());
    
    if(!summary){
      if(TIDid.module()[0] == 1)name+=backward;
      else if(TIDid.module()[0] == 2)name+=forward;
      
      name+=wheelstring;
      int wheel = TIDid.wheel();    
      wheelnum << wheel;
      name+=wheelnum.str();
      
      if(TIDid.side() == 1)name+=negative;
      else if(TIDid.side() == 2)name+=positive;
    }
    name+=ringstring;
    int ring = TIDid.ring();
    ringnum << ring;
    name+=ringnum.str();
    if(summary) return name.c_str();
    
    if(TIDid.stereo() == 0) name+=mono;
    else if(TIDid.stereo() == 1) name+=stereo;    
  }
  
  //TOB
  
  else if(detid.subdetId() == int (StripSubdetector::TOB)){
    name="TOB";
    
    TOBDetId TOBid=TOBDetId(detid.rawId());
    
    if(!summary){
      if(TOBid.rod()[0] == 1)name+=backward;
      else if(TOBid.rod()[0] == 2)name+=forward;
    }
    
    name+=layerstring;
    int layer = TOBid.layer();
    layernum << layer;
    name+=layernum.str();
    if(summary) return name.c_str();
    
    name+=rodstring;
    int rod = TOBid.rod()[1];
    rodnum << rod;
    name+=rodnum.str();
    
    if(TOBid.stereo() == 0)name+=mono;
    else if(TOBid.stereo() == 1)name+=stereo;    
  }
  
  //TEC
  
  else if(detid.subdetId() == int (StripSubdetector::TEC)){
    name="TEC";
    
    TECDetId TECid=TECDetId(detid.rawId());
    
    if(!summary){
      if(TECid.petal()[0] == 1)name+=backward;
      else if(TECid.petal()[0] == 2)name+=forward;
      
      name+=wheelstring;
      int wheel = TECid.wheel();    
      wheelnum << wheel;
      name+=wheelnum.str();
      
      if(TECid.side() == 1)name+=negative;
      else if(TECid.side() == 2)name+=positive;
    }
    
    name+=ringstring;
    int ring = TECid.ring();
    ringnum << ring;
    name+=ringnum.str();
    if(summary) return name.c_str();
    
    name+=petalstring;
    int petal = TECid.petal()[1];
    petalnum << petal;
    name+=petalnum.str();
    
    if(TECid.stereo() == 0)name+=mono;
    else if(TECid.stereo() == 1)name+=stereo;    
  }
    
  if(!description)name+="_";
  else name+=" IdNumber = ";
  
  name+=idnum.str();
  
  if(description)name+=")";
  //  edm::LogInfo("makename")<<name.c_str();
  return name;
  
}
 
void SiStripLorentzAngleAlgorithm::fit(fitmap & fits){
  
  fitmap summaryfit;
  //Histograms fit
  TF1 *fitfunc=0;
  double ModuleRangeMin=conf_.getParameter<double>("ModuleRangeMin");
  double ModuleRangeMax=conf_.getParameter<double>("ModuleRangeMax");
  double TIBRangeMin=conf_.getParameter<double>("TIBRangeMin");
  double TIBRangeMax=conf_.getParameter<double>("TIBRangeMax");
  double TOBRangeMin=conf_.getParameter<double>("TOBRangeMin");
  double TOBRangeMax=conf_.getParameter<double>("TOBRangeMax");
  
  histomap::iterator hist_it;
  fitfunc= new TF1("fitfunc","([4]/[3])*[1]*(TMath::Abs(x-[0]))+[2]",-1,1);
  
  for(hist_it=histos.begin();hist_it!=histos.end(); hist_it++){
    if(hist_it->second->GetEntries()>100){
      float thickness=0,pitch=-1;
      detparmap::iterator detparit=detmap.find(hist_it->first);
      if(detparit!=detmap.end()){
	thickness = detparit->second->thickness;
	pitch = detparit->second->pitch;
      }
      
      fitfunc->SetParameter(0, 0);
      fitfunc->SetParameter(1, 0);
      fitfunc->SetParameter(2, 1);
      fitfunc->FixParameter(3, pitch);
      fitfunc->FixParameter(4, thickness);
      edm::LogInfo("test")<<hist_it->second->GetEntries();
      int fitresult=-1;      
      fitresult=hist_it->second->Fit(fitfunc,"E","",ModuleRangeMin, ModuleRangeMax);

      histofit *fit= new histofit;
      fits[hist_it->first] =fit;
      
      fit->chi2 = fitfunc->GetChisquare();
      fit->ndf  = fitfunc->GetNDF();
      fit->p0   = fitfunc->GetParameter(0);
      fit->p1   = fitfunc->GetParameter(1);
      fit->p2   = fitfunc->GetParameter(2);
      fit->errp0   = fitfunc->GetParError(0);
      fit->errp1   = fitfunc->GetParError(1);
      fit->errp2   = fitfunc->GetParError(2);
    }
  }
  
  histomap::iterator summaryhist_it;
  
  for(summaryhist_it=summaryhisto.begin();summaryhist_it!=summaryhisto.end(); summaryhist_it++){
    if(summaryhist_it->second->GetEntries()>100){
      float thickness=0,pitch=-1;
      detparmap::iterator detparit=summarydetmap.find(summaryhist_it->first);
      if(detparit!=summarydetmap.end()){
	thickness = detparit->second->thickness;
	pitch = detparit->second->pitch;
      }
      
      fitfunc->SetParameter(0, 0);
      fitfunc->SetParameter(1, 0);
      fitfunc->SetParameter(2, 1);
      fitfunc->FixParameter(3, pitch);
      fitfunc->FixParameter(4, thickness);
      int fitresult=-1;
      if ((summaryhist_it->first)/10==int (StripSubdetector::TIB)||(summaryhist_it->first)/10==int (StripSubdetector::TID))
	fitresult=summaryhist_it->second->Fit(fitfunc,"E","",TIBRangeMin, TIBRangeMax);
      else if ((summaryhist_it->first)/10==int (StripSubdetector::TOB)||(summaryhist_it->first)/10==int (StripSubdetector::TEC))
	fitresult=summaryhist_it->second->Fit(fitfunc,"E","",TOBRangeMin, TOBRangeMax);
      //if(fitresult==0){
	histofit * summaryfit=new histofit;
	summaryfits[summaryhist_it->first] = summaryfit;
	
	summaryfit->chi2 = fitfunc->GetChisquare();
	summaryfit->ndf  = fitfunc->GetNDF();
	summaryfit->p0   = fitfunc->GetParameter(0);
	summaryfit->p1   = fitfunc->GetParameter(1);
	summaryfit->p2   = fitfunc->GetParameter(2);
	summaryfit->errp0   = fitfunc->GetParError(0);
	summaryfit->errp1   = fitfunc->GetParError(1);
	summaryfit->errp2   = fitfunc->GetParError(2);
	// }
    }
  }
  delete fitfunc;
  
  //File with fit parameters  
  
  std::string fitName=conf_.getParameter<std::string>("fitName");
  fitName+=".txt";
  
  ofstream fit;
  fit.open(fitName.c_str());
  
  fit<<">>> ANALYZED RUNS = ";
  for(int n=0;n!=runcounter;n++){
  fit<<runvector[n]<<", ";}
  fit<<endl;
  
  fit<<">>> TOTAL EVENTS = "<<eventcounter<<endl;
  //  fit<<">>> NUMBER OF TRACKS = "<<trackcounter<<endl<<endl;
  //fit<<">>> NUMBER OF RECHITS = "<<hitcounter<<endl<<endl;
  
  fit<<">>> NUMBER OF DETECTOR HISTOGRAMS = "<<histos.size()<<endl;
     
  std::string subdetector;
  for(summaryhist_it=summaryhisto.begin();summaryhist_it!=summaryhisto.end(); summaryhist_it++){
    if ((summaryhist_it->first)/10==int (StripSubdetector::TIB))subdetector="TIB";
    else if ((summaryhist_it->first)/10==int (StripSubdetector::TID))subdetector="TID";
    else if ((summaryhist_it->first)/10==int (StripSubdetector::TOB))subdetector="TOB";
    else if ((summaryhist_it->first)/10==int (StripSubdetector::TEC))subdetector="TEC";
    float thickness=0,pitch=-1;
    detparmap::iterator detparit=summarydetmap.find(summaryhist_it->first);
    if(detparit!=summarydetmap.end()){
      thickness = detparit->second->thickness;
      pitch = detparit->second->pitch;
    }    
    fitmap::iterator  fitpar=summaryfits.find(summaryhist_it->first);
    
    fit<<endl<<"--------------------------- SUMMARY FIT: "<<subdetector<<" LAYER/RING "<<(summaryhist_it->first%10)<<" -------------------------"<<endl<<endl;
    fit<<"Number of entries = "<<summaryhist_it->second->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<pitch<<" um "<<endl<<endl;    
    if(fitpar!=summaryfits.end()){
      fit<<"Chi Square/ndf = "<<(fitpar->second->chi2)/(fitpar->second->ndf)<<endl;
      fit<<"NdF        = "<<fitpar->second->ndf<<endl;
      fit<<"p0 = "<<fitpar->second->p0<<"     err p0 = "<<fitpar->second->errp0<<endl;
      fit<<"p1 = "<<fitpar->second->p1<<"     err p1 = "<<fitpar->second->errp1<<endl;
      fit<<"p2 = "<<fitpar->second->p2<<"     err p2 = "<<fitpar->second->errp2<<endl<<endl;
    }
    else fit<<"no fit parameters available"<<endl;
    summaryhist_it->second->SetDirectory(summary);
  }
  
  for(hist_it=histos.begin();hist_it!=histos.end(); hist_it++){   
    float thickness=0,pitch=-1;
    detparmap::iterator detparit=detmap.find(hist_it->first);
    if(detparit!=detmap.end()){
      thickness = detparit->second->thickness;
      pitch = detparit->second->pitch;
    }    
    fitmap::iterator  fitpar=fits.find(hist_it->first);
    fit<<endl<<"-------------------------- MODULE HISTOGRAM FIT ------------------------"<<endl<<endl;
    DetId id= DetId(hist_it->first);
    fit<<makename(id,true,false).c_str()<<endl<<endl;

    //edm::LogInfo("SiStripLorentzAngle") <<"detid "<<hist_it->first;
    //edm::LogInfo("SiStripLorentzAngle") <<"histo pointer"<<hist_it->second;

    fit<<"Number of entries = "<<hist_it->second->GetEntries()<<endl<<endl;
    fit<<"Detector thickness = "<<thickness<<" um "<<endl;
    fit<<"Detector pitch = "<<pitch<<" um "<<endl<<endl;
    if(fitpar!=fits.end()){
      fit<<"Chi Square/ndf = "<<(fitpar->second->chi2)/(fitpar->second->ndf)<<endl;
      fit<<"NdF        = "<<fitpar->second->ndf<<endl;
      fit<<"p0 = "<<fitpar->second->p0<<"     err p0 = "<<fitpar->second->errp0<<endl;
      fit<<"p1 = "<<fitpar->second->p1<<"     err p1 = "<<fitpar->second->errp1<<endl;
      fit<<"p2 = "<<fitpar->second->p2<<"     err p2 = "<<fitpar->second->errp2<<endl<<endl;
    }    
    if(id.subdetId() == int (StripSubdetector::TIB)){
      
      TIBDetId TIBid=TIBDetId(id);
      int layer = TIBid.layer()-1;
      if(TIBid.string()[0] == 1)hist_it->second->SetDirectory(TIBbwl[layer]);
      else if(TIBid.string()[0] == 2)hist_it->second->SetDirectory(TIBfwl[layer]);
    }
    
    else if(id.subdetId() == int (StripSubdetector::TID)){
      
      TIDDetId TIDid=TIDDetId(id);
      int ring = TIDid.ring()-1;
      if(TIDid.module()[0] == 1)hist_it->second->SetDirectory(TIDbwr[ring]);
      else if(TIDid.module()[0] == 2)hist_it->second->SetDirectory(TIDfwr[ring]);
    }
    
    else  if(id.subdetId()== int (StripSubdetector::TOB)){
      
      TOBDetId TOBid=TOBDetId(id);
      int layer = TOBid.layer()-1;
      if(TOBid.rod()[0] == 1)hist_it->second->SetDirectory(TOBbwl[layer]);
      else if(TOBid.rod()[0] == 2)hist_it->second->SetDirectory(TOBfwl[layer]);
    }
    else if(id.subdetId() == int (StripSubdetector::TEC)){
      
      TECDetId TECid=TECDetId(id);
      int ring = TECid.ring()-1;
      if(TECid.petal()[0] == 1)hist_it->second->SetDirectory(TECbwr[ring]);  
      else if(TECid.petal()[0] == 2)hist_it->second->SetDirectory(TECfwr[ring]);  
    }        
  }
  fit.close(); 
  hFile->Write();
  hFile->Close();
  
}
