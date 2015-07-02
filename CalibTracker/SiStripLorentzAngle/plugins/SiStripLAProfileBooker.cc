/*  VI Janurary 2012 
 * This file need to be migrated to the new interface of matched hit as the mono/stero are not there anymore
 * what is returned are hits w/o localpoistion, just cluster and id
 */
#include <string>
#include <iostream>
#include <fstream>
#include <functional>


#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLAProfileBooker.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include <TProfile.h>
#include <TStyle.h>
#include <TF1.h>

#include<list>

class DetIdLess 
  : public std::binary_function< const SiStripRecHit2D*,const SiStripRecHit2D*,bool> {
public:
  
  DetIdLess() {}
  
  bool operator()( const SiStripRecHit2D* a, const SiStripRecHit2D* b) const {
    return *(a->cluster())<*(b->cluster());
  }
};
  

  //Constructor

SiStripLAProfileBooker::SiStripLAProfileBooker(edm::ParameterSet const& conf) : 
  conf_(conf)
{
}

  //BeginRun

void SiStripLAProfileBooker::beginRun(const edm::EventSetup& c){

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  c.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
 
  //get magnetic field and geometry from ES
  edm::ESHandle<MagneticField> esmagfield;
  c.get<IdealMagneticFieldRecord>().get(esmagfield);
  const MagneticField *  magfield=&(*esmagfield);
  
  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker); 

  std::vector<uint32_t> activeDets;
  edm::ESHandle<SiStripDetCabling> tkmechstruct=0;
  if (conf_.getParameter<bool>("UseStripCablingDB")){ 
    c.get<SiStripDetCablingRcd>().get(tkmechstruct);
    activeDets.clear();
    tkmechstruct->addActiveDetectorsRawIds(activeDets);
  }
  else {
    const TrackerGeometry::DetIdContainer& Id = estracker->detIds();
    TrackerGeometry::DetIdContainer::const_iterator Iditer;    
    activeDets.clear();
    for(Iditer=Id.begin();Iditer!=Id.end();Iditer++){
      activeDets.push_back(Iditer->rawId());
    }
  }
   
  edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("Tracks");
  //Get Ids;
  double ModuleRangeMin=conf_.getParameter<double>("ModuleXMin");
  double ModuleRangeMax=conf_.getParameter<double>("ModuleXMax");
  double TIBRangeMin=conf_.getParameter<double>("TIBXMin");
  double TIBRangeMax=conf_.getParameter<double>("TIBXMax");
  double TOBRangeMin=conf_.getParameter<double>("TOBXMin");
  double TOBRangeMax=conf_.getParameter<double>("TOBXMax");
  int TIB_bin=conf_.getParameter<int>("TIB_bin");
  int TOB_bin=conf_.getParameter<int>("TOB_bin");
  int SUM_bin=conf_.getParameter<int>("SUM_bin");
    
  hFile = new TFile (conf_.getUntrackedParameter<std::string>("treeName").c_str(), "RECREATE" );
  
  Hit_Tree = hFile->mkdir("Hit_Tree");
  Track_Tree = hFile->mkdir("Track_Tree");
  Event_Tree = hFile->mkdir("Event_Tree");
  
  HitsTree = new TTree("HitsTree", "HitsTree");
  
  HitsTree->Branch("RunNumber", &RunNumber, "RunNumber/I");
  HitsTree->Branch("EventNumber", &EventNumber, "EventNumber/I");
  HitsTree->Branch("TanTrackAngle", &TanTrackAngle, "TanTrackAngle/F");
  HitsTree->Branch("TanTrackAngleParallel", &TanTrackAngleParallel, "TanTrackAngleParallel/F");
  HitsTree->Branch("ClSize", &ClSize, "ClSize/I");
  HitsTree->Branch("HitCharge", &HitCharge, "HitCharge/I");
  HitsTree->Branch("Hit_Std_Dev", &hit_std_dev, "hit_std_dev/F");
  HitsTree->Branch("Type", &Type, "Type/I");
  HitsTree->Branch("Layer", &Layer, "Layer/I");
  HitsTree->Branch("Wheel", &Wheel, "Wheel/I");
  HitsTree->Branch("bw_fw", &bw_fw, "bw_fw/I");
  HitsTree->Branch("Ext_Int", &Ext_Int, "Ext_Int/I");
  HitsTree->Branch("MonoStereo", &MonoStereo, "MonoStereo/I");
  HitsTree->Branch("MagField", &MagField, "MagField/F");
  HitsTree->Branch("SignCorrection", &SignCorrection, "SignCorrection/F");
  HitsTree->Branch("XGlobal", &XGlobal, "XGlobal/F");
  HitsTree->Branch("YGlobal", &YGlobal, "YGlobal/F");
  HitsTree->Branch("ZGlobal", &ZGlobal, "ZGlobal/F");
  HitsTree->Branch("ParticleCharge", &ParticleCharge, "ParticleCharge/I");
  HitsTree->Branch("Momentum", &Momentum, "Momentum/F");
  HitsTree->Branch("pt", &pt, "pt/F");
  HitsTree->Branch("chi2norm", &chi2norm, "chi2norm/F");
  HitsTree->Branch("EtaTrack", &EtaTrack, "EtaTrack/F");
  HitsTree->Branch("PhiTrack", &PhiTrack, "PhiTrack/F");
  HitsTree->Branch("TrajSize", &trajsize, "trajsize/I");
  HitsTree->Branch("HitNr", &HitNr, "HitNr/I");
  HitsTree->Branch("HitPerTrack", &HitPerTrack, "HitPerTrack/I");
  HitsTree->Branch("id_detector", &id_detector, "id_detector/I");
  HitsTree->Branch("thick_detector", &thick_detector, "thick_detector/F");
  HitsTree->Branch("pitch_detector", &pitch_detector, "pitch_detector/F");  
  HitsTree->Branch("Amplitudes", Amplitudes, "Amplitudes[ClSize]/I");
  
  HitsTree->SetDirectory(Hit_Tree);
  
  TrackTree = new TTree("TrackTree", "TrackTree");
  
  TrackTree->Branch("TrackCounter", &TrackCounter, "TrackCounter/I");
  
  TrackTree->SetDirectory(Track_Tree);
  
  EventTree = new TTree("EventTree", "EventTree");
  
  EventTree->Branch("EventCounter", &EventCounter, "EventCounter/I");
  
  EventTree->SetDirectory(Event_Tree);
  
      
  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;

  // create SiStripFolderOrganizer
  SiStripFolderOrganizer folder_organizer;

  dbe_ = edm::Service<DQMStore>().operator->();
  
  //get all detids
  
  for(std::vector<uint32_t>::const_iterator Id = activeDets.begin(); Id!=activeDets.end(); Id++){
    
    //  for(Iditer=Id.begin();Iditer!=Id.end();Iditer++){ //loop on detids
    DetId Iditero=DetId(*Id);
    DetId *Iditer=&Iditero;
    if((Iditer->subdetId() == int(StripSubdetector::TIB)) || (Iditer->subdetId() == int(StripSubdetector::TOB))){ //include only barrel
      
      int module_bin = 0;
      if(Iditer->subdetId() == int(StripSubdetector::TIB)){
	module_bin = TIB_bin;
      }else{
	module_bin = TOB_bin;
      }
      
      // create a TProfile for each module
      StripSubdetector subid(*Iditer);
      std::string hid;
      //Mono single sided detectors
      LocalPoint p;
      auto stripdet = tracker->idToDet(subid);
      if(!stripdet->isLeaf())continue;
      const StripTopology& topol=(const StripTopology&)stripdet->topology();
      float thickness=stripdet->specificSurface().bounds().thickness();
      
      folder_organizer.setDetectorFolder(Iditer->rawId(), tTopo);
      hid = hidmanager.createHistoId(TkTag.label().c_str(),"det",Iditer->rawId());
      MonitorElement * profile=dbe_->bookProfile(hid,hid,module_bin,ModuleRangeMin,ModuleRangeMax,20,0,5,"");
      detparameters *param=new detparameters;
      histos[Iditer->rawId()] = profile;
      detmap[Iditer->rawId()] = param;
      param->thickness = thickness*10000;
      param->pitch = topol.localPitch(p)*10000;
      
      const GlobalPoint globalp = (stripdet->surface()).toGlobal(p);
      GlobalVector globalmagdir = magfield->inTesla(globalp);
      param->magfield=(stripdet->surface()).toLocal(globalmagdir);
      
      profile->setAxisTitle("tan(#theta_{t})",1);
      profile->setAxisTitle("Cluster size",2);
      
      // create a summary histo if it does not exist already
      std::string name;
      unsigned int layerid;
      getlayer(subid,tTopo,name,layerid);
      name+=TkTag.label().c_str();
      if(summaryhisto.find(layerid)==(summaryhisto.end())){
	folder_organizer.setSiStripFolder();
	MonitorElement * summaryprofile=0;
	if (subid.subdetId()==int (StripSubdetector::TIB)||subid.subdetId()==int (StripSubdetector::TID))
	  summaryprofile=dbe_->bookProfile(name,name,SUM_bin,TIBRangeMin,TIBRangeMax,20,0,5,"");
	else if (subid.subdetId()==int (StripSubdetector::TOB)||subid.subdetId()==int (StripSubdetector::TEC))
	  summaryprofile=dbe_->bookProfile(name,name,SUM_bin,TOBRangeMin,TOBRangeMax,20,0,5,"");
	if(summaryprofile){
	  detparameters *summaryparam=new detparameters;
	  summaryhisto[layerid] = summaryprofile;
	  summarydetmap[layerid] = summaryparam;
	  summaryparam->thickness = thickness*10000;
	  summaryparam->pitch = topol.localPitch(p)*10000;
	  summaryprofile->setAxisTitle("tan(#theta_{t})",1);
	  summaryprofile->setAxisTitle("Cluster size",2);
	}
      }
      
    } 
  } 
  
  trackcollsize = 0;
  trajsize = 0;
  RunNumber = 0;
  EventNumber = 0;
  hitcounter = 0;
  hitcounter_2ndloop = 0;
  worse_double_hit = 0;
  better_double_hit = 0;
  eventcounter = 0;
  
  EventCounter = 1;
  TrackCounter = 1;
  
}

SiStripLAProfileBooker::~SiStripLAProfileBooker() {  
  detparmap::iterator detpariter;
  for( detpariter=detmap.begin(); detpariter!=detmap.end();++detpariter)delete detpariter->second;
  for( detpariter=summarydetmap.begin(); detpariter!=summarydetmap.end();++detpariter)delete detpariter->second;
  delete hFile;
}  

// Analyzer: Functions that gets called by framework every event

void SiStripLAProfileBooker::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  
  RunNumber = e.id().run();
  EventNumber = e.id().event();
  
  eventcounter++;
  
  EventTree->Fill();
  
  //Analysis of Trajectory-RecHits
  
  edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("Tracks");
  
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByLabel(TkTag,trackCollection);
  
  edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
  e.getByLabel(TkTag,TrajectoryCollection);
  
  edm::Handle<TrajTrackAssociationCollection> TrajTrackMap;
  e.getByLabel(TkTag, TrajTrackMap);
  
  const reco::TrackCollection *tracks=trackCollection.product();
  
  // FIXME this has to be changed to use pointers to clusters...
  std::map<const SiStripRecHit2D*,std::pair<float,float>,DetIdLess> hitangleassociation;
  std::list<SiStripRecHit2D> cache;  // ugly, inefficient, effective in making the above working
  
  trackcollsize=tracks->size();
  trajsize=TrajectoryCollection->size();
  
  edm::LogInfo("SiStripLAProfileBooker::analyze") <<" Number of tracks in event = "<<trackcollsize<<"\n";
  edm::LogInfo("SiStripLAProfileBooker::analyze") <<" Number of trajectories in event = "<<trajsize<<"\n";
  
  TrajTrackAssociationCollection::const_iterator TrajTrackIter;
  
  for(TrajTrackIter = TrajTrackMap->begin(); TrajTrackIter!= TrajTrackMap->end(); TrajTrackIter++){ //loop on trajectories
    
    if(TrajTrackIter->key->foundHits()>=5){
      
      TrackTree->Fill();
      
      ParticleCharge = -99;
      Momentum = -99;
      pt = -99;
      chi2norm = -99;
      HitPerTrack = -99;
      EtaTrack = -99;
      PhiTrack = -99;
      
      ParticleCharge = TrajTrackIter->val->charge();
      pt = TrajTrackIter->val->pt();
      Momentum = TrajTrackIter->val->p();
      chi2norm = TrajTrackIter->val->normalizedChi2();
      EtaTrack = TrajTrackIter->val->eta();
      PhiTrack = TrajTrackIter->val->phi();
      HitPerTrack = TrajTrackIter->key->foundHits();
      
      std::vector<TrajectoryMeasurement> TMeas=TrajTrackIter->key->measurements();
      std::vector<TrajectoryMeasurement>::iterator itm;
      
      for (itm=TMeas.begin();itm!=TMeas.end();itm++){ //loop on hits
	
	int i;
	for(i=0;i<100;i++){Amplitudes[i]=0;}
	
	TanTrackAngle = -99;
	TanTrackAngleParallel=-99;
	ClSize = -99;
	HitCharge = 0;
	Type = -99;
	Layer = -99;
	Wheel = -99;
	bw_fw = -99;
	Ext_Int = -99;
	MonoStereo = -99;
	MagField = -99;
	SignCorrection = -99;
	XGlobal = -99;
	YGlobal = -99;
	ZGlobal = -99;
	barycenter = -99;
	hit_std_dev = -99;
	sumx = 0;
	id_detector=-1;
	thick_detector=-1;
	pitch_detector=-1;       
	HitNr = 1;    
	
        SiStripRecHit2D lhit;
	TrajectoryStateOnSurface tsos=itm->updatedState();
	const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
	if((thit->geographicalId().subdetId() == int(StripSubdetector::TIB)) ||  thit->geographicalId().subdetId()== int(StripSubdetector::TOB)){ //include only barrel
	  const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>((*thit).hit());
	  const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>((*thit).hit());
	  const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>((*thit).hit());
	  if(phit) {lhit = phit->originalHit(); hit = &lhit;}
	  
	  LocalVector trackdirection=tsos.localDirection();
	  
	  if(matchedhit){//if matched hit...
	    
	    GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(matchedhit->geographicalId());
	    
	    GlobalVector gtrkdir=gdet->toGlobal(trackdirection);	
	    
	    // THIS THE POINTER TO THE MONO HIT OF A MATCHED HIT 
	    
	    // top be migrated to the more direct interface of matchedhit
	    cache.push_back(matchedhit->monoHit()); 
	    const SiStripRecHit2D * monohit = &cache.back();   
	    const SiStripRecHit2D::ClusterRef & monocluster=monohit->cluster();
	    const GeomDetUnit * monodet=gdet->monoDet();
	    // this does not exists anymore! either project the matched or use CPE
	    const LocalPoint monoposition = monohit->localPosition();   
	    
            StripSubdetector detid=(StripSubdetector)monohit->geographicalId();
	    id_detector = detid.rawId();
	    thick_detector=monodet->specificSurface().bounds().thickness();
            const StripTopology& mtopol=(StripTopology&)monodet->topology();
            pitch_detector = mtopol.localPitch(monoposition);
            const GlobalPoint monogposition = (monodet->surface()).toGlobal(monoposition);
	    ClSize = (monocluster->amplitudes()).size();
	    
	    const auto & amplitudes = monocluster->amplitudes();
	    
	    barycenter = monocluster->barycenter()- 0.5; 
	    uint16_t FirstStrip = monocluster->firstStrip();
	    auto begin=amplitudes.begin();
	    nstrip=0;
	    for(auto idigi=begin; idigi!=amplitudes.end(); idigi++){
	      Amplitudes[nstrip]=*idigi;
	      sumx+=pow(((FirstStrip+idigi-begin)-barycenter),2)*(*idigi);
	      HitCharge+=*idigi;
	    }
	    hit_std_dev = sqrt(sumx/HitCharge);
	    
	    
            XGlobal = monogposition.x();
	    YGlobal = monogposition.y();
	    ZGlobal = monogposition.z();
	    
	    Type = detid.subdetId();
	    MonoStereo=detid.stereo();
	    
	    if(detid.subdetId() == int (StripSubdetector::TIB)){
	      
	      Layer = tTopo->tibLayer(detid);
	      bw_fw = tTopo->tibStringInfo(detid)[0];
	      Ext_Int = tTopo->tibStringInfo(detid)[1];
            }
	    if(detid.subdetId() == int (StripSubdetector::TOB)){
	      
	      Layer = tTopo->tobLayer(detid);
	      bw_fw = tTopo->tobRodInfo(detid)[0];
            }
	    if(detid.subdetId() == int (StripSubdetector::TID)){
	      
	      Wheel = tTopo->tidWheel(detid);
	      bw_fw = tTopo->tidModuleInfo(detid)[0];
            }
	    if(detid.subdetId() == int (StripSubdetector::TEC)){
	      
	      Wheel = tTopo->tecWheel(detid);
	      bw_fw = tTopo->tecPetalInfo(detid)[0];
            }
	    
	    
	    LocalVector monotkdir=monodet->toLocal(gtrkdir);
	    
	    if(monotkdir.z()!=0){
	      
	      // THE LOCAL ANGLE (MONO)
	      float tanangle = monotkdir.x()/monotkdir.z();
	      TanTrackAngleParallel = monotkdir.y()/monotkdir.z();	      
	      TanTrackAngle = tanangle;
	      detparmap::iterator TheDet=detmap.find(detid.rawId());
              LocalVector localmagdir;
              if(TheDet!=detmap.end())localmagdir=TheDet->second->magfield;
              MagField = localmagdir.mag();
	      if(MagField != 0.){
		LocalVector monoylocal(0,1,0);
		float signcorrection = (localmagdir * monoylocal)/(MagField);
		if(signcorrection!=0)SignCorrection=1/signcorrection;
	      }
	      
	      std::map<const SiStripRecHit2D *,std::pair<float,float>,DetIdLess>::iterator alreadystored=hitangleassociation.find(monohit);
	      
	      
	      if(alreadystored != hitangleassociation.end()){//decide which hit take
		if(itm->estimate() >  alreadystored->second.first){
		  worse_double_hit++;}
		if(itm->estimate() <  alreadystored->second.first){
		  better_double_hit++;
		  hitangleassociation.insert(std::make_pair(monohit, std::make_pair(itm->estimate(),tanangle)));
		  
		}}
	      else{
		hitangleassociation.insert(make_pair(monohit, std::make_pair(itm->estimate(),tanangle)));
		HitsTree->Fill();
		hitcounter++;}
	      
	      // THIS THE POINTER TO THE STEREO HIT OF A MATCHED HIT 
	      
	      // top be migrated to the more direct interface of matchedhit
	      cache.push_back(matchedhit->stereoHit());
	      const SiStripRecHit2D * stereohit = &cache.back();
	      const SiStripRecHit2D::ClusterRef & stereocluster=stereohit->cluster();
	      const GeomDetUnit * stereodet=gdet->stereoDet();
	      // this does not exists anymore! either project the matched or use CPE
	      const LocalPoint stereoposition = stereohit->localPosition();    
	      StripSubdetector detid=(StripSubdetector)stereohit->geographicalId();
	      id_detector = detid.rawId();
	      thick_detector=stereodet->specificSurface().bounds().thickness();
	      const StripTopology& stopol=(StripTopology&)stereodet->topology();
	      pitch_detector = stopol.localPitch(stereoposition);
	      const GlobalPoint stereogposition = (stereodet->surface()).toGlobal(stereoposition);
	      
	      ClSize = (stereocluster->amplitudes()).size();
	      
	      const auto &  amplitudes = stereocluster->amplitudes();
	      
	      barycenter = stereocluster->barycenter()- 0.5; 
	      uint16_t FirstStrip = stereocluster->firstStrip();
	      auto begin=amplitudes.begin();
	      nstrip=0;
	      for(auto idigi=begin; idigi!=amplitudes.end(); idigi++){
		Amplitudes[nstrip]=*idigi;
		sumx+=pow(((FirstStrip+idigi-begin)-barycenter),2)*(*idigi);
		HitCharge+=*idigi;
	      }
	      hit_std_dev = sqrt(sumx/HitCharge);
	      
	      XGlobal = stereogposition.x();
	      YGlobal = stereogposition.y();
	      ZGlobal = stereogposition.z();
	      
	      Type = detid.subdetId();
	      MonoStereo=detid.stereo();
	      
	      if(detid.subdetId() == int (StripSubdetector::TIB)){
		
		Layer = tTopo->tibLayer(detid);
		bw_fw = tTopo->tibStringInfo(detid)[0];
		Ext_Int = tTopo->tibStringInfo(detid)[1];
	      }
	      if(detid.subdetId() == int (StripSubdetector::TOB)){
		
		Layer = tTopo->tobLayer(detid);
		bw_fw = tTopo->tobRodInfo(detid)[0];
	      }
	      if(detid.subdetId() == int (StripSubdetector::TID)){
		
		Wheel = tTopo->tidWheel(detid);
		bw_fw = tTopo->tidModuleInfo(detid)[0];
	      }
	      if(detid.subdetId() == int (StripSubdetector::TEC)){
		
		Wheel = tTopo->tecWheel(detid);
		bw_fw = tTopo->tecPetalInfo(detid)[0];
	      }
	      
	      
	      LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	      
	      if(stereotkdir.z()!=0){
		
		// THE LOCAL ANGLE (STEREO)
		float tanangle = stereotkdir.x()/stereotkdir.z();
		TanTrackAngleParallel = stereotkdir.y()/stereotkdir.z();
		TanTrackAngle = tanangle;
		detparmap::iterator TheDet=detmap.find(detid.rawId());
                LocalVector localmagdir;
                if(TheDet!=detmap.end())localmagdir=TheDet->second->magfield;
                MagField = localmagdir.mag();
		LocalVector stereoylocal(0,1,0);
	        if(MagField != 0.){
		  float signcorrection = (localmagdir * stereoylocal)/(MagField);
		  if(signcorrection!=0)SignCorrection=1/signcorrection;}
		
		std::map<const SiStripRecHit2D *,std::pair<float,float>,DetIdLess>::iterator alreadystored=hitangleassociation.find(stereohit);
		
		if(alreadystored != hitangleassociation.end()){//decide which hit take
		  if(itm->estimate() >  alreadystored->second.first){
		    worse_double_hit++;}
		  if(itm->estimate() <  alreadystored->second.first){
		    better_double_hit++;
		    hitangleassociation.insert(std::make_pair(stereohit, std::make_pair(itm->estimate(),tanangle)));
		    
		  }}
		else{
		  hitangleassociation.insert(std::make_pair(stereohit, std::make_pair(itm->estimate(),tanangle)));
		  HitsTree->Fill();
		  hitcounter++;}
		
	      }
	    }
	  }
	  else if(hit){
	    
	    
	    //  hit= POINTER TO THE RECHIT
	    
	    const SiStripRecHit2D::ClusterRef & cluster=hit->cluster();
	    
	    GeomDetUnit * gdet=(GeomDetUnit *)tracker->idToDet(hit->geographicalId());
	    const LocalPoint position = hit->localPosition();    
            StripSubdetector detid=(StripSubdetector)hit->geographicalId();
	    id_detector = detid.rawId();
	    thick_detector=gdet->specificSurface().bounds().thickness();
            const StripTopology& topol=(StripTopology&)gdet->topology();
            pitch_detector = topol.localPitch(position);
            const GlobalPoint gposition = (gdet->surface()).toGlobal(position);
	    
	    ClSize = (cluster->amplitudes()).size();
	    
	    const auto &  amplitudes = cluster->amplitudes();
	    
	    barycenter = cluster->barycenter()- 0.5; 
	    uint16_t FirstStrip = cluster->firstStrip();
	    nstrip=0;
            auto begin =amplitudes.begin();
	    for(auto idigi=amplitudes.begin(); idigi!=amplitudes.end(); idigi++){
	      Amplitudes[nstrip]=*idigi;
	      sumx+=pow(((FirstStrip+idigi-begin)-barycenter),2)*(*idigi);
	      HitCharge+=*idigi;
	    }
	    hit_std_dev = sqrt(sumx/HitCharge);
	    
            XGlobal = gposition.x();
	    YGlobal = gposition.y();
	    ZGlobal = gposition.z();
	    
	    Type = detid.subdetId();
	    MonoStereo=detid.stereo();
	    
	    if(detid.subdetId() == int (StripSubdetector::TIB)){
	      
	      Layer = tTopo->tibLayer(detid);
	      bw_fw = tTopo->tibStringInfo(detid)[0];
	      Ext_Int = tTopo->tibStringInfo(detid)[1];
            }
	    if(detid.subdetId() == int (StripSubdetector::TOB)){
	      
	      Layer = tTopo->tobLayer(detid);
	      bw_fw = tTopo->tobRodInfo(detid)[0];
            }
	    if(detid.subdetId() == int (StripSubdetector::TID)){
	      
	      Wheel = tTopo->tidWheel(detid);
	      bw_fw = tTopo->tidModuleInfo(detid)[0];
            }
	    if(detid.subdetId() == int (StripSubdetector::TEC)){
	      
	      Wheel = tTopo->tecWheel(detid);
	      bw_fw = tTopo->tecPetalInfo(detid)[0];
            }
	    
	    if(trackdirection.z()!=0){
	      
	      // THE LOCAL ANGLE 
	      float tanangle = trackdirection.x()/trackdirection.z();
	      TanTrackAngleParallel = trackdirection.y()/trackdirection.z();
	      TanTrackAngle = tanangle;
              detparmap::iterator TheDet=detmap.find(detid.rawId());
              LocalVector localmagdir;
              if(TheDet!=detmap.end())localmagdir=TheDet->second->magfield;
              MagField = localmagdir.mag();
	      if(MagField != 0.){
		LocalVector ylocal(0,1,0);
		float signcorrection = (localmagdir * ylocal)/(MagField);
		if(signcorrection!=0)SignCorrection=1/signcorrection;}
	      
	      std::map<const SiStripRecHit2D *,std::pair<float,float>, DetIdLess>::iterator alreadystored=hitangleassociation.find(hit);
	      
	      if(alreadystored != hitangleassociation.end()){//decide which hit take
		if(itm->estimate() >  alreadystored->second.first){
		  worse_double_hit++;}
		if(itm->estimate() <  alreadystored->second.first){
		  better_double_hit++;
		  hitangleassociation.insert(std::make_pair(hit, std::make_pair(itm->estimate(),tanangle)));
		  
		}}
	      else{
		hitangleassociation.insert(std::make_pair(hit,std::make_pair(itm->estimate(), tanangle) ) );
		HitsTree->Fill();
		hitcounter++;}
	      
	      
	    }
	  }
	}
      }
    }
  }
  std::map<const SiStripRecHit2D *,std::pair<float,float>,DetIdLess>::iterator hitsiter;
  
  
  for(hitsiter=hitangleassociation.begin();hitsiter!=hitangleassociation.end();hitsiter++){
    
    hitcounter_2ndloop++;
    
    const SiStripRecHit2D* hit=hitsiter->first;
    const SiStripRecHit2D::ClusterRef & cluster=hit->cluster();
    
    size=(cluster->amplitudes()).size();
    
    StripSubdetector detid=(StripSubdetector)hit->geographicalId();  
    
    float tangent = hitsiter->second.second;
    
    //Sign and XZ plane projection correction applied in TrackLocalAngle (TIB|TOB layers)
    
    detparmap::iterator thedet=detmap.find(detid.rawId());
    LocalVector localmagdir;
    if(thedet!=detmap.end())localmagdir=thedet->second->magfield;
    float localmagfield = localmagdir.mag();
    
    if(localmagfield != 0.){
      
      LocalVector ylocal(0,1,0);
      
      float normprojection = (localmagdir * ylocal)/(localmagfield);
      
      if(normprojection == 0.)LogDebug("SiStripLAProfileBooker::analyze")<<"Error: YBprojection = 0";
      
      else{
	float signprojcorrection = 1/normprojection;
	tangent*=signprojcorrection;
      }
    }
    
    //Filling histograms
    
    histomap::iterator thehisto=histos.find(detid.rawId());
    
    if(thehisto==histos.end())edm::LogError("SiStripLAProfileBooker::analyze")<<"Error: the profile associated to"<<detid.rawId()<<"does not exist! ";    
    else thehisto->second->Fill(tangent,size);
    
    //Summary histograms
    std::string name;
    unsigned int layerid;
    getlayer(detid,tTopo,name,layerid);
    histomap::iterator thesummaryhisto=summaryhisto.find(layerid);
    if(thesummaryhisto==summaryhisto.end())edm::LogError("SiStripLAProfileBooker::analyze")<<"Error: the profile associated to subdet "<<name<<"does not exist! ";   
    else thesummaryhisto->second->Fill(tangent,size);
    
  }
  
  
}


//Makename function
 
void SiStripLAProfileBooker::getlayer(const DetId & detid, const TrackerTopology* tTopo, std::string &name,unsigned int &layerid){
    int layer=0;
    std::stringstream layernum;

    if(detid.subdetId() == int (StripSubdetector::TIB)){
      
      name+="TIB_Layer_";
      layer = tTopo->tibLayer(detid);
    }

    else if(detid.subdetId() == int (StripSubdetector::TID)){
      
      name+="TID_Ring_";
      layer = tTopo->tidRing(detid);
    }

    else if(detid.subdetId() == int (StripSubdetector::TOB)){
      
      name+="TOB_Layer_";
      layer = tTopo->tobLayer(detid);

    }

    else if(detid.subdetId() == int (StripSubdetector::TEC)){
      
      name+="TEC_Ring_";
      layer = tTopo->tecRing(detid);
    }
    layernum<<layer;
    name+=layernum.str();
    layerid=detid.subdetId()*10+layer;
  
}
 
void SiStripLAProfileBooker::endJob(){

  std::string outputFile_ =conf_.getUntrackedParameter<std::string>("fileName", "LorentzAngle.root");
  dbe_->save(outputFile_);
  
  hFile->Write();
  hFile->Close();
}
