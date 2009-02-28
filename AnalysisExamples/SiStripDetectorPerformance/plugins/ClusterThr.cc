#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterThr.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "sstream"

#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TNtupleD.h"
#include "TKey.h"
#include "langaus.C"

namespace cms{
  ClusterThr::ClusterThr(edm::ParameterSet const& conf): 
    conf_(conf),
    fileName_(conf.getParameter<std::string>("fileName")), 
    Cluster_src_( conf.getParameter<edm::InputTag>( "Cluster_src" ) ),
    NoiseMode_( conf.getParameter<uint32_t>("NoiseMode") ),
    ModulesToBeExcluded_(conf.getParameter< std::vector<uint32_t> >("ModulesToBeExcluded") ),
    subDets(conf.getParameter< std::vector<std::string> >("SubDets") ),
    layers(conf.getParameter< std::vector<uint32_t> >("Layers") ),
    ThC_(conf.getParameter< edm::ParameterSet > ("ThC") ),
    ThS_(conf.getParameter< edm::ParameterSet > ("ThS") ),
    ThN_(conf.getParameter< edm::ParameterSet > ("ThN") ),
    startThC_( ThC_.getParameter<double>("startThC") ),
    stopThC_(ThC_.getParameter<double>("stopThC") ),
    stepThC_(ThC_.getParameter<double>("stepThC") ),
    startThS_( ThS_.getParameter<double>("startThS")),
    stopThS_( ThS_.getParameter<double>("stopThS")),
    stepThS_( ThS_.getParameter<double>("stepThS")),
    startThN_( ThN_.getParameter<double>("startThN")),
    stopThN_( ThN_.getParameter<double>("stopThN")),
    stepThN_( ThN_.getParameter<double>("stepThN"))
  {
  }
  
  ClusterThr::~ClusterThr(){
  }
  
  void ClusterThr::beginRun(const edm::Run& run, const edm::EventSetup& es ) {
    
    es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );
    es.get<TrackerDigiGeometryRecord>().get( tkgeom );    
    es.get<SiStripQualityRcd>().get(SiStripQuality_);
    
    
    book();
  }
  
  void ClusterThr::book() {
    
    TFileDirectory ClusterNoise = fFile->mkdir( "ClusterNoise" );
    TFileDirectory ClusterSignal = fFile->mkdir("ClusterSignal");
    TFileDirectory ClusterStoN = fFile->mkdir("ClusterStoN");
    TFileDirectory ClusterStoNCorr = fFile->mkdir("ClusterStoNCorr");
    TFileDirectory ClusterWidth = fFile->mkdir("ClusterWidth");
    TFileDirectory ClusterPos = fFile->mkdir("ClusterPos");
    TFileDirectory ClusterNum = fFile->mkdir("ClusterNum");
    
    //Create histograms
    Hlist = new THashList();
    
    for (float Thc=startThC_;Thc<stopThC_;Thc+=stepThC_)
      for (float Ths=startThS_;Ths<stopThS_ && Ths<Thc; Ths+=stepThS_)	
	for (float Thn=startThN_;Thn<stopThN_ && Thn<Ths; Thn+=stepThN_)
	  for (int k=0;k<2;k++){
	    
	    char cappOn[128], cappOff[128];
	    sprintf(cappOn,"_%s_Th_%2.1f_%2.1f_%2.1f_On",k==0?"S":"B",Thc,Ths,Thn);
	    sprintf(cappOff,"_%s_Th_%2.1f_%2.1f_%2.1f_Off",k==0?"S":"B",Thc,Ths,Thn);

	    TString appStringOn(cappOn),appStringOff(cappOff);
	    
	    //Cluster Width
	    name="cWidth"+appStringOn;
	    bookHlist("TH1ClusterWidth",ClusterWidth, name, "Nstrip" );
	    
	    name="cWidth"+appStringOff;
	    bookHlist("TH1ClusterWidth",ClusterWidth, name, "Nstrip" );

	    //Cluster Noise
	    name="cNoise"+appStringOn;
	    bookHlist("TH1ClusterNoise",ClusterNoise, name, "ADC count" );

	    name="cNoise"+appStringOff;
	    bookHlist("TH1ClusterNoise",ClusterNoise, name, "ADC count" );
	    
	    //Cluster Signal
	    name="cSignal"+appStringOn;
	    bookHlist("TH1ClusterSignal",ClusterSignal, name, "ADC count" );

	    name="cSignal"+appStringOff;
	    bookHlist("TH1ClusterSignal",ClusterSignal, name, "ADC count" );	    

	    //Cluster StoN
	    name="cStoN"+appStringOn;
	    bookHlist("TH1ClusterStoN",ClusterStoN, name );

	    name="cStoN"+appStringOff;
	    bookHlist("TH1ClusterStoN",ClusterStoN, name );

	    //Cluster StoNCorr
	    name="cStoNCorr"+appStringOn;
	    bookHlist("TH1ClusterStoN",ClusterStoNCorr, name );

	    //Cluster Pos
	    name="cPos"+appStringOn;
	    bookHlist("TH1ClusterPos",ClusterPos, name, "Nstrip" );

	    name="cPos"+appStringOff;
	    bookHlist("TH1ClusterPos",ClusterPos, name, "Nstrip" );
	    	    
	    //Cluster Number
	    name="cNum"+appStringOn;
	    bookHlist("TH1ClusterNum",ClusterNum, name );

	    name="cNum"+appStringOff;
	    bookHlist("TH1ClusterNum",ClusterNum, name );
	  }
    
    
  }
  
  //------------------------------------------------------------------------------------------
  
  void ClusterThr::endJob() {  
    TNtupleD *tntuple = fFile->make<TNtupleD>("results","results","Tc:Ts:Tn:NTsOn:NsOn:MeanWsOn:RmsWsOn:MPVs:FWHMs:NTbOn:NbOn:MeanWbOn:RmsWbOn:NTsOff:NsOff:MeanWsOff:RmsWsOff:NTbOff:NbOff:MeanWbOff:RmsWbOff");
    
    std::vector<double> values(tntuple->GetNvar(),0);
    
    for (float Tc=startThC_;Tc<stopThC_;Tc+=stepThC_)
      for (float Ts=startThS_;Ts<stopThS_ && Ts<Tc; Ts+=stepThS_)	
	for (float Tn=startThN_;Tn<stopThN_ && Tn<Ts; Tn+=stepThN_){

	  char cappSOn[128],cappBOn[128],cappSOff[128],cappBOff[128];
	  sprintf(cappSOn,"_S_Th_%2.1f_%2.1f_%2.1f_On",Tc,Ts,Tn);
	  sprintf(cappBOn,"_B_Th_%2.1f_%2.1f_%2.1f_On",Tc,Ts,Tn);
	  sprintf(cappSOff,"_S_Th_%2.1f_%2.1f_%2.1f_Off",Tc,Ts,Tn);
	  sprintf(cappBOff,"_B_Th_%2.1f_%2.1f_%2.1f_Off",Tc,Ts,Tn);
	  TString appSOn(cappSOn), appSOff(cappSOff);
	  TString appBOn(cappBOn), appBOff(cappBOff);
	  
	  values[0]=Tc;
	  values[1]=Ts;
	  values[2]=Tn;
	  for (int k=0;k<2;k++){
	    if(k==0){//SIGNAL clusters ontrack
	      values[iNsOn]=((TH1F*) Hlist->FindObject("cNum"+appSOn))->GetMean();
	      values[iNTsOn]=((TH1F*) Hlist->FindObject("cWidth" +appSOn))->GetEntries();
	      values[iMeanWsOn]=((TH1F*) Hlist->FindObject("cWidth" +appSOn))->GetMean();
	      values[iRmsWsOn]=((TH1F*) Hlist->FindObject("cWidth" +appSOn))->GetRMS();
	      //and off track
	      values[iNsOff]=((TH1F*) Hlist->FindObject("cNum"+appSOff))->GetMean();
	      values[iNTsOff]=((TH1F*) Hlist->FindObject("cWidth" +appSOff))->GetEntries();
	      values[iMeanWsOff]=((TH1F*) Hlist->FindObject("cWidth" +appSOff))->GetMean();
	      values[iRmsWsOff]=((TH1F*) Hlist->FindObject("cWidth" +appSOff))->GetRMS();

	      TH1F *h = ((TH1F*) Hlist->FindObject("cStoNCorr"+appSOn));
	      double peak=0;
	      double fwhm=0;
	      langausN(h,peak,fwhm,0,0,true,"RB");
	      h->Write();
	      values[iMPVs]=peak;
	      values[iFWHMs]=fwhm;
	    }else{//BG clusters ontrack
	      values[iNbOn]=((TH1F*) Hlist->FindObject("cNum"+appBOn))->GetMean();
	      values[iNTbOn]=((TH1F*) Hlist->FindObject("cWidth" +appBOn))->GetEntries();
	      values[iMeanWbOn]=((TH1F*) Hlist->FindObject("cWidth" +appBOn))->GetMean();
	      values[iRmsWbOn]=((TH1F*) Hlist->FindObject("cWidth" +appBOn))->GetRMS();
	      //and off track
	      values[iNbOff]=((TH1F*) Hlist->FindObject("cNum"+appBOff))->GetMean();
	      values[iNTbOff]=((TH1F*) Hlist->FindObject("cWidth" +appBOff))->GetEntries();
	      values[iMeanWbOff]=((TH1F*) Hlist->FindObject("cWidth" +appBOff))->GetMean();
	      values[iRmsWbOff]=((TH1F*) Hlist->FindObject("cWidth" +appBOff))->GetRMS();

	    }
	  }
	  tntuple->Fill((double*) &values[0]);
	}  
    
    tntuple->Write();
    
    edm::LogInfo("ClusterThr") << "[ClusterThr::endJob()] ........ Closed"<< std::endl;
    
  }
  
//------------------------------------------------------------------------
  
  void ClusterThr::analyze(const edm::Event& e, const edm::EventSetup& es){
    
    runNb   = e.id().run();
    eventNb = e.id().event();
    vPSiStripCluster.clear();
    countOn=0;
    countOff=0;
    
    edm::LogInfo("ClusterAnalysis") << "[ClusterThr::analyze] Processing run " << runNb << " event " << eventNb << std::endl;
    
    e.getByLabel( Cluster_src_, dsv_SiStripCluster);
    std::string TrackProducer = conf_.getParameter<std::string>("TrackProducer");
    std::string TrackLabel = conf_.getParameter<std::string>("TrackLabel");
    
    e.getByLabel(TrackProducer, TrackLabel, trackCollection);//takes the track collection
    
    if (trackCollection.isValid()){
    }else{
      edm::LogError("ClusterThrFilter")<<" Track Collection is not valid !! " << TrackLabel<<std::endl;
      tracksCollection_in_EventTree=false;
    }
    
    // trajectory input
    e.getByLabel(TrackProducer, TrackLabel, TrajectoryCollection);
    e.getByLabel(TrackProducer, TrackLabel, TItkAssociatorCollection);
    if( TItkAssociatorCollection.isValid()){
    }else{
      edm::LogError("ClusterThrFilter")<<"Association not found "<<std::endl;
      trackAssociatorCollection_in_EventTree=false;
    }
    
    
    //Perform track study
    if (tracksCollection_in_EventTree && trackAssociatorCollection_in_EventTree) trackStudy(es);

    OffTrackClusters(es);       
    
    // 	  //Perform quality discrimination      
    // 	  const  edm::ParameterSet ps_b = conf_.getParameter<edm::ParameterSet>("BadModuleStudies");
    // 	  if  ( ps_b.getParameter<bool>("Bad") ) {//it will perform Bad modules discrimination
    // 	    short n_Apv;
    // 	    switch((int)SiStripClusterInfo_.firstStrip()/128){
    // 	    case 0:
    // 	      n_Apv=0;
    // 	      break;
    // 	    case 1:
    // 	      n_Apv=1;
    // 	      break;
    // 	    case 2:
    // 	      n_Apv=2;
    // 	      break;
    // 	    case 3:
    // 	      n_Apv=3;
    // 	      break;
    // 	    case 4:
    // 	      n_Apv=4;
    // 	      break;
    // 	    case 5:
    // 	      n_Apv=5;
    // 	      break;
    // 	    }
	    
    // 	    if ( ps_b.getParameter<bool>("justGood") ){//it will analyze just good modules 
    // 	      //	  edm::LogInfo("SiStripQuality") << "Just good module selected " << std::endl;
    // 	      if(SiStripQuality_->IsModuleBad(detid)){
    // 		edm::LogInfo("SiStripQuality") << "\n Excluding cluster on bad module " << detid << std::endl;
    // 		continue;
    // 	      }else if(SiStripQuality_->IsApvBad(detid, n_Apv)){
    // 		//	    edm::LogInfo("SiStripQuality") << "\n Excluding bad module and APV " << detid << n_Apv << std::endl;
    // 		continue;
    // 	      }
    // 	    }else{
    // 	      //	  edm::LogInfo("SiStripQuality") << "Just bad module selected " << std::endl;
    // 	      if(!SiStripQuality_->IsModuleBad(detid) || !SiStripQuality_->IsApvBad(detid, n_Apv)){
    // 		//	    edm::LogInfo("SiStripQuality") << "\n Skipping good module " << detid << std::endl;
    // 		continue;
    // 	      }
    // 	    }
    // 	  }
    
  }
  
  void ClusterThr::trackStudy(const edm::EventSetup& es)
  {
    
    const reco::TrackCollection tC = *(trackCollection.product());
    int i=0;
    std::vector<TrajectoryMeasurement> measurements;
    for(TrajTrackAssociationCollection::const_iterator it =  TItkAssociatorCollection->begin();it !=  TItkAssociatorCollection->end(); ++it){
      const edm::Ref<std::vector<Trajectory> > traj_iterator = it->key;  
      // Trajectory Map, extract Trajectory for this track
      reco::TrackRef trackref = it->val;
      edm::LogInfo("ClusterThr")
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
	
	const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>( ttrh->hit() );
	const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>( ttrh->hit() );
	const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>( ttrh->hit() );	
	
	RecHitType type=Single;
	
	if(matchedhit){
	  edm::LogInfo("ClusterThr")<<"\nMatched recHit found"<< std::endl;
	  type=Matched;
	  
	  GluedGeomDet * gdet=(GluedGeomDet *)tkgeom->idToDet(matchedhit->geographicalId());
	  GlobalVector gtrkdirup=gdet->toGlobal(updatedtsos.localMomentum());	    
	  //mono side
	  const GeomDetUnit * monodet=gdet->monoDet();
	  statedirection=monodet->toLocal(gtrkdirup);
	  if(statedirection.mag() != 0)	  RecHitInfo(matchedhit->monoHit(),statedirection,trackref,es);
	  //stereo side
	  const GeomDetUnit * stereodet=gdet->stereoDet();
	  statedirection=stereodet->toLocal(gtrkdirup);
	  if(statedirection.mag() != 0)	  RecHitInfo(matchedhit->stereoHit(),statedirection,trackref,es);
	  ss<<"\nLocalMomentum (stereo): " <<  statedirection;
	}
	else if(phit){
	  edm::LogInfo("ClusterThr")<<"\nProjected recHit found"<< std::endl;
	  type=Projected;
	  GluedGeomDet * gdet=(GluedGeomDet *)tkgeom->idToDet(phit->geographicalId());
	  
	  GlobalVector gtrkdirup=gdet->toGlobal(updatedtsos.localMomentum());
	  const SiStripRecHit2D&  originalhit=phit->originalHit();
	  const GeomDetUnit * det;
	  if(!StripSubdetector(originalhit.geographicalId().rawId()).stereo()){
	    //mono side
	    edm::LogInfo("ClusterThr")<<"\nProjected recHit found  MONO"<< std::endl;
	    det=gdet->monoDet();
	    statedirection=det->toLocal(gtrkdirup);
	    if(statedirection.mag() != 0) RecHitInfo(&(phit->originalHit()),statedirection,trackref,es);
	  }
	  else{
	    edm::LogInfo("ClusterThr")<<"\nProjected recHit found STEREO"<< std::endl;
	    //stereo side
	    det=gdet->stereoDet();
	    statedirection=det->toLocal(gtrkdirup);
	    if(statedirection.mag() != 0) RecHitInfo(&(phit->originalHit()),statedirection,trackref,es);
	  }
	}else {
	  if(hit!=0){
	    ss<<"\nSingle recHit found"<< std::endl;	  
	    statedirection=updatedtsos.localMomentum();
	    if(statedirection.mag() != 0) RecHitInfo(hit,statedirection,trackref,es);
	  }
	}
	ss <<"LocalMomentum: "<<statedirection
	   << "\nLocal x-z plane angle: "<<atan2(statedirection.x(),statedirection.z());	      
	edm::LogInfo("ClusterThr") <<ss.str() << std::endl;
      }
      
    }
  }
  
  void ClusterThr::RecHitInfo(const SiStripRecHit2D* tkrecHit, LocalVector LV,reco::TrackRef track_ref, const edm::EventSetup& es){
    
    if(!tkrecHit->isValid()){
      edm::LogInfo("ClusterThr") <<"\t\t Invalid Hit " << std::endl;
      return;  
    }
    
    const uint32_t& detid = tkrecHit->geographicalId().rawId();
    if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()){
      edm::LogInfo("ClusterThr") << "Modules Excluded" << std::endl;
      return;
    }
    
    edm::LogInfo("ClusterThr")
      <<"\n\t\tRecHit on det "<<tkrecHit->geographicalId().rawId()
      <<"\n\t\tRecHit in LP "<<tkrecHit->localPosition()
      <<"\n\t\tRecHit in GP "<<tkgeom->idToDet(tkrecHit->geographicalId())->surface().toGlobal(tkrecHit->localPosition()) 
      <<"\n\t\tRecHit trackLocal vector "<<LV.x() << " " << LV.y() << " " << LV.z() <<std::endl; 
    
    //Get SiStripCluster from SiStripRecHit
    if ( tkrecHit != NULL ){
      edm::LogInfo("ClusterThr") << "GOOD hit" << std::endl;
      const SiStripCluster* SiStripCluster_ = &*(tkrecHit->cluster());
      SiStripClusterInfo* SiStripClusterInfo_ = new SiStripClusterInfo(*SiStripCluster_,es);
 
      SiStripDetId a(detid);
      if (a.det()!=DetId::Tracker)
	return;
      
      std::string subdet;
      unsigned short int layer;
      
      if ( a.subdetId() == 3 ){
	TIBDetId b(detid);
	subdet="TIB";
	layer=b.layer();
      } else if ( a.subdetId() == 4 ) {
	TIDDetId b(detid);
	subdet="TID";
	layer=b.ring();
      } else if ( a.subdetId() == 5 ) {
	TOBDetId b(detid);
	subdet="TOB";
	layer=b.layer();
      } else if ( a.subdetId() == 6 ) {
	TECDetId b(detid);
	subdet="TEC";
	layer=b.ring();
      } 
      
      
      if (subDets.begin()!=subDets.end())
	if (std::find(subDets.begin(),subDets.end(),subdet)==subDets.end()){
	  //	  LogDebug("ClusterThr") << "Skipping SubDet " << subdet << std::endl;
	  return;
	}else{
	  LogDebug("ClusterThr") << "SubDet " << subdet << std::endl;
	}
      if (layers.begin()!=layers.end())
	if (std::find(layers.begin(),layers.end(),layer)==layers.end()){
	  //	  LogDebug("ClusterThr") << "Skipping Layer " << layer << std::endl;
	  return;
	}else{
	  LogDebug("ClusterThr") << "Layer " << layer << std::endl;
	}

      bool passedSeed=true;
      bool passedClus=true;
      for (float Thc=startThC_;Thc<stopThC_ && passedClus && passedSeed;Thc+=stepThC_){
	for (float Ths=startThS_;Ths<stopThS_  && Ths<Thc && passedSeed;Ths+=stepThS_){	
	  for (float Thn=startThN_;Thn<stopThN_ && Thn<Ths && passedSeed; Thn+=stepThN_){    
	    // 	    if (Thc<Ths)
	    // 	      continue;
	    if(clusterizer(SiStripClusterInfo_,Thc,Ths,Thn,passedSeed,passedClus,"OnTrack",LV)){
	      vPSiStripCluster.push_back(SiStripCluster_);
	      countOn++;
	    }
	  }
	}
      } 
     
	for (float Thc=startThC_;Thc<stopThC_;Thc+=stepThC_)
	  for (float Ths=startThS_;Ths<stopThS_ && Ths<Thc; Ths+=stepThS_)	
	    for (float Thn=startThN_;Thn<stopThN_ && Thn<Ths; Thn+=stepThN_)
	      for (int k=0;k<2;k++){
	      
	      char capp[128];
	      sprintf(capp,"_%s_Th_%2.1f_%2.1f_%2.1f_On",k==0?"S":"B",Thc,Ths,Thn);
	      std::map<std::string,int>::iterator iter=cNum.find(capp);
	      if (iter!=cNum.end()){
		if(iter->second == 0) LogDebug("ClusterThr") << "Filling with a zero for " << capp << std::endl;
		((TH1F*) Hlist->FindObject("cNum"+TString(capp)))->Fill(iter->second);
		iter->second=0;
	      }
	    }
      
      delete SiStripClusterInfo_; 
      
    }else{
      edm::LogError("ClusterThr") << "NULL hit" << std::endl;
    }	  
  }

void ClusterThr::OffTrackClusters( const edm::EventSetup& es)
{

  //Loop on Dets
  for ( edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=dsv_SiStripCluster->begin(); DSViter!=dsv_SiStripCluster->end();DSViter++){
    uint32_t detid=DSViter->id();
    if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()) continue;
    //Loop on Clusters
    edm::LogInfo("ClusterThr") << "on detid "<< detid << " N Cluster= " << DSViter->size();
    edmNew::DetSet<SiStripCluster>::const_iterator ClusIter = DSViter->begin();
    for(; ClusIter!=DSViter->end(); ClusIter++) {
      SiStripClusterInfo* SiStripClusterInfo_= new SiStripClusterInfo(*ClusIter,es);
	LogDebug("ClusterThr") << "ClusIter " << &*ClusIter << "\t " 
	                                << std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter)-vPSiStripCluster.begin();
	if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	  bool passedClus=true;
	  bool passedSeed=true;
	  for (float Thc=startThC_;Thc<stopThC_ && passedClus && passedSeed;Thc+=stepThC_){
	    for (float Ths=startThS_;Ths<stopThS_ && Ths<Thc && passedSeed;Ths+=stepThS_){	
	      for (float Thn=startThN_;Thn<stopThN_ && Thn<Ths && passedSeed; Thn+=stepThN_){    
		//		if (Thc<=Ths)
		//		  continue;
		if(clusterizer(SiStripClusterInfo_,Thc,Ths,Thn,passedSeed,passedClus,"OffTrack",LV)){
		  countOff++;
		  edm::LogInfo("ClusterThr") << "countOff" << countOff << std::endl;

		}
	      }
	    }
	  } 
	  
	  for (float Thc=startThC_;Thc<stopThC_;Thc+=stepThC_)
	    for (float Ths=startThS_;Ths<stopThS_ && Ths<Thc; Ths+=stepThS_)	
	      for (float Thn=startThN_;Thn<stopThN_ && Thn<Ths; Thn+=stepThN_)
		for (int k=0;k<2;k++){
		  
		  char capp[128];
		  sprintf(capp,"_%s_Th_%2.1f_%2.1f_%2.1f_Off",k==0?"S":"B",Thc,Ths,Thn);
		  std::map<std::string,int>::iterator iter=cNum.find(capp);
		  if (iter!=cNum.end()){
		    if(iter->second == 0) LogDebug("ClusterThr") << "Filling with a zero for " << capp << std::endl;
		    ((TH1F*) Hlist->FindObject("cNum"+TString(capp)))->Fill(iter->second);
		    iter->second=0;
		  }
		}
	  
	}
	
	delete SiStripClusterInfo_; 
    }
  }
}
//------------------------------------------------------------------------
    bool ClusterThr::clusterizer(SiStripClusterInfo* siStripClusterInfo,float Thc,float Ths,float Thn,bool& passedSeed,bool& passedClus,char* flag,const LocalVector LV){

      edm::LogInfo("ClusterThr") << "Clusterizer begin for " << flag << std::endl;
      //takes the parameters for max Background threshold and the min Signal threshold to define the overlap region
      const edm::ParameterSet StoNThr_ = conf_.getParameter<edm::ParameterSet>("StoNThr");
      double StoNBmax_ = StoNThr_.getParameter<double>("StoNBmax");
      double StoNSmin_ = StoNThr_.getParameter<double>("StoNSmin");
      
      //takes the noises of the strips involved in the cluster   
      const std::vector<float>&  stripNoises_ = siStripClusterInfo->stripNoisesRescaledByGain();
      
      //Clusterizer Selection
      
      // if the max charge of the cluster is less than the SeedThr*NoiseOfTheMaxChargeStrip, it goes out of the loop
      if(siStripClusterInfo->maxCharge()<Ths*stripNoises_[siStripClusterInfo->maxStrip()-siStripClusterInfo->firstStrip()]){
	passedSeed=false;
	edm::LogInfo("ClusterThr") << "Exit from loop due to Seed at " << Thc << " " << Ths << " " << Thn << std::endl;
	return false;
      }

      edm::LogInfo("ClusterThr") << "Seed threshold ok at " << Thc << " " << Ths << " " << Thn << std::endl;

      //initialization of the variables       
      float Signal=0;
      float Noise=0;
      float Nstrip=0;
      float Pos=0;

      //takes the amplitudes of the strips of the cluster
      const std::vector<uint8_t>&  stripAmplitudes_ = siStripClusterInfo->stripCharges();
     
      for (size_t istrip=0;istrip<stripAmplitudes_.size();istrip++){

	//if the amplitude of the strip is less than the NeighbourThr*NoiseOfTheStrip it goes out
	if (stripAmplitudes_[istrip]<Thn*stripNoises_[istrip])
	  continue;
	if(stripAmplitudes_[istrip]<2*stripNoises_[istrip]) edm::LogInfo("ClusterThr") << "Under digi threshold??" << std::endl;
	Signal+=stripAmplitudes_[istrip];//increase Signal with the sum of the amplitudes of all the strips of the cluster
	Noise+=stripNoises_[istrip]*stripNoises_[istrip];//increase Noise with the sum of the strip noise squared
	Nstrip++;//increase the strip counter
	Pos+=istrip*stripAmplitudes_[istrip];//increase Pos with ??????
      }
      
      float NoiseNorm=0;//initialize Normalized Noise
      //different ways of calculating noise
      if (NoiseMode_==0){
	Noise=sqrt(Noise/Nstrip);
	NoiseNorm=Noise;
      }else if(NoiseMode_==1){
	Noise=sqrt(Noise);
	NoiseNorm=Noise/sqrt(Nstrip);
      }else{
	Noise=sqrt(Noise);
	NoiseNorm=stripNoises_[siStripClusterInfo->maxStrip()-siStripClusterInfo->firstStrip()];
      }

      float cosRZ = -2;
      LogTrace("SiStripMonitorTrack")<< "\n\tLV " << LV.x() << " " << LV.y() << " " << LV.z() << " " << LV.mag() << std::endl;
      if (LV.mag()!=0){
	cosRZ= fabs(LV.z())/LV.mag();
	LogTrace("SiStripMonitorTrack")<< "\n\t cosRZ " << cosRZ << std::endl;
      }
      
      float StoN=Signal/NoiseNorm;
      float StoNCorr=StoN*cosRZ;
      Pos/=Signal;
      Pos+=siStripClusterInfo->firstStrip();
      //Cluster
      if (Signal < Thc * Noise){
	passedClus=false;
	edm::LogInfo("ClusterThr") << "Exit from loop due to Clus at " << Thc << " " << Ths << " " << Thn << std::endl;
	return false;
      }

      edm::LogInfo("ClusterThr") << "Cluster Threshold ok at " << Thc << " " << Ths << " " << Thn << std::endl;
	
      if (StoN>StoNBmax_ && StoN<StoNSmin_)
 	return true; //doesn't fill histos if StoN in overlap region btw Signal and Background
      
      if (StoN>StoNBmax_) {
	edm::LogInfo("ClusterThr") << "Signal cluster" << std::endl;
      }else{
	edm::LogInfo("ClusterThr") << "Background cluster" << std::endl;
      }

      edm::LogInfo("ClusterThr") << "Signal " << Signal << " Noise " << Noise << " Nstrip " << Nstrip << " StoN " << StoN <<  " StoNCorr " << StoNCorr << std::endl;
      
      char capp[128];
      sprintf(capp,"_%s_Th_%2.1f_%2.1f_%2.1f_%s",StoN>StoNBmax_?"S":"B",Thc,Ths,Thn,flag == "OnTrack"?"On":"Off");
      TString app(capp);
      cNum[capp]++;//increase cNum for this cluster and for this set of thresholds
            
      edm::LogInfo("ClusterThr") << "cNum just increased for " << capp << " " << cNum[capp] << std::endl;

      try{  
	if(flag=="OnTrack"){  
	  edm::LogInfo("ClusterThr") << "Flag is " << flag << std::endl;
	  
	  ((TH1F*) Hlist->FindObject("cSignal"+app))->Fill(Signal);
	  edm::LogInfo("ClusterThr") << "Filled " << app << " on track" << std::endl;
	  
	  ((TH1F*) Hlist->FindObject("cNoise" +app))->Fill(NoiseNorm);
	  ((TH1F*) Hlist->FindObject("cStoN"  +app))->Fill(StoN);
	  ((TH1F*) Hlist->FindObject("cStoNCorr"  +app))->Fill(StoNCorr);
	  ((TH1F*) Hlist->FindObject("cWidth" +app))->Fill(Nstrip);
	  ((TH1F*) Hlist->FindObject("cPos"   +app))->Fill(Pos);
	}else{
	  edm::LogInfo("ClusterThr") << "Flag is " << flag << " and " << app << std::endl;
	  
	  ((TH1F*) Hlist->FindObject("cSignal"+app))->Fill(Signal);
	  ((TH1F*) Hlist->FindObject("cNoise" +app))->Fill(NoiseNorm);
	  ((TH1F*) Hlist->FindObject("cStoN"  +app))->Fill(StoN);
	  ((TH1F*) Hlist->FindObject("cWidth" +app))->Fill(Nstrip);
	  ((TH1F*) Hlist->FindObject("cPos"   +app))->Fill(Pos);
	}
	
      }catch(cms::Exception& e){
	edm::LogError("ClusterThr") << "[ClusterThr::fillHistos]  cms::Exception:  DetName " << e.what() ;
      }
      
      edm::LogInfo("ClusterThr") << "Returning" << std::endl;
      
      return true;
    }

  //---------------------------------------------------------------------------------------------

  void ClusterThr::bookHlist(char* ParameterSetLabel, TFileDirectory subDir, TString & HistoName, char* xTitle){
    Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
    TH1F* p = subDir.make<TH1F>(HistoName,HistoName,
				Parameters.getParameter<int32_t>("Nbinx"),
				Parameters.getParameter<double>("xmin"),
				Parameters.getParameter<double>("xmax")
				);
    if ( xTitle != "" )
      p->SetXTitle(xTitle);
    Hlist->Add(p);

  }
}
