// -*- C++ -*-
//
// Package:    PFTracking
// Class:      GoodSeedProducer
// 
// Original Author:  Michele Pioppi
// March 2010. F. Beaudette. Produce PreId information


#include "RecoParticleFlow/PFTracking/interface/GoodSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"  
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <fstream>
#include <string>
#include "TMath.h"
#include "Math/VectorUtil.h"

using namespace edm;
using namespace std;
using namespace reco;

GoodSeedProducer::GoodSeedProducer(const ParameterSet& iConfig):
  pfTransformer_(nullptr),
  conf_(iConfig),
  resMapEtaECAL_(nullptr),
  resMapPhiECAL_(nullptr),
  reader(nullptr)
{
  LogInfo("GoodSeedProducer")<<"Electron PreIdentification started  ";
  
  //now do what ever initialization is needed
 
  tracksContainers_ = 
    iConfig.getParameter< vector < InputTag > >("TkColList");
  
  minPt_=iConfig.getParameter<double>("MinPt");
  maxPt_=iConfig.getParameter<double>("MaxPt");
  maxEta_=iConfig.getParameter<double>("MaxEta");


  //ISOLATION REQUEST AS DONE IN THE TAU GROUP
  applyIsolation_ =iConfig.getParameter<bool>("ApplyIsolation");
  HcalIsolWindow_                       =iConfig.getParameter<double>("HcalWindow");
  EcalStripSumE_minClusEnergy_ = iConfig.getParameter<double>("EcalStripSumE_minClusEnergy");
  EcalStripSumE_deltaEta_ = iConfig.getParameter<double>("EcalStripSumE_deltaEta");
  EcalStripSumE_deltaPhiOverQ_minValue_ = iConfig.getParameter<double>("EcalStripSumE_deltaPhiOverQ_minValue");
  EcalStripSumE_deltaPhiOverQ_maxValue_ = iConfig.getParameter<double>("EcalStripSumE_deltaPhiOverQ_maxValue");
   minEoverP_= iConfig.getParameter<double>("EOverPLead_minValue");
   maxHoverP_= iConfig.getParameter<double>("HOverPLead_maxValue");
 
  //
  pfCLusTagECLabel_=
    iConfig.getParameter<InputTag>("PFEcalClusterLabel");

  pfCLusTagHCLabel_=
   iConfig.getParameter<InputTag>("PFHcalClusterLabel");  

  pfCLusTagPSLabel_=
    iConfig.getParameter<InputTag>("PFPSClusterLabel");
  
  preidgsf_ = iConfig.getParameter<string>("PreGsfLabel");
  preidckf_ = iConfig.getParameter<string>("PreCkfLabel");
  preidname_= iConfig.getParameter<string>("PreIdLabel");
  
  
  fitterName_ = iConfig.getParameter<string>("Fitter");
  smootherName_ = iConfig.getParameter<string>("Smoother");
  
  
  nHitsInSeed_=iConfig.getParameter<int>("NHitsInSeed");

  clusThreshold_=iConfig.getParameter<double>("ClusterThreshold");
  
  minEp_=iConfig.getParameter<double>("MinEOverP");
  maxEp_=iConfig.getParameter<double>("MaxEOverP");

  //collection to produce
  produceCkfseed_ = iConfig.getUntrackedParameter<bool>("ProduceCkfSeed",false);

  // to disable the electron part (for HI collisions for examples) 
  disablePreId_ = iConfig.getUntrackedParameter<bool>("DisablePreId",false);  

  producePreId_ = iConfig.getUntrackedParameter<bool>("ProducePreId",true);  
  //  if no electron, cannot produce the preid
  if(disablePreId_) 
    producePreId_=false;
  PtThresholdSavePredId_ = iConfig.getUntrackedParameter<double>("PtThresholdSavePreId",1.);  

  LogDebug("GoodSeedProducer")<<"Seeds for GSF will be produced ";

  // no disablePreId_ switch here. The collection will be empty if it is true
  produces<ElectronSeedCollection>(preidgsf_);

  if(produceCkfseed_){
    LogDebug("GoodSeedProducer")<<"Seeds for CKF will be produced ";
    produces<TrajectorySeedCollection>(preidckf_);
  }
  

  if(producePreId_){
    LogDebug("GoodSeedProducer")<<"PreId debugging information will be produced ";

    produces<PreIdCollection>(preidname_);
    if(tracksContainers_.size()==1) // do not make a value map if more than one input track collection
      produces<edm::ValueMap<reco::PreIdRef> >(preidname_);
  } 
  
  useQuality_   = iConfig.getParameter<bool>("UseQuality");
  trackQuality_=TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));

  useTmva_= iConfig.getUntrackedParameter<bool>("UseTMVA",false);
  
  usePreshower_ = iConfig.getParameter<bool>("UsePreShower");

}


GoodSeedProducer::~GoodSeedProducer()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.) 

  delete pfTransformer_;
  delete resMapEtaECAL_;
  delete resMapPhiECAL_;
  if(useTmva_) {
    delete reader;
  }
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GoodSeedProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  LogDebug("GoodSeedProducer")<<"START event: "<<iEvent.id().event()
			      <<" in run "<<iEvent.id().run();
  //Create empty output collections
  auto_ptr<ElectronSeedCollection> output_preid(new ElectronSeedCollection);
  auto_ptr<TrajectorySeedCollection> output_nopre(new TrajectorySeedCollection);
  auto_ptr<PreIdCollection> output_preidinfo(new PreIdCollection);
  auto_ptr<edm::ValueMap<reco::PreIdRef> > preIdMap_p(new edm::ValueMap<reco::PreIdRef>);
  edm::ValueMap<reco::PreIdRef>::Filler mapFiller(*preIdMap_p);

  //Tracking Tools
  if(!disablePreId_)
    {
      iSetup.get<TrajectoryFitter::Record>().get(fitterName_, fitter_);
      iSetup.get<TrajectoryFitter::Record>().get(smootherName_, smoother_);
    }

  // clear temporary maps
  refMap_.clear();

  //Handle input collections
  //ECAL clusters	      
  Handle<PFClusterCollection> theECPfClustCollection;
  iEvent.getByLabel(pfCLusTagECLabel_,theECPfClustCollection);
  
  vector<PFCluster> basClus;
  vector<PFCluster>::const_iterator iklus;
  for (iklus=theECPfClustCollection.product()->begin();
       iklus!=theECPfClustCollection.product()->end();
       iklus++){
    if((*iklus).energy()>clusThreshold_) basClus.push_back(*iklus);
  }

  //HCAL clusters
  Handle<PFClusterCollection> theHCPfClustCollection;
  iEvent.getByLabel(pfCLusTagHCLabel_,theHCPfClustCollection);
  
  //PS clusters
  Handle<PFClusterCollection> thePSPfClustCollection;
  iEvent.getByLabel(pfCLusTagPSLabel_,thePSPfClustCollection);
  
  ps1Clus.clear();
  ps2Clus.clear();
  
  for (iklus=thePSPfClustCollection.product()->begin();
       iklus!=thePSPfClustCollection.product()->end();
       iklus++){
    //layer==-11 first layer of PS
    //layer==-12 secon layer of PS
    if ((*iklus).layer()==-11) ps1Clus.push_back(*iklus);
    if ((*iklus).layer()==-12) ps2Clus.push_back(*iklus);
  }
  
  //Vector of track collections
  for (unsigned int istr=0; istr<tracksContainers_.size();istr++){
    
    //Track collection
    Handle<TrackCollection> tkRefCollection;
    iEvent.getByLabel(tracksContainers_[istr], tkRefCollection);
    const TrackCollection&  Tk=*(tkRefCollection.product());
    
    //Trajectory collection
    Handle<vector<Trajectory> > tjCollection;
    iEvent.getByLabel(tracksContainers_[istr], tjCollection);
    vector<Trajectory> Tj=*(tjCollection.product());
    
    
    LogDebug("GoodSeedProducer")<<"Number of tracks in collection "
                                <<tracksContainers_[istr] <<" to be analyzed "
                                <<Tj.size();
    

    //loop over the track collection
    for(unsigned int i=0;i<Tk.size();i++){		
      if (useQuality_ &&
	  (!(Tk[i].quality(trackQuality_)))) continue;
      
      reco::PreId myPreId;
      bool GoodPreId=false;

      TrackRef trackRef(tkRefCollection, i);
      // TrajectorySeed Seed=Tj[i].seed();
      TrajectorySeed Seed=(*trackRef->seedRef());
      if(!disablePreId_)
	{
	  int ipteta=getBin(Tk[i].eta(),Tk[i].pt());
	  int ibin=ipteta*8;
	  
	  float PTOB=Tj[i].lastMeasurement().updatedState().globalMomentum().mag();
	  float chikfred=Tk[i].normalizedChi2();
	  int nhitpi=Tj[i].foundHits();
	  float EP=0;
      
	  // set track info
	  myPreId.setTrack(trackRef);
	  //CLUSTERS - TRACK matching
      
	  float pfmass=  0.0005;
	  float pfoutenergy=sqrt((pfmass*pfmass)+Tk[i].outerMomentum().Mag2());
	  XYZTLorentzVector mom =XYZTLorentzVector(Tk[i].outerMomentum().x(),
						   Tk[i].outerMomentum().y(),
						   Tk[i].outerMomentum().z(),
						   pfoutenergy);
	  XYZTLorentzVector pos =   XYZTLorentzVector(Tk[i].outerPosition().x(),
						      Tk[i].outerPosition().y(),
						      Tk[i].outerPosition().z(),
						      0.);

	  BaseParticlePropagator theOutParticle = 
	    BaseParticlePropagator( RawParticle(mom,pos),
				    0,0,B_.z());
	  theOutParticle.setCharge(Tk[i].charge());
      
	  theOutParticle.propagateToEcalEntrance(false);
      
      
	  float toteta=1000;
	  float totphi=1000;
	  float dr=1000;
	  float EE=0;
	  float feta=0;
	  math::XYZPointF ElecTrkEcalPos(0,0,0);
	  PFClusterRef clusterRef;
	  math::XYZPoint meanShowerSaved;
	  if(theOutParticle.getSuccess()!=0){
	    ElecTrkEcalPos=math::XYZPointF(theOutParticle.vertex().x(),
					   theOutParticle.vertex().y(),
					   theOutParticle.vertex().z());
	    bool isBelowPS=(fabs(theOutParticle.vertex().eta())>1.65) ? true :false;	
	
	    unsigned clusCounter=0;

	    for(vector<PFCluster>::const_iterator aClus = basClus.begin();
		aClus != basClus.end(); aClus++,++clusCounter) {
	  
	      double ecalShowerDepth
		= PFCluster::getDepthCorrection(aClus->energy(),
						isBelowPS,
						false);
	
	      math::XYZPoint meanShower=math::XYZPoint(theOutParticle.vertex())+
		math::XYZTLorentzVector(theOutParticle.momentum()).Vect().Unit()*ecalShowerDepth;	
	  
	      float etarec=meanShower.eta();
	      float phirec=meanShower.phi();
	      float tmp_ep=aClus->energy()/PTOB;
	      float tmp_phi=fabs(aClus->position().phi()-phirec);
	      if (tmp_phi>TMath::TwoPi()) tmp_phi-= TMath::TwoPi();
	      float tmp_dr=sqrt(pow(tmp_phi,2)+
				pow((aClus->position().eta()-etarec),2));
	  
	      if ((tmp_dr<dr)&&(tmp_ep>minEp_)&&(tmp_ep<maxEp_)){
		dr=tmp_dr;
		toteta=aClus->position().eta()-etarec;
		totphi=tmp_phi;
		EP=tmp_ep;
		EE=aClus->energy();
		feta= aClus->position().eta();
		clusterRef = PFClusterRef(theECPfClustCollection,clusCounter);
		meanShowerSaved = meanShower;
	      }
	    }
	  }

	  //Resolution maps
	  double ecaletares 
	    = resMapEtaECAL_->GetBinContent(resMapEtaECAL_->FindBin(feta,EE));
	  double ecalphires 
	    = resMapPhiECAL_->GetBinContent(resMapPhiECAL_->FindBin(feta,EE)); 
      
	  //geomatrical compatibility
	  float chieta=(toteta!=1000)? toteta/ecaletares : toteta;
	  float chiphi=(totphi!=1000)? totphi/ecalphires : totphi;
	  float chichi= sqrt(chieta*chieta + chiphi*chiphi);
      
      
	  //Matching criteria
	  float chi2cut=thr[ibin+0];
	  float ep_cutmin=thr[ibin+1];
	  bool GoodMatching= ((chichi<chi2cut) &&(EP>ep_cutmin) && (nhitpi>10));
	  bool EcalMatching=GoodMatching;
      
	  if (Tk[i].pt()>maxPt_) GoodMatching=true;
	  if (Tk[i].pt()<minPt_) GoodMatching=false;

	  //ENDCAP
	  //USE OF PRESHOWER 
	  bool GoodPSMatching=false;
	  if ((fabs(Tk[i].eta())>1.68)&&(usePreshower_)){
	    int iptbin =4*getBin(Tk[i].pt());
	    ps2En=0;ps1En=0;
	    ps2chi=100.; ps1chi=100.;
	    PSforTMVA(mom,pos);
	    float p1e=thrPS[iptbin];
	    float p2e=thrPS[iptbin+1];
	    float p1c=thrPS[iptbin+2];
	    float p2c=thrPS[iptbin+3];
	    GoodPSMatching= 
	      ((ps2En>p2e)
	       &&(ps1En>p1e)
	       &&(ps1chi<p1c)
	       &&(ps2chi<p2c));
	    GoodMatching = (GoodMatching && GoodPSMatching);
	  }
  
	  math::XYZPoint myPoint(ElecTrkEcalPos.X(),ElecTrkEcalPos.Y(),ElecTrkEcalPos.Z());
	  myPreId.setECALMatchingProperties(clusterRef,myPoint,meanShowerSaved,toteta,totphi,chieta,
					    chiphi,chichi,EP);
	  myPreId.setECALMatching(EcalMatching);
	  myPreId.setESMatching(GoodPSMatching);

	  if(applyIsolation_){
	    if(IsIsolated(float(Tk[i].charge()),Tk[i].p(),
			  ElecTrkEcalPos,*theECPfClustCollection,*theHCPfClustCollection)) 
	      GoodMatching=true;
	  }
	  bool GoodRange= ((fabs(Tk[i].eta())<maxEta_) && 
			   (Tk[i].pt()>minPt_));
	  //KF FILTERING FOR UNMATCHED EVENTS
	  int hit1max=int(thr[ibin+2]);
	  float chiredmin=thr[ibin+3];
	  bool GoodKFFiltering =
	    ((chikfred>chiredmin) || (nhitpi<hit1max));
      
	  myPreId.setTrackFiltering(GoodKFFiltering);

	  bool GoodTkId= false;
      
	  if((!GoodMatching) &&(GoodKFFiltering) &&(GoodRange)){
	    chi=chichi;
	    chired=1000;
	    chiRatio=1000;
	    dpt=0;
	    nhit=nhitpi;
      
	    Trajectory::ConstRecHitContainer tmp;
	    Trajectory::ConstRecHitContainer hits=Tj[i].recHits();
	    for (int ih=hits.size()-1; ih>=0; ih--)  tmp.push_back(hits[ih]);
	    vector<Trajectory> FitTjs=(fitter_.product())->fit(Seed,tmp,Tj[i].lastMeasurement().updatedState());
	
	    if(FitTjs.size()>0){
	      if(FitTjs[0].isValid()){
		vector<Trajectory> SmooTjs=(smoother_.product())->trajectories(FitTjs[0]);
		if(SmooTjs.size()>0){
		  if(SmooTjs[0].isValid()){
		
		    //Track refitted with electron hypothesis
		
		    float pt_out=SmooTjs[0].firstMeasurement().
		      updatedState().globalMomentum().perp();
		    float pt_in=SmooTjs[0].lastMeasurement().
		      updatedState().globalMomentum().perp();
		    dpt=(pt_in>0) ? fabs(pt_out-pt_in)/pt_in : 0.;
		    // the following is simply the number of degrees of freedom
		    chiRatio=SmooTjs[0].chiSquared()/Tj[i].chiSquared();
		    chired=chiRatio*chikfred;
		  }
		}
	      }
	    }
	
	    //TMVA Analysis
	    if(useTmva_){
	
	      eta=Tk[i].eta();
	      pt=Tk[i].pt();
	      eP=EP;
	  
	      float Ytmva=reader->EvaluateMVA( method_ );
	      
	      float BDTcut=thr[ibin+4]; 
	      if ( Ytmva>BDTcut) GoodTkId=true;
	      myPreId.setMVA(GoodTkId,Ytmva);
	      myPreId.setTrackProperties(chired,chiRatio,dpt);
	    }else{ 
	  	 	  
	      //
	      float chiratiocut=thr[ibin+5]; 
	      float gschicut=thr[ibin+6]; 
	      float gsptmin=thr[ibin+7];

	      GoodTkId=((dpt>gsptmin)&&(chired<gschicut)&&(chiRatio<chiratiocut));      
       
	    }
	  }
    
	  GoodPreId= (GoodTkId || GoodMatching); 

	  myPreId.setFinalDecision(GoodPreId);
      
	  if(GoodPreId)
	    LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	      "GeV/c, eta= "<<Tk[i].eta() <<
	      ") preidentified for agreement between  track and ECAL cluster";
	  if(GoodPreId &&(!GoodMatching))
	    LogDebug("GoodSeedProducer")<<"Track (pt= "<<Tk[i].pt()<<
	      "GeV/c, eta= "<<Tk[i].eta() <<
	      ") preidentified only for track properties";
	
	} // end of !disablePreId_
      
      if (GoodPreId){

	//NEW SEED with n hits	
	ElectronSeed NewSeed(Seed);
	NewSeed.setCtfTrack(trackRef);
	output_preid->push_back(NewSeed);
      }else{
	if (produceCkfseed_){
	  output_nopre->push_back(Seed);
	}
      }
      if(producePreId_ && myPreId.pt()>PtThresholdSavePredId_)
	{
	  // save the index of the PreId object as to be able to create a Ref later
	  refMap_[trackRef] = output_preidinfo->size();
	  output_preidinfo->push_back(myPreId);	  
	}
    } //end loop on track collection
  } //end loop on the vector of track collections
  
  // no disablePreId_ switch, it is simpler to have an empty collection rather than no collection
  iEvent.put(output_preid,preidgsf_);
  if (produceCkfseed_)
    iEvent.put(output_nopre,preidckf_);
  if(producePreId_)
    {
      const edm::OrphanHandle<reco::PreIdCollection> preIdRefProd = iEvent.put(output_preidinfo,preidname_);
      // now make the Value Map, but only if one input collection
      if(tracksContainers_.size()==1)
	{
	  Handle<TrackCollection> tkRefCollection ;
	  iEvent.getByLabel(tracksContainers_[0],tkRefCollection);
	  fillPreIdRefValueMap(tkRefCollection,preIdRefProd,mapFiller);
	  mapFiller.fill();
	  iEvent.put(preIdMap_p,preidname_);
	}
    }

   // clear temporary maps
  refMap_.clear();

}
// ------------ method called once each job just before starting event loop  ------------
void 
GoodSeedProducer::beginRun(const edm::Run & run,
			   const EventSetup& es)
{
  //Magnetic Field
  ESHandle<MagneticField> magneticField;
  es.get<IdealMagneticFieldRecord>().get(magneticField);
  B_=magneticField->inTesla(GlobalPoint(0,0,0));
  
  pfTransformer_= new PFTrackTransformer(B_);
  pfTransformer_->OnlyProp();


  
  //Resolution maps
  FileInPath ecalEtaMap(conf_.getParameter<string>("EtaMap"));
  FileInPath ecalPhiMap(conf_.getParameter<string>("PhiMap"));
  resMapEtaECAL_ = new PFResolutionMap("ECAL_eta",ecalEtaMap.fullPath().c_str());
  resMapPhiECAL_ = new PFResolutionMap("ECAL_phi",ecalPhiMap.fullPath().c_str());

  if(useTmva_){
    reader = new TMVA::Reader("!Color:Silent");
    method_ = conf_.getParameter<string>("TMVAMethod");
    
    reader->AddVariable("eP",&eP);
    reader->AddVariable("chi",&chi);
    reader->AddVariable("chired",&chired);
    reader->AddVariable("chiRatio",&chiRatio);
    reader->AddVariable("dpt",&dpt);
    reader->AddVariable("nhit",&nhit);
    reader->AddVariable("eta",&eta);
    reader->AddVariable("pt",&pt);
    FileInPath Weigths(conf_.getParameter<string>("Weights"));
    reader->BookMVA( method_, Weigths.fullPath().c_str()  );	
    }

    
    //read threshold
    FileInPath parFile(conf_.getParameter<string>("ThresholdFile"));
    ifstream ifs(parFile.fullPath().c_str());
    for (int iy=0;iy<72;iy++) ifs >> thr[iy];
    
    //read PS threshold
    FileInPath parPSFile(conf_.getParameter<string>("PSThresholdFile"));
    ifstream ifsPS(parPSFile.fullPath().c_str());
    for (int iy=0;iy<12;iy++) ifsPS >> thrPS[iy];

}

void 
GoodSeedProducer::endRun(const edm::Run &, const edm::EventSetup&) {
  delete pfTransformer_;
  pfTransformer_ = nullptr;
  delete resMapEtaECAL_;
  resMapEtaECAL_ = nullptr;
  delete resMapPhiECAL_;
  resMapPhiECAL_ = nullptr;
  if(useTmva_) {
    delete reader;
    reader = nullptr;
  }
}

int 
GoodSeedProducer::getBin(float pt){
int ip=0;
  if (pt<6) ip=0;
  else {  if (pt<12) ip=1;
        else ip=2;
  }
return ip;
}

int 
GoodSeedProducer::getBin(float eta, float pt){
  int ie=0;
  int ip=0;
  if (fabs(eta)<1.2) ie=0;
  else{ if (fabs(eta)<1.68) ie=1;
    else ie=2;
  }
  if (pt<6) ip=0;
  else {  if (pt<12) ip=1;     
	else ip=2;
  }
  int iep= ie*3+ip;
  LogDebug("GoodSeedProducer")<<"Track pt ="<<pt<<" eta="<<eta<<" bin="<<iep;
  return iep;
}

void 
GoodSeedProducer::PSforTMVA(const XYZTLorentzVector& mom,const XYZTLorentzVector& pos ){

  BaseParticlePropagator OutParticle(RawParticle(mom,pos)
				     ,0.,0.,B_.z()) ;

  OutParticle.propagateToPreshowerLayer1(false);
  if (OutParticle.getSuccess()!=0){
    //   GlobalPoint v1=ps1TSOS.globalPosition();
    math::XYZPoint v1=math::XYZPoint(OutParticle.vertex());
    if ((v1.Rho() >=
	 PFGeometry::innerRadius(PFGeometry::PS1)) &&
	(v1.Rho() <=
	 PFGeometry::outerRadius(PFGeometry::PS1))) {
      float enPScl1=0;
      float chi1=100;
      vector<PFCluster>::const_iterator ips;
      for (ips=ps1Clus.begin(); ips!=ps1Clus.end();ips++){
	float ax=((*ips).position().x()-v1.x())/0.114;
	float ay=((*ips).position().y()-v1.y())/2.43;
	float pschi= sqrt(ax*ax+ay*ay);
	if (pschi<chi1){
	  chi1=pschi;
	  enPScl1=(*ips).energy();
	}
      }
      ps1En=enPScl1;
      ps1chi=chi1;


      OutParticle.propagateToPreshowerLayer2(false);
      if (OutParticle.getSuccess()!=0){
	math::XYZPoint v2=math::XYZPoint(OutParticle.vertex());
	if ((v2.Rho() >=
	     PFGeometry::innerRadius(PFGeometry::PS2)) &&
	    (v2.Rho() <=
	     PFGeometry::outerRadius(PFGeometry::PS2))){
	  float enPScl2=0;
	  float chi2=100;
	  for (ips=ps2Clus.begin(); ips!=ps2Clus.end();ips++){
	    float ax=((*ips).position().x()-v2.x())/1.88;
	    float ay=((*ips).position().y()-v2.y())/0.1449;
	    float pschi= sqrt(ax*ax+ay*ay);
	    if (pschi<chi2){
	      chi2=pschi;
	      enPScl2=(*ips).energy();
	    }
	  }

	  ps2En=enPScl2;
	  ps2chi=chi2;

	}
      }
      
    }
    
  }
}

bool 
GoodSeedProducer::IsIsolated(float charge, float P,
			     math::XYZPointF myElecTrkEcalPos,
			     const PFClusterCollection &ecalColl,
			     const PFClusterCollection &hcalColl){


  double myHCALenergy3x3=0.;
  double myStripClusterE=0.;
 

  //  reco::TrackRef myElecTrk;
  
  if (fabs(myElecTrkEcalPos.z())<1. && myElecTrkEcalPos.x()<1. && myElecTrkEcalPos.y()<1. ) return false; 

  
  
  PFClusterCollection::const_iterator hc=hcalColl.begin();
  PFClusterCollection::const_iterator hcend=hcalColl.end();
  for (;hc!=hcend;++hc){
    math::XYZPoint clusPos = hc->position();
    double en = hc->energy();
    double deltaR = ROOT::Math::VectorUtil::DeltaR(myElecTrkEcalPos,clusPos);
    if (deltaR<HcalIsolWindow_) {
      myHCALenergy3x3 += en;
      
    }
  }



  PFClusterCollection::const_iterator ec=ecalColl.begin();
  PFClusterCollection::const_iterator ecend=ecalColl.end();
  for (;ec!=ecend;++ec){
    math::XYZPoint clusPos = ec->position();
    double en = ec->energy();


    double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(myElecTrkEcalPos,clusPos);
    double deltaEta = abs(myElecTrkEcalPos.eta()-clusPos.eta());
    double deltaPhiOverQ = deltaPhi/charge;
    if (en >= EcalStripSumE_minClusEnergy_ && deltaEta<EcalStripSumE_deltaEta_ && deltaPhiOverQ > EcalStripSumE_deltaPhiOverQ_minValue_ && deltaPhiOverQ < EcalStripSumE_deltaPhiOverQ_maxValue_) { 
      myStripClusterE += en;
    }
  }	  
  
  double EoP=myStripClusterE/P;
  double HoP=myHCALenergy3x3/P;


  return ((EoP>minEoverP_)&&(EoP<2.5) && (HoP<maxHoverP_))?true:false;
}

void GoodSeedProducer::fillPreIdRefValueMap( Handle<TrackCollection> tracks,
					     const edm::OrphanHandle<reco::PreIdCollection>& preidhandle,
					     edm::ValueMap<reco::PreIdRef>::Filler & filler)
{
  std::vector<reco::PreIdRef> values;

  unsigned ntracks=tracks->size();
  for(unsigned itrack=0;itrack<ntracks;++itrack)
   {
     reco::TrackRef theTrackRef(tracks,itrack);
     std::map<reco::TrackRef,unsigned>::const_iterator itcheck=refMap_.find(theTrackRef);
     if(itcheck==refMap_.end()) 
       {
	 // the track has been early discarded
	 values.push_back(reco::PreIdRef());
       }
     else
       {
	 edm::Ref<reco::PreIdCollection> preIdRef(preidhandle,itcheck->second);
	 values.push_back(preIdRef);
	 //	 std::cout << " Checking Refs " << (theTrackRef==preIdRef->trackRef()) << std::endl;
       }
   }
  filler.insert(tracks,values.begin(),values.end());
}
