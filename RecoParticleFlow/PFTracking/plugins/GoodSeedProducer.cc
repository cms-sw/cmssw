// -*- C++ -*-
//
// Package:    PFTracking
// Class:      GoodSeedProducer
// 
// Original Author:  Michele Pioppi
// March 2010. F. Beaudette. Produce PreId information

#include "RecoParticleFlow/PFTracking/plugins/GoodSeedProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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

#include "DataFormats/Math/interface/deltaR.h"

#include <fstream>
#include <string>
#include "TMath.h"
#include "Math/VectorUtil.h"
#include "TMVA/MethodBDT.h"

using namespace edm;
using namespace std;
using namespace reco;

GoodSeedProducer::GoodSeedProducer(const ParameterSet& iConfig, const goodseedhelpers::HeavyObjectCache*):
  pfTransformer_(nullptr),
  conf_(iConfig),
  resMapEtaECAL_(nullptr),
  resMapPhiECAL_(nullptr)
{
  LogInfo("GoodSeedProducer")<<"Electron PreIdentification started  ";
  
  //now do what ever initialization is needed
  std::vector<edm::InputTag> tags =   iConfig.getParameter< vector < InputTag > >("TkColList");
  for(unsigned int i=0;i<tags.size();++i) {
    trajContainers_.push_back(consumes<vector<Trajectory> >(tags[i]));
    tracksContainers_.push_back(consumes<reco::TrackCollection>(tags[i]));
  }
  
  minPt_=iConfig.getParameter<double>("MinPt");
  maxPt_=iConfig.getParameter<double>("MaxPt");
  maxEta_=iConfig.getParameter<double>("MaxEta");

  HcalIsolWindow_                       =iConfig.getParameter<double>("HcalWindow");
  EcalStripSumE_minClusEnergy_ = iConfig.getParameter<double>("EcalStripSumE_minClusEnergy");
  EcalStripSumE_deltaEta_ = iConfig.getParameter<double>("EcalStripSumE_deltaEta");
  EcalStripSumE_deltaPhiOverQ_minValue_ = iConfig.getParameter<double>("EcalStripSumE_deltaPhiOverQ_minValue");
  EcalStripSumE_deltaPhiOverQ_maxValue_ = iConfig.getParameter<double>("EcalStripSumE_deltaPhiOverQ_maxValue");
  minEoverP_= iConfig.getParameter<double>("EOverPLead_minValue");
  maxHoverP_= iConfig.getParameter<double>("HOverPLead_maxValue");
 
  pfCLusTagECLabel_=consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFEcalClusterLabel"));

  pfCLusTagHCLabel_=consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFHcalClusterLabel"));  

  pfCLusTagPSLabel_=consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFPSClusterLabel"));
  
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

  Min_dr_ = iConfig.getParameter<double>("Min_dr");

  trackerRecHitBuilderName_ = iConfig.getParameter<std::string>("TTRHBuilder");

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
      edm::ESHandle<TrajectoryFitter> aFitter;
      edm::ESHandle<TrajectorySmoother> aSmoother;
      iSetup.get<TrajectoryFitter::Record>().get(fitterName_, aFitter);
      iSetup.get<TrajectoryFitter::Record>().get(smootherName_, aSmoother);
      smoother_.reset(aSmoother->clone());
      fitter_ = aFitter->clone();
      edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;
      iSetup.get<TransientRecHitRecord>().get(trackerRecHitBuilderName_,theTrackerRecHitBuilder);
      hitCloner = static_cast<TkTransientTrackingRecHitBuilder const *>(theTrackerRecHitBuilder.product())->cloner();
      fitter_->setHitCloner(&hitCloner);
      smoother_->setHitCloner(&hitCloner);
    }

  // clear temporary maps
  refMap_.clear();

  //Handle input collections
  //ECAL clusters	      
  Handle<PFClusterCollection> theECPfClustCollection;
  iEvent.getByToken(pfCLusTagECLabel_,theECPfClustCollection);
  

 vector<PFCluster const *> basClus;
  for ( auto const & klus : *theECPfClustCollection.product() ) {
    if(klus.correctedEnergy()>clusThreshold_) basClus.push_back(&klus);
  }

  //HCAL clusters
  Handle<PFClusterCollection> theHCPfClustCollection;
  iEvent.getByToken(pfCLusTagHCLabel_,theHCPfClustCollection);
  
  //PS clusters
  Handle<PFClusterCollection> thePSPfClustCollection;
  iEvent.getByToken(pfCLusTagPSLabel_,thePSPfClustCollection);

  //Vector of track collections
  for (unsigned int istr=0; istr<tracksContainers_.size();++istr){
    
    //Track collection
    Handle<TrackCollection> tkRefCollection;
    iEvent.getByToken(tracksContainers_[istr], tkRefCollection);
    const TrackCollection&  Tk=*(tkRefCollection.product());
    
    //Trajectory collection
    Handle<vector<Trajectory> > tjCollection;
    iEvent.getByToken(trajContainers_[istr], tjCollection);
    auto const & Tj=*(tjCollection.product());
    
    LogDebug("GoodSeedProducer")<<"Number of tracks in collection "
                                <<tracksContainers_[istr] <<" to be analyzed "
                                <<Tj.size();

    //loop over the track collection
    for(unsigned int i=0;i<Tk.size();++i){		
      if (useQuality_ &&
	  (!(Tk[i].quality(trackQuality_)))) continue;
      
      reco::PreId myPreId;
      bool GoodPreId=false;

      TrackRef trackRef(tkRefCollection, i);
      // TrajectorySeed Seed=Tj[i].seed();
      math::XYZVectorF tkmom(Tk[i].momentum());
      auto tketa= tkmom.eta();
      auto tkpt = std::sqrt(tkmom.perp2());
      auto const & Seed=(*trackRef->seedRef());
      if(!disablePreId_)
	{
	  int ipteta=getBin(Tk[i].eta(),Tk[i].pt());
	  int ibin=ipteta*9;
	  
	  float oPTOB=1.f/Tj[i].lastMeasurement().updatedState().globalMomentum().mag();
	  //  float chikfred=Tk[i].normalizedChi2();
	  float nchi=Tk[i].normalizedChi2();

	  int nhitpi=Tj[i].foundHits();
	  float EP=0;
      
	  // set track info
	  myPreId.setTrack(trackRef);
	  //CLUSTERS - TRACK matching
      
	  auto pfmass=  0.0005;
	  auto pfoutenergy=sqrt((pfmass*pfmass)+Tk[i].outerMomentum().Mag2());

	  XYZTLorentzVector mom =XYZTLorentzVector(Tk[i].outerMomentum().x(),
						   Tk[i].outerMomentum().y(),
						   Tk[i].outerMomentum().z(),
						   pfoutenergy);
	  XYZTLorentzVector pos =   XYZTLorentzVector(Tk[i].outerPosition().x(),
						      Tk[i].outerPosition().y(),
						      Tk[i].outerPosition().z(),
						      0.);

	  BaseParticlePropagator theOutParticle( RawParticle(mom,pos),
				    0,0,B_.z());
	  theOutParticle.setCharge(Tk[i].charge());
      
	  theOutParticle.propagateToEcalEntrance(false);
      

      
	  float toteta=1000.f;
	  float totphi=1000.f;
	  float dr=1000.f;
	  float EE=0.f;
	  float feta=0.f;
	  GlobalPoint ElecTrkEcalPos(0,0,0);

	  PFClusterRef clusterRef;
	  math::XYZPoint meanShowerSaved;
	  if(theOutParticle.getSuccess()!=0){
	     ElecTrkEcalPos=GlobalPoint(theOutParticle.vertex().x(),
			       	        theOutParticle.vertex().y(),
					theOutParticle.vertex().z()
                                       );

            constexpr float psLim = 2.50746495928f; // std::sinh(1.65f);
            bool isBelowPS= (ElecTrkEcalPos.z()*ElecTrkEcalPos.z()) > (psLim*psLim)*ElecTrkEcalPos.perp2();
	    // bool isBelowPS=(std::abs(ElecTrkEcalPos.eta())>1.65f);	
	
	    unsigned clusCounter=0;
	    float max_ee = 0;
	    for(auto aClus : basClus) {

	      float tmp_ep=float(aClus->correctedEnergy())*oPTOB;
              if ((tmp_ep<minEp_)|(tmp_ep>maxEp_)) { ++clusCounter; continue;}
	    
	      double ecalShowerDepth
		= PFCluster::getDepthCorrection(aClus->correctedEnergy(),
						isBelowPS,
						false);
	      auto mom = theOutParticle.momentum().Vect();
	      auto meanShower = ElecTrkEcalPos +
		GlobalVector(mom.x(),mom.y(),mom.z()).unit()*ecalShowerDepth;	
	  
	      float etarec=meanShower.eta();
	      float phirec=meanShower.phi();
	     

	      float tmp_phi=std::abs(aClus->positionREP().phi()-phirec);
	      if (tmp_phi>float(TMath::Pi())) tmp_phi-= float(TMath::TwoPi());
	      
	      float tmp_dr=std::sqrt(std::pow(tmp_phi,2.f)+
				std::pow(aClus->positionREP().eta()-etarec,2.f));
	  
	      if (tmp_dr<dr){
		dr=tmp_dr;
		if(dr < Min_dr_){ // find the most closest and energetic ECAL cluster
		  if(aClus->correctedEnergy() > max_ee){

		    toteta=aClus->positionREP().eta()-etarec;
		    totphi=tmp_phi;
		    EP=tmp_ep;
		    EE=aClus->correctedEnergy();
		    feta= aClus->positionREP().eta();
		    clusterRef = PFClusterRef(theECPfClustCollection,clusCounter);
		    meanShowerSaved = meanShower;
		    
		  }
		}
	      }
              ++clusCounter;
	    }
	  }
	  float trk_ecalDeta_ = fabs(toteta);
	  float trk_ecalDphi_ = fabs(totphi);

	  //Resolution maps
	  auto ecaletares 
	    = resMapEtaECAL_->GetBinContent(resMapEtaECAL_->FindBin(feta,EE));
	  auto ecalphires 
	    = resMapPhiECAL_->GetBinContent(resMapPhiECAL_->FindBin(feta,EE)); 
      
	  //geomatrical compatibility
	  float chieta=(toteta!=1000.f)? toteta/ecaletares : toteta;
	  float chiphi=(totphi!=1000.f)? totphi/ecalphires : totphi;
	  float chichi= sqrt(chieta*chieta + chiphi*chiphi);
      
	  //Matching criteria
	  float eta_cut = thr[ibin+0];
	  float phi_cut = thr[ibin+1];
	  float ep_cutmin=thr[ibin+2];
	  bool GoodMatching= ((trk_ecalDeta_<eta_cut) && (trk_ecalDphi_<phi_cut) && (EP>ep_cutmin) && (nhitpi>10));

	  bool EcalMatching=GoodMatching;
      
	  if (tkpt>maxPt_) GoodMatching=true;
	  if (tkpt<minPt_) GoodMatching=false;


  
	  math::XYZPoint myPoint(ElecTrkEcalPos.x(),ElecTrkEcalPos.y(),ElecTrkEcalPos.z());
	  myPreId.setECALMatchingProperties(clusterRef,myPoint,meanShowerSaved,std::abs(toteta),std::abs(totphi),chieta,
					    chiphi,chichi,EP);
	  myPreId.setECALMatching(EcalMatching);


	  bool GoodRange= ((std::abs(tketa)<maxEta_) & 
			   (tkpt>minPt_));
	  //KF FILTERING FOR UNMATCHED EVENTS
	  int hit1max=int(thr[ibin+3]);
	  float chiredmin=thr[ibin+4];
	  bool GoodKFFiltering =
	    ((nchi>chiredmin) | (nhitpi<hit1max));
      

	  myPreId.setTrackFiltering(GoodKFFiltering);

	  bool GoodTkId= false;
      
	  if((!GoodMatching) &&(GoodKFFiltering) &&(GoodRange)){
	    chired=1000;
	    chiRatio=1000;
	    dpt=0;
	    nhit=nhitpi;
	    chikfred = nchi;
	    trk_ecalDeta = trk_ecalDeta_;
	    trk_ecalDphi = trk_ecalDphi_;
      
	    Trajectory::ConstRecHitContainer tmp;
	    Trajectory::ConstRecHitContainer && hits=Tj[i].recHits();
	    for (int ih=hits.size()-1; ih>=0; ih--)  tmp.push_back(hits[ih]);
	    Trajectory  && FitTjs= fitter_->fitOne(Seed,tmp,Tj[i].lastMeasurement().updatedState());
	
	      if(FitTjs.isValid()){
		Trajectory && SmooTjs= smoother_->trajectory(FitTjs);
		  if(SmooTjs.isValid()){
		
		    //Track refitted with electron hypothesis
		
		    float pt_out=SmooTjs.firstMeasurement().
		      updatedState().globalMomentum().perp();
		    float pt_in=SmooTjs.lastMeasurement().
		      updatedState().globalMomentum().perp();
		    dpt=(pt_in>0) ? fabs(pt_out-pt_in)/pt_in : 0.;
		    // the following is simply the number of degrees of freedom
		    chiRatio=SmooTjs.chiSquared()/Tj[i].chiSquared();
		    chired=chiRatio*chikfred;

		  }
		}
	     
	
	    //TMVA Analysis
	    if(useTmva_){
	
	      eta=tketa;
	      pt=tkpt;
	      eP=EP;
              float vars[10] = { nhit, chikfred, dpt, eP, chiRatio, chired, trk_ecalDeta, trk_ecalDphi, pt, eta};
              
	      float Ytmva = globalCache()->gbr[ipteta]->GetClassifier( vars );
	      
	      float BDTcut=thr[ibin+5]; 
	      if ( Ytmva>BDTcut) GoodTkId=true;
	      myPreId.setMVA(GoodTkId,Ytmva);
	      myPreId.setTrackProperties(chired,chiRatio,dpt);
	    }else{ 
	  	 	  
	      float chiratiocut=thr[ibin+6]; 
	      float gschicut=thr[ibin+7]; 
	      float gsptmin=thr[ibin+8];

	      GoodTkId=((dpt>gsptmin)&(chired<gschicut)&(chiRatio<chiratiocut));      
       
	    }
	  }
    
	  GoodPreId= GoodTkId | GoodMatching; 

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
	  iEvent.getByToken(tracksContainers_[0],tkRefCollection);
	  fillPreIdRefValueMap(tkRefCollection,preIdRefProd,mapFiller);
	  mapFiller.fill();
	  iEvent.put(preIdMap_p,preidname_);
	}
    }
  // clear temporary maps
  refMap_.clear();
}

// intialize the cross-thread cache to hold gbr trees and resolution maps
namespace goodseedhelpers {
  HeavyObjectCache::HeavyObjectCache( const edm::ParameterSet& conf) {    
    // mvas
    const bool useTmva = conf.getUntrackedParameter<bool>("UseTMVA",false);
    
    if( useTmva ) {
      const std::string method_ = conf.getParameter<string>("TMVAMethod");
      std::array<edm::FileInPath,kMaxWeights> weights = {{ edm::FileInPath(conf.getParameter<string>("Weights1")),
                                                           edm::FileInPath(conf.getParameter<string>("Weights2")),
                                                           edm::FileInPath(conf.getParameter<string>("Weights3")),
                                                           edm::FileInPath(conf.getParameter<string>("Weights4")),
                                                           edm::FileInPath(conf.getParameter<string>("Weights5")),
                                                           edm::FileInPath(conf.getParameter<string>("Weights6")),
                                                           edm::FileInPath(conf.getParameter<string>("Weights7")),
                                                           edm::FileInPath(conf.getParameter<string>("Weights8")),
                                                           edm::FileInPath(conf.getParameter<string>("Weights9")) }};
            
      for(UInt_t j = 0; j < gbr.size(); ++j){
        TMVA::Reader reader("!Color:Silent");
                
        reader.AddVariable("NHits", &nhit);
        reader.AddVariable("NormChi", &chikfred);
        reader.AddVariable("dPtGSF", &dpt);
        reader.AddVariable("EoP", &eP);
        reader.AddVariable("ChiRatio", &chiRatio);
        reader.AddVariable("RedChi", &chired);
        reader.AddVariable("EcalDEta", &trk_ecalDeta);
        reader.AddVariable("EcalDPhi", &trk_ecalDphi);
        reader.AddVariable("pt", &pt);
        reader.AddVariable("eta", &eta);
        
        std::unique_ptr<TMVA::IMethod> temp( reader.BookMVA(method_, weights[j].fullPath().c_str()) );
        
        gbr[j].reset( new GBRForest( dynamic_cast<TMVA::MethodBDT*>( reader.FindMVA(method_) ) ) );
      }    
    }
  }
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
  
  pfTransformer_.reset( new PFTrackTransformer(B_) );
  pfTransformer_->OnlyProp();
  
  //Resolution maps
    FileInPath ecalEtaMap(conf_.getParameter<string>("EtaMap"));
    FileInPath ecalPhiMap(conf_.getParameter<string>("PhiMap"));
    resMapEtaECAL_.reset( new PFResolutionMap("ECAL_eta",ecalEtaMap.fullPath().c_str()) );
    resMapPhiECAL_.reset( new PFResolutionMap("ECAL_phi",ecalPhiMap.fullPath().c_str()) );

  //read threshold
  FileInPath parFile(conf_.getParameter<string>("ThresholdFile"));
  ifstream ifs(parFile.fullPath().c_str());
  for (int iy=0;iy<81;++iy) ifs >> thr[iy];
}

int 
GoodSeedProducer::getBin(float eta, float pt){
  int ie=0;
  int ip=0;
  if (fabs(eta)<0.8) ie=0;
  else{ if (fabs(eta)<1.479) ie=1;
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

DEFINE_FWK_MODULE(GoodSeedProducer);
