#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFTracking/interface/GoodSeedProducer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoParticleFlow/PFAlgo/interface/PFGeometry.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <fstream>
#include <string>
using namespace edm;
using namespace std;
reco::PFResolutionMap* GoodSeedProducer::resMapEtaECAL_ = 0;
reco::PFResolutionMap* GoodSeedProducer::resMapPhiECAL_ = 0;
GoodSeedProducer::GoodSeedProducer(const ParameterSet& iConfig):conf_(iConfig),
							       trackAlgo_(iConfig)
{
  LogInfo("GoodSeedProducer")<<"Electron PreIdentification started  ";

  //now do what ever initialization is needed
  
  recTrackCandidateModuleLabel_
    = iConfig.getParameter<string>
    ("RecTrackCandidateModuleLabel");
  recTrackCollectionLabel_
    = iConfig.getParameter<string>
    ("RecTrackModuleLabel");
 
  pfCLusTagECLabel_=iConfig.getParameter<InputTag>
	("PFEcalClusterLabel");
  pfCLusTagPSLabel_=iConfig.getParameter<InputTag>
         ("PFPSClusterLabel");

  preidgsf_=iConfig.getParameter<string>
    ("PreGsfLabel");
  preidckf_=iConfig.getParameter<string>
    ("PreCkfLabel");

  propagatorName_ = iConfig.getParameter<string>("Propagator");
  builderName_ = iConfig.getParameter<string>("TTRHBuilder"); 
  fitterName_ = iConfig.getParameter<string>("Fitter");   
  

  produceCkfseed = iConfig.getUntrackedParameter<bool>("ProduceCkfSeed",false);
  produceCkfPFT = iConfig.getUntrackedParameter<bool>("ProduceCkfPFTracks",true);  
  LogDebug("GoodSeedProducer")<<"Seeds for GSF will be produced ";

  produces<TrajectorySeedCollection>(preidgsf_);
  if(produceCkfseed){
    LogDebug("GoodSeedProducer")<<"Seeds for CKF will be produced ";
   produces<TrajectorySeedCollection>(preidckf_);
  }
  if(produceCkfPFT){
    LogDebug("GoodSeedProducer")<<"PFTracks from CKF tracks will be produced ";
    produces<reco::PFRecTrackCollection>();
  }
}


GoodSeedProducer::~GoodSeedProducer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.) 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GoodSeedProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  pftracks.clear();
  LogDebug("GoodSeedProducer")<<"START event: "<<iEvent.id().event()
			      <<" in run "<<iEvent.id().run();


  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  string builderName = conf_.getParameter<string>("TTRHBuilder");   
  iSetup.get<TransientRecHitRecord>().get(builderName,theBuilder);  
  RHBuilder=   theBuilder.product();



  auto_ptr<TrajectorySeedCollection> output_preid(new TrajectorySeedCollection);
  auto_ptr<TrajectorySeedCollection> output_nopre(new TrajectorySeedCollection);
  auto_ptr< reco::PFRecTrackCollection > 
    pOutputPFRecTrackCollection(new reco::PFRecTrackCollection);


  float etarec=0;
  float phirec=0;
  float PTOB=0.;
  float nhitpi=11110.;
  float chired=0.;
  float PUB=0;
  float poutin=0;


  Handle<TrackCandidateCollection> theTCCollection;
  iEvent.getByLabel(recTrackCandidateModuleLabel_, theTCCollection);

  Handle<reco::TrackCollection> tkCollection;
  iEvent.getByLabel(recTrackCollectionLabel_, tkCollection);
  reco::TrackCollection  hj=*(tkCollection.product());


  Handle<reco::PFClusterCollection> theECPfClustCollection;
  iEvent.getByLabel(pfCLusTagECLabel_,theECPfClustCollection);
  Handle<reco::PFClusterCollection> thePSPfClustCollection;
  iEvent.getByLabel(pfCLusTagPSLabel_,thePSPfClustCollection);
  vector<reco::PFCluster> basClus;
  basClus.insert(basClus.end(),theECPfClustCollection.product()->begin(),theECPfClustCollection.product()->end());
  basClus.insert(basClus.end(),thePSPfClustCollection.product()->begin(),thePSPfClustCollection.product()->end());


  AlgoProductCollection algoResults;
  float toteta=1000;
  float totphi=1000;
  float dr=1000;
  float EP=900;
  float PTFIN=0;
  float PTIN=0;
  float pttin=0;
  float EE=0;
  float feta=0;
  trackAlgo_.runWithCandidate(theG.product(), theMF.product(), 
			      *theTCCollection,
			      theFitter.product(), thePropagator.product(),
			      theBuilder.product(), algoResults);

  LogDebug("GoodSeedProducer")<<"Number of tracks to be analyzed "<<algoResults.size();

  for(AlgoProductCollection::iterator itTrack = algoResults.begin();
      itTrack != algoResults.end(); itTrack++) {
    
    
    PTOB=(*itTrack).first->lastMeasurement().updatedState().globalMomentum().mag();
    PUB=(*itTrack).first->firstMeasurement().updatedState().globalMomentum().mag();
    PTIN=(*itTrack).first->firstMeasurement().updatedState().globalMomentum().perp();
    PTFIN=(*itTrack).first->lastMeasurement().updatedState().globalMomentum().perp();
     
    chired=(*itTrack).second->normalizedChi2();
    nhitpi=(*itTrack).first->foundHits();
    Seed=(*itTrack).first->seed();
 
    if(PTOB!=0) poutin=abs(PTOB-PUB)/PUB;
    if(PTIN!=0) pttin=abs(PTFIN-PTIN)/PTIN;

    TSOS ecalTsos=
      PFTransformer->getStateOnSurface(PFGeometry::ECALInnerWall,
				       (*itTrack).first->firstMeasurement().updatedState(),
				       thePropagator.product(),side);  
 

    if(ecalTsos.isValid()){
      etarec=ecalTsos.globalPosition().eta();
      phirec=ecalTsos.globalPosition().phi();
      math::XYZPoint showerDirection=math::XYZPoint(ecalTsos.globalMomentum().x(),
						    ecalTsos.globalMomentum().y(),
						    ecalTsos.globalMomentum().z());
      
    
      for(vector<reco::PFCluster>::const_iterator aClus = basClus.begin();
	  aClus != basClus.end(); aClus++) {
	if (sqrt(pow((aClus->positionXYZ().phi()-phirec),2)+
		 pow((aClus->positionXYZ().eta()-etarec),2))<dr) {
	  toteta=aClus->positionXYZ().eta()-etarec;
	  totphi=aClus->positionXYZ().phi()-phirec;
	  EP=aClus->energy()/PTOB;
	  EE=aClus->energy();
	  feta= aClus->positionXYZ().eta();
	}
      }
      
      
      double ecalShowerDepth 
	= reco::PFCluster::getDepthCorrection(EE, 
					      true, 
					      false);

      showerDirection *= ecalShowerDepth/showerDirection.R();

      
      double rCyl = PFGeometry::innerRadius(PFGeometry::ECALBarrel) + 
	showerDirection.Rho();
      double zCyl = PFGeometry::innerZ(PFGeometry::ECALEndcap) + 
	fabs(showerDirection.Z());
      ReferenceCountingPointer<Surface> showerMaxWall;
      const float epsilon = 0.001; // should not matter at all
      switch (side) {
      case 0: 
	showerMaxWall 
	  = ReferenceCountingPointer<Surface>( new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(rCyl, rCyl, -zCyl, zCyl))); 
	break;
      case +1: 
	showerMaxWall 
	  = ReferenceCountingPointer<Surface>( new BoundPlane(Surface::PositionType(0,0,zCyl), TkRotation<float>(), SimpleDiskBounds(0., rCyl, -epsilon, epsilon))); 
	break;
      case -1: 
	showerMaxWall 
	  = ReferenceCountingPointer<Surface>(new BoundPlane(Surface::PositionType(0,0,-zCyl), TkRotation<float>(), SimpleDiskBounds(0., rCyl, -epsilon, epsilon))); 
	break;
      }
      if (&(*showerMaxWall)!=0){
	TSOS maxShTsos=thePropagator.product()->propagate(ecalTsos, *showerMaxWall);
	if (maxShTsos.isValid()){
	  toteta+=(etarec-maxShTsos.globalPosition().eta());
	  totphi+=(phirec-maxShTsos.globalPosition().phi());
	}
      }
    }

    float etatt=fabs((*itTrack).second->eta());
    float pttt=(*itTrack).second->pt();
    int ibin=getBin(etatt,pttt)*7;

    //thresholds
    float qphi=thr[ibin+0];
    float chi2cut=thr[ibin+1];
    float ep_cutmin=thr[ibin+2];
    float chiredmin=thr[ibin+3];
    float pttmin=thr[ibin+4];
    int hit1max=int(thr[ibin+5]);
    int hit2max=int(thr[ibin+6]);
    //

    double ecaletares 
      = resMapEtaECAL_->GetBinContent(resMapEtaECAL_->FindBin(feta,EE));
    double ecalphires 
      = resMapPhiECAL_->GetBinContent(resMapPhiECAL_->FindBin(feta,EE)); 


    float chieta= toteta/ecaletares;
    float chiphi= totphi/(ecalphires+qphi);
    float chichi= chieta*chieta + chiphi*chiphi;

    bool aa1=(chichi<chi2cut)  ? true : false;
    bool aa2= ((EP>ep_cutmin)&&(EP<1.2)) ? true : false;

    bool bb1= (aa1 && aa2);
    bool bb2=((nhitpi<hit1max) && (chired>chiredmin))  ? true : false;
    bool bb3=((nhitpi<hit2max) && (pttin>pttmin))  ? true : false;


    bool cc1= (bb1 || bb2 || bb3); 
    
    if(bb1)
      LogDebug("GoodSeedProducer")<<"Track (pt= "<<(*itTrack).second->pt()<<
	"GeV/c, eta= "<<(*itTrack).second->eta() <<
	") preidentified for agreement between  track and ECAL cluster";
    if(cc1 &&(!bb1))
      LogDebug("GoodSeedProducer")<<"Track (pt= "<<(*itTrack).second->pt()<<
	"GeV/c, eta= "<<(*itTrack).second->eta() <<
	") preidentified only for track properties";

    index=-1;
    if (produceCkfPFT) index=findIndex(hj,*itTrack);
    
    AlgoProduct ap=*itTrack;
      
    //QUESTI PIONI CHE FINE FANNO
    if (cc1){
  
      output_preid->push_back(Seed);
      if(produceCkfPFT){
	reco::PFRecTrack pft=PFTransformer->
	  producePFtrackKf(ap,reco::PFRecTrack::KF_ELCAND,index);
	  pftracks.push_back(pft);
      }   
    }  else{
      if (produceCkfseed){
	output_nopre->push_back(Seed);
      }
      if(produceCkfPFT){
	reco::PFRecTrack pft=PFTransformer->
	  producePFtrackKf(ap,reco::PFRecTrack::KF,index);
	pftracks.push_back(pft); 
      }
    }
  }

  if(produceCkfPFT){
    for(uint ipf=0; ipf<pftracks.size();ipf++)
      pOutputPFRecTrackCollection->push_back(pftracks[ipf]);
    iEvent.put(pOutputPFRecTrackCollection);
  }
  
  iEvent.put(output_preid,preidgsf_);
  if (produceCkfseed)
    iEvent.put(output_nopre,preidckf_);

}
// ------------ method called once each job just before starting event loop  ------------
void 
GoodSeedProducer::beginJob(const EventSetup& es)
{

  //Get Magnetic Field
  es.get<IdealMagneticFieldRecord>().get(theMF);   
  magField = theMF.product();
  es.get<TrackerDigiGeometryRecord>().get(theG);
  es.get<TrackingComponentsRecord>().get(propagatorName_, thePropagator);
  es.get<TrackingComponentsRecord>().get(fitterName_, theFitter);
  es.get<TransientRecHitRecord>().get(builderName_, theBuilder);
  PFTransformer= new PFTrackTransformer(magField);

  resMapEtaECAL_ = new reco::PFResolutionMap("ECAL_eta", 
					     "RecoParticleFlow/PFProducer/data/resmap_ECAL_eta.dat");
  resMapPhiECAL_ = new reco::PFResolutionMap("ECAL_phi", 
					     "RecoParticleFlow/PFProducer/data/resmap_ECAL_phi.dat");
  //read threshold
  std::string parfile = conf_.getParameter<string>
    ("ThresholdFile");
  std::string name = "RecoParticleFlow/PFTracking/data/";
  name+=parfile;
  edm::FileInPath parFile(name);
  std::ifstream ifs(parFile.fullPath().c_str());
  for (int iy=0;iy<105;iy++) ifs >> thr[iy];

}
int GoodSeedProducer::findIndex(reco::TrackCollection  hj, 
			     AlgoProduct ap){
  int itrackSel=-1;
  for(uint ttt=0;ttt<hj.size();ttt++){
    if ((ap.second->phi()==hj[ttt].phi()) &&
       	(ap.second->pt()==hj[ttt].pt()) &&
       	(ap.second->eta()==hj[ttt].eta())) itrackSel=ttt;
  }
  return itrackSel;
}
int GoodSeedProducer::getBin(float eta, float pt){
  int ie=0;
  int ip=0;
  if (eta<0.8) ie=0;
  else{ if (eta<1.6) ie=1;
    else ie=2;
  }
  if (pt<2) ip=0;
  else {  if (pt<5) ip=1;
    else {  if (pt<9) ip=2;
      else {  if (pt<20) ip=4;
	else ip=3;
      }
    }
  }
  int iep= ie*5+ip;
  return iep;
}
