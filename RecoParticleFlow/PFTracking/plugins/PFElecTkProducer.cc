// -*- C++ -*-
//
// Package:    PFTracking
// Class:      PFElecTkProducer
// 
// Original Author:  Michele Pioppi
//         Created:  Tue Jan 23 15:26:39 CET 2007



// system include files
#include <memory>

// user include files
#include "RecoParticleFlow/PFTracking/interface/PFElecTkProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "RecoParticleFlow/PFTracking/interface/ConvBremPFTrackFinder.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"

#include "TMath.h"
using namespace std;
using namespace edm;
using namespace reco;
PFElecTkProducer::PFElecTkProducer(const ParameterSet& iConfig):
  conf_(iConfig),
  pfTransformer_(0),
  convBremFinder_(0)
{
  LogInfo("PFElecTkProducer")<<"PFElecTkProducer started";

  gsfTrackLabel_ = iConfig.getParameter<InputTag>
    ("GsfTrackModuleLabel");

  pfTrackLabel_ = iConfig.getParameter<InputTag>
    ("PFRecTrackLabel");

  primVtxLabel_ = iConfig.getParameter<InputTag>
    ("PrimaryVertexLabel");

  pfEcalClusters_ = iConfig.getParameter<InputTag>
    ("PFEcalClusters");

  pfNuclear_ = iConfig.getParameter<InputTag>
    ("PFNuclear");
  
  pfConv_ = iConfig.getParameter<InputTag>
    ("PFConversions");
  
  pfV0_ = iConfig.getParameter<InputTag>
    ("PFV0");

  useNuclear_ = iConfig.getParameter<bool>("useNuclear");
  useConversions_ = iConfig.getParameter<bool>("useConversions");
  useV0_ = iConfig.getParameter<bool>("useV0");


  produces<GsfPFRecTrackCollection>();
  produces<GsfPFRecTrackCollection>( "Secondary" ).setBranchAlias( "secondary" );


  trajinev_ = iConfig.getParameter<bool>("TrajInEvents");
  modemomentum_ = iConfig.getParameter<bool>("ModeMomentum");
  applySel_ = iConfig.getParameter<bool>("applyEGSelection");
  applyGsfClean_ = iConfig.getParameter<bool>("applyGsfTrackCleaning");
  useFifthStep_ = iConfig.getParameter<bool>("useFifthTrackingStep");
  useFifthStepSec_ = iConfig.getParameter<bool>("useFifthTrackingStepForSecondaries");
  maxPtConvReco_ = iConfig.getParameter<double>("MaxConvBremRecoPT");
  detaGsfSC_ = iConfig.getParameter<double>("MinDEtaGsfSC");
  dphiGsfSC_ = iConfig.getParameter<double>("MinDPhiGsfSC");
  SCEne_ = iConfig.getParameter<double>("MinSCEnergy");
  
  // set parameter for convBremFinder
  useConvBremFinder_ =     iConfig.getParameter<bool>("useConvBremFinder");
  mvaConvBremFinderID_
    = iConfig.getParameter<double>("pf_convBremFinderID_mvaCut");
  
  string mvaWeightFileConvBrem
    = iConfig.getParameter<string>("pf_convBremFinderID_mvaWeightFile");
  
  
  if(useConvBremFinder_) 
    path_mvaWeightFileConvBrem_ = edm::FileInPath ( mvaWeightFileConvBrem.c_str() ).fullPath();

}


PFElecTkProducer::~PFElecTkProducer()
{
 
  delete pfTransformer_;
  delete convBremFinder_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PFElecTkProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  LogDebug("PFElecTkProducer")<<"START event: "<<iEvent.id().event()
			      <<" in run "<<iEvent.id().run();

  //create the empty collections 
  auto_ptr< GsfPFRecTrackCollection > 
    gsfPFRecTrackCollection(new GsfPFRecTrackCollection);

  
  auto_ptr< GsfPFRecTrackCollection > 
    gsfPFRecTrackCollectionSecondary(new GsfPFRecTrackCollection);

  //read collections of tracks
  Handle<GsfTrackCollection> gsftrackscoll;
  iEvent.getByLabel(gsfTrackLabel_,gsftrackscoll);

  //read collections of trajectories
  Handle<vector<Trajectory> > TrajectoryCollection;
 
  //read pfrectrack collection
  Handle<PFRecTrackCollection> thePfRecTrackCollection;
  iEvent.getByLabel(pfTrackLabel_,thePfRecTrackCollection);
  const PFRecTrackCollection& PfRTkColl = *(thePfRecTrackCollection.product());

  // PFClusters
  Handle<PFClusterCollection> theECPfClustCollection;
  iEvent.getByLabel(pfEcalClusters_,theECPfClustCollection);
  const PFClusterCollection& theEcalClusters = *(theECPfClustCollection.product());

  //Primary Vertexes
  Handle<reco::VertexCollection> thePrimaryVertexColl;
  iEvent.getByLabel(primVtxLabel_,thePrimaryVertexColl);



  // Displaced Vertex
  Handle< reco::PFDisplacedTrackerVertexCollection > pfNuclears; 
  if( useNuclear_ ) {
    bool found = iEvent.getByLabel(pfNuclear_, pfNuclears);
    
    
    if(!found )
      LogError("PFElecTkProducer")<<" cannot get PFNuclear : "
				  <<  pfNuclear_
				  << " please set useNuclear=False in RecoParticleFlow/PFTracking/python/pfTrackElec_cfi.py" << endl;
  }

  // Conversions 
  Handle< reco::PFConversionCollection > pfConversions;
  if( useConversions_ ) {
    bool found = iEvent.getByLabel(pfConv_,pfConversions);
    if(!found )
      LogError("PFElecTkProducer")<<" cannot get PFConversions : "
				  << pfConv_ 
				  << " please set useConversions=False in RecoParticleFlow/PFTracking/python/pfTrackElec_cfi.py" << endl;
  }

  // V0
  Handle< reco::PFV0Collection > pfV0;
  if( useV0_ ) {
    bool found = iEvent.getByLabel(pfV0_, pfV0);
    
    if(!found )
      LogError("PFElecTkProducer")<<" cannot get PFV0 : "
				  << pfV0_
				  << " please set useV0=False  RecoParticleFlow/PFTracking/python/pfTrackElec_cfi.py" << endl;
  }
  
  

  GsfTrackCollection gsftracks = *(gsftrackscoll.product());	
  vector<Trajectory> tjvec(0);
  if (trajinev_){
    bool foundTraj = iEvent.getByLabel(gsfTrackLabel_,TrajectoryCollection); 
    if(!foundTraj) 
      LogError("PFElecTkProducer")
	<<" cannot get Trajectories of : "
	<<  gsfTrackLabel_
	<< " please set TrajInEvents = False in RecoParticleFlow/PFTracking/python/pfTrackElec_cfi.py" << endl;
    
    tjvec= *(TrajectoryCollection.product());
  }
  

  vector<reco::GsfPFRecTrack> selGsfPFRecTracks(0);
  selGsfPFRecTracks.clear();
  
  
  for (unsigned int igsf=0; igsf<gsftracks.size();igsf++) {
    
    GsfTrackRef trackRef(gsftrackscoll, igsf);
    
    int kf_ind=FindPfRef(PfRTkColl,gsftracks[igsf],false);
    
    if (kf_ind>=0) {
      
      PFRecTrackRef kf_ref(thePfRecTrackCollection,
			   kf_ind);
      pftrack_=GsfPFRecTrack( gsftracks[igsf].charge(), 
			      reco::PFRecTrack::GSF, 
			      igsf, trackRef,
			      kf_ref);
    } else  {
      PFRecTrackRef dummyRef;
      pftrack_=GsfPFRecTrack( gsftracks[igsf].charge(), 
			      reco::PFRecTrack::GSF, 
			      igsf, trackRef,
			      dummyRef);
    }
    
    
    bool validgsfbrem = false;
    if(trajinev_) {
      validgsfbrem = pfTransformer_->addPointsAndBrems(pftrack_, 
						       gsftracks[igsf], 
						       tjvec[igsf],
						       modemomentum_);
    } else {
      validgsfbrem = pfTransformer_->addPointsAndBrems(pftrack_, 
						       gsftracks[igsf], 
						       mtsTransform_);
    }
    
    bool passSel = true;
    if(applySel_) 
      passSel = applySelection(gsftracks[igsf]);      
    
    
    if(validgsfbrem && passSel) 
      selGsfPFRecTracks.push_back(pftrack_);
  }
  if(selGsfPFRecTracks.size() > 0) {
    for(unsigned int ipfgsf=0; ipfgsf<selGsfPFRecTracks.size();ipfgsf++) {
      
      vector<unsigned int> secondaries(0);
      secondaries.clear();
      bool keepGsf = true;
      
      if(applyGsfClean_) {
	keepGsf = resolveGsfTracks(selGsfPFRecTracks,ipfgsf,secondaries);
      }
      
      //is primary? 
      if(keepGsf == true) {
	PFRecTrackRef refprimKF =  selGsfPFRecTracks[ipfgsf].kfPFRecTrackRef();
	
	// SKIP 5TH STEP IF REQUESTED
	if(refprimKF.isNonnull()) {
	  if(useFifthStep_ == false) {
	    bool isFifthStepTrack = isFifthStep(refprimKF);
	    if(isFifthStepTrack)
	      continue;
	  }
	}
	

	// Find kf tracks from converted brem photons
	if(convBremFinder_->foundConvBremPFRecTrack(thePfRecTrackCollection,thePrimaryVertexColl,
						    pfNuclears,pfConversions,pfV0,
						    useNuclear_,useConversions_,useV0_,
						    theEcalClusters,selGsfPFRecTracks[ipfgsf])) {
	  const vector<PFRecTrackRef>& convBremPFRecTracks (convBremFinder_->getConvBremPFRecTracks());
	  for(unsigned int ii = 0; ii<convBremPFRecTracks.size(); ii++) {
	    selGsfPFRecTracks[ipfgsf].addConvBremPFRecTrackRef(convBremPFRecTracks[ii]);
	  }
	}
	
	// save primaries gsf tracks
	gsfPFRecTrackCollection->push_back(selGsfPFRecTracks[ipfgsf]);
	
	
	
	// NOTE:: THE TRACKID IS USED TO LINK THE PRIMARY GSF TRACK. THIS NEEDS 
	// TO BE CHANGED AS SOON AS IT IS POSSIBLE TO CHANGE DATAFORMATS
	// A MODIFICATION HERE IMPLIES A MODIFICATION IN PFBLOCKALGO.CC/H
	unsigned int primGsfIndex = selGsfPFRecTracks[ipfgsf].trackId();
	if(secondaries.size() > 0) {
	  // loop on secondaries gsf tracks (from converted brems)
	  for(unsigned int isecpfgsf=0; isecpfgsf<secondaries.size();isecpfgsf++) {
	    
	    PFRecTrackRef refsecKF =  selGsfPFRecTracks[(secondaries[isecpfgsf])].kfPFRecTrackRef();
	    
	    // SKIP 5TH STEP IF REQUESTED
	    if(refsecKF.isNonnull()) {
	      if(useFifthStepSec_ == false) {
		bool isFifthStepTrack = isFifthStep(refsecKF);
		if(isFifthStepTrack)
		  continue;
	      }
	    }
	    
	    unsigned int secGsfIndex = selGsfPFRecTracks[(secondaries[isecpfgsf])].trackId();
	    GsfTrackRef secGsfRef = selGsfPFRecTracks[(secondaries[isecpfgsf])].gsfTrackRef();
	    TrajectoryStateOnSurface outTSOS = mtsTransform_.outerStateOnSurface((*secGsfRef));
	    GlobalVector outMomCart;   
	    if(outTSOS.isValid()){
	      mtsMode_->momentumFromModeCartesian(outTSOS,outMomCart);
	    }
	    
	    if(refsecKF.isNonnull()) {
	      // NOTE::IT SAVED THE TRACKID OF THE PRIMARY!!! THIS IS USED IN PFBLOCKALGO.CC/H
	      secpftrack_= GsfPFRecTrack( gsftracks[secGsfIndex].charge(), 
					  reco::PFRecTrack::GSF, 
					  primGsfIndex, secGsfRef,
					  refsecKF);
	    }
	    else{
	      PFRecTrackRef dummyRef;
	      // NOTE::IT SAVED THE TRACKID OF THE PRIMARY!!! THIS IS USED IN PFBLOCKALGO.CC/H
	      secpftrack_= GsfPFRecTrack( gsftracks[secGsfIndex].charge(), 
					  reco::PFRecTrack::GSF, 
					  primGsfIndex, secGsfRef,
					  dummyRef);
	    }

	    bool validgsfbrem = false;
	    if(trajinev_) {
	      validgsfbrem = pfTransformer_->addPointsAndBrems(secpftrack_,
							       gsftracks[secGsfIndex], 
							       tjvec[secGsfIndex],
							       modemomentum_);
	    } else {
	      validgsfbrem = pfTransformer_->addPointsAndBrems(secpftrack_,
							       gsftracks[secGsfIndex], 
							       mtsTransform_);
	    }	      

	    if(validgsfbrem) 
	      gsfPFRecTrackCollectionSecondary->push_back(secpftrack_);
	  }
	}
      }
    }
  }
  
  iEvent.put(gsfPFRecTrackCollection);
  iEvent.put(gsfPFRecTrackCollectionSecondary,"Secondary");
  
  
}
// ------------- method for find the corresponding kf pfrectrack ---------------------
int
PFElecTkProducer::FindPfRef(const reco::PFRecTrackCollection  & PfRTkColl, 
			    reco::GsfTrack gsftk,
			    bool otherColl){


  if (&(*gsftk.seedRef())==0) return -1;
  ElectronSeedRef ElSeedRef=gsftk.extra()->seedRef().castTo<ElectronSeedRef>();
  //CASE 1 ELECTRONSEED DOES NOT HAVE A REF TO THE CKFTRACK
  if (ElSeedRef->ctfTrack().isNull()){
    reco::PFRecTrackCollection::const_iterator pft=PfRTkColl.begin();
    reco::PFRecTrackCollection::const_iterator pftend=PfRTkColl.end();
    unsigned int i_pf=0;
    int ibest=-1;
    unsigned int ish_max=0;
    float dr_min=1000;
    //SEARCH THE PFRECTRACK THAT SHARES HITS WITH THE ELECTRON SEED
    for(;pft!=pftend;++pft){
      unsigned int ish=0;
      
      float dph= fabs(pft->trackRef()->phi()-gsftk.phi()); 
      if (dph>TMath::Pi()) dph-= TMath::TwoPi();
      float det=fabs(pft->trackRef()->eta()-gsftk.eta());
      float dr =sqrt(dph*dph+det*det);  
      
      trackingRecHit_iterator  hhit=
	pft->trackRef()->recHitsBegin();
      trackingRecHit_iterator  hhit_end=
	pft->trackRef()->recHitsEnd();
      
    
      
      for(;hhit!=hhit_end;++hhit){
	if (!(*hhit)->isValid()) continue;
	TrajectorySeed::const_iterator hit=
	  gsftk.seedRef()->recHits().first;
	TrajectorySeed::const_iterator hit_end=
	  gsftk.seedRef()->recHits().second;
 	for(;hit!=hit_end;++hit){
	  if (!(hit->isValid())) continue;
	  if((*hhit)->sharesInput(&*(hit),TrackingRecHit::all))  ish++; 
	//   if((hit->geographicalId()==(*hhit)->geographicalId())&&
        //     (((*hhit)->localPosition()-hit->localPosition()).mag()<0.01)) ish++;
 	}	
	
      }
      

      if ((ish>ish_max)||
	  ((ish==ish_max)&&(dr<dr_min))){
	ish_max=ish;
	dr_min=dr;
	ibest=i_pf;
      }
      
   
    
      i_pf++;
    }
    if (ibest<0) return -1;
    
    if((ish_max==0) || (dr_min>0.05))return -1;
    if(otherColl && (ish_max==0)) return -1;
    return ibest;
  }
  else{
    //ELECTRON SEED HAS A REFERENCE
   
    reco::PFRecTrackCollection::const_iterator pft=PfRTkColl.begin();
    reco::PFRecTrackCollection::const_iterator pftend=PfRTkColl.end();
    unsigned int i_pf=0;
    
    for(;pft!=pftend;++pft){
      //REF COMPARISON
      if (pft->trackRef()==ElSeedRef->ctfTrack()){
	return i_pf;
      }
      i_pf++;
    }
  }
  return -1;
}
bool PFElecTkProducer::isFifthStep(reco::PFRecTrackRef pfKfTrack) {

  bool isFithStep = false;
  

  TrackRef kfref = pfKfTrack->trackRef();
  unsigned int Algo = 0; 
  switch (kfref->algo()) {
  case TrackBase::ctf:
  case TrackBase::iter0:
  case TrackBase::iter1:
    Algo = 0;
    break;
  case TrackBase::iter2:
    Algo = 1;
    break;
  case TrackBase::iter3:
    Algo = 2;
    break;
  case TrackBase::iter4:
    Algo = 3;
    break;
  case TrackBase::iter5:
    Algo = 4;
    break;
  default:
    Algo = 5;
    break;
  }
  if ( Algo >= 4 ) {
    isFithStep = true;
  }

  return isFithStep;
}
// -- method to apply gsf electron selection to EcalDriven seeds
bool 
PFElecTkProducer::applySelection(reco::GsfTrack gsftk) {
  if (&(*gsftk.seedRef())==0) return false;
  ElectronSeedRef ElSeedRef=gsftk.extra()->seedRef().castTo<ElectronSeedRef>();

  bool passCut = false;
  if (ElSeedRef->ctfTrack().isNull()){
    if(ElSeedRef->caloCluster().isNull()) return passCut;
    SuperClusterRef scRef = ElSeedRef->caloCluster().castTo<SuperClusterRef>();
    //do this just to know if exist a SC? 
    if(scRef.isNonnull()) {
      float caloEne = scRef->energy();
      float feta = fabs(scRef->eta()-gsftk.etaMode());
      float fphi = fabs(scRef->phi()-gsftk.phiMode());
      if (fphi>TMath::Pi()) fphi-= TMath::TwoPi();
      if(caloEne > SCEne_ && feta < detaGsfSC_ && fabs(fphi) < dphiGsfSC_)
	passCut = true;
    }
  }
  else {
    // get all the gsf found by tracker driven
    passCut = true;
  }
  return passCut;
}
bool 
PFElecTkProducer::resolveGsfTracks(const vector<reco::GsfPFRecTrack>  & GsfPFVec, 
				   unsigned int ngsf, 
				   vector<unsigned int> &secondaries) {


  reco::GsfTrackRef nGsfTrack = GsfPFVec[ngsf].gsfTrackRef();

  if (&(*nGsfTrack->seedRef())==0) return false;    
  ElectronSeedRef nElSeedRef=nGsfTrack->extra()->seedRef().castTo<ElectronSeedRef>();
  

  bool n_keepGsf = true;
  const math::XYZPoint nxyz = nGsfTrack->innerPosition();
  int nhits=nGsfTrack->numberOfValidHits();
  int ncharge = nGsfTrack->chargeMode();
  TrajectoryStateOnSurface outTSOS = mtsTransform_.outerStateOnSurface((*nGsfTrack));
  TrajectoryStateOnSurface inTSOS = mtsTransform_.innerStateOnSurface((*nGsfTrack));
  GlobalVector ninnMom;

  int outCharge = -2;
  int inCharge = -2;
  float nPin =  nGsfTrack->pMode();
  if(outTSOS.isValid())
    outCharge = mtsMode_->chargeFromMode(outTSOS);	  
  if(inTSOS.isValid()){
    inCharge = mtsMode_->chargeFromMode(inTSOS);
    mtsMode_->momentumFromModeCartesian(inTSOS,ninnMom);
    nPin = ninnMom.mag();
  }

  float nchi2 = nGsfTrack->chi2();
  float neta = nGsfTrack->innerMomentum().eta();
  float nphi = nGsfTrack->innerMomentum().phi();
  float ndist = sqrt(nxyz.x()*nxyz.x()+
		     nxyz.y()*nxyz.y()+
		     nxyz.z()*nxyz.z());
  

  
  for (unsigned int igsf=0; igsf< GsfPFVec.size();igsf++) {
    if(igsf != ngsf ) {

      reco::GsfTrackRef iGsfTrack = GsfPFVec[igsf].gsfTrackRef();

      float ieta = iGsfTrack->innerMomentum().eta();
      float iphi = iGsfTrack->innerMomentum().phi();
      float feta = fabs(neta - ieta);
      float fphi = fabs(nphi - iphi);
      if (fphi>TMath::Pi()) fphi-= TMath::TwoPi();     
      const math::XYZPoint ixyz = iGsfTrack->innerPosition();
      float idist = sqrt(ixyz.x()*ixyz.x()+
			 ixyz.y()*ixyz.y()+
			 ixyz.z()*ixyz.z());
      
      float minBremDphi =  selectSecondaries(GsfPFVec[ngsf],GsfPFVec[igsf]);
  
      if(feta < 0.05 && (fabs(fphi) < 0.3 ||  minBremDphi < 0.05)) {

	TrajectoryStateOnSurface i_outTSOS = mtsTransform_.outerStateOnSurface((*iGsfTrack));
	TrajectoryStateOnSurface i_inTSOS = mtsTransform_.innerStateOnSurface((*iGsfTrack));
	GlobalVector i_innMom;
	int i_outCharge = -2;
	int i_inCharge = -2;
	float iPin = iGsfTrack->pMode();
	if(i_outTSOS.isValid())
	  i_outCharge = mtsMode_->chargeFromMode(i_outTSOS);	  
	if(i_inTSOS.isValid()){
	  i_inCharge = mtsMode_->chargeFromMode(i_inTSOS);
	  mtsMode_->momentumFromModeCartesian(i_inTSOS,i_innMom);  
	  iPin = i_innMom.mag();
	}

	if (&(*iGsfTrack->seedRef())==0) continue;    
	ElectronSeedRef iElSeedRef=iGsfTrack->extra()->seedRef().castTo<ElectronSeedRef>();
	
	// First Case: both gsf track have a reference to a SC: cleaning using SC 
	if(nElSeedRef->caloCluster().isNonnull() && iElSeedRef->caloCluster().isNonnull()) {

	  SuperClusterRef nscRef = nElSeedRef->caloCluster().castTo<SuperClusterRef>();
	  if(nscRef.isNull()) {
	    n_keepGsf = false;
	    return n_keepGsf;
	  }    
	  float nEP = nscRef->energy()/nPin;
	  SuperClusterRef iscRef = iElSeedRef->caloCluster().castTo<SuperClusterRef>();
	  if(iscRef.isNonnull()) {
	    if(nscRef == iscRef) {
	      float iEP =  iscRef->energy()/iPin;
	     
	      
	      trackingRecHit_iterator  nhit=nGsfTrack->recHitsBegin();
	      trackingRecHit_iterator  nhit_end=nGsfTrack->recHitsEnd();
	      unsigned int tmp_sh = 0;
	      for (;nhit!=nhit_end;++nhit){
		if ((*nhit)->isValid()){
		  trackingRecHit_iterator  ihit=iGsfTrack->recHitsBegin();
		  trackingRecHit_iterator  ihit_end=iGsfTrack->recHitsEnd();
		  for (;ihit!=ihit_end;++ihit){
		    if ((*ihit)->isValid()) {
		      if((*nhit)->sharesInput(&*(*ihit),TrackingRecHit::all))  tmp_sh++; 
		    }
		  }
		}
	      }
	      if (tmp_sh>0) {
		// if same SC take the closest or if same
		// distance the best E/p
		if (idist < (ndist-5)) {
		  n_keepGsf = false;
		  return n_keepGsf;
		}
		else if(ndist > (idist-5)){
		  if(fabs(iEP-1) < fabs(nEP-1)) {
		    n_keepGsf = false;
		    return n_keepGsf;
		  }
		  else{
		    if(minBremDphi < 0.05) 
		      secondaries.push_back(igsf);
		  }
		}
		else {
		  // save secondaries gsf track (put selection)
		  if(minBremDphi < 0.05) 
		    secondaries.push_back(igsf);
		}
	      }			      
	    }
	  }
	}
	else {
	  // Second Case: One Gsf has reference to a SC and the other one not or both not
	  // Cleaning using: radious first hit
	 
	  int ihits=iGsfTrack->numberOfValidHits();
	  float ichi2 = iGsfTrack->chi2();
	  int icharge = iGsfTrack->chargeMode();
	  
	  if (idist < (ndist-5)) {
	    n_keepGsf = false;
	    return n_keepGsf;
	  }
	  else if(ndist > (idist-5)){
	    // Thirt Case:  One Gsf has reference to a SC and the other one not or both not
	    // gsf tracks starts from the same layer
	    // check number of sharing modules (at least 50%)
	    // check number of sharing hits (at least 2)
	    // check charge flip inner/outer
	    
	    unsigned int sharedMod = 0;
	    unsigned int sharedHits = 0;
	    
	    trackingRecHit_iterator  nhit=nGsfTrack->recHitsBegin();
	    trackingRecHit_iterator  nhit_end=nGsfTrack->recHitsEnd();
	    for (;nhit!=nhit_end;++nhit){
	      if ((*nhit)->isValid()){
		trackingRecHit_iterator  ihit=iGsfTrack->recHitsBegin();
		trackingRecHit_iterator  ihit_end=iGsfTrack->recHitsEnd();
		for (;ihit!=ihit_end;++ihit){
		  if ((*ihit)->isValid()) {
		    if((*ihit)->geographicalId()==(*nhit)->geographicalId()) sharedMod++;
		    if((*nhit)->sharesInput(&*(*ihit),TrackingRecHit::all))  sharedHits++; 
		  }
		}
	      }
	    }
	    unsigned int den = ihits;
	    if(nhits < ihits)
	      den = nhits;
	    if (den == 0) den = 1;	    
	    float fracMod = sharedMod*1./den*1.;
	    
	    TrajectoryStateOnSurface i_outTSOS = mtsTransform_.outerStateOnSurface((*iGsfTrack));
	    TrajectoryStateOnSurface i_inTSOS = mtsTransform_.innerStateOnSurface((*iGsfTrack));
	    int i_outCharge = -2;
	    int i_inCharge = -2;
	    if(i_outTSOS.isValid())
	      i_outCharge = mtsMode_->chargeFromMode(i_outTSOS);	  
	    if(i_inTSOS.isValid())
	    i_inCharge = mtsMode_->chargeFromMode(i_inTSOS);
	    
	    
	    if(fracMod > 0.5 && sharedHits > 1 && icharge == ncharge && i_outCharge == i_inCharge) {
	      if(ihits > nhits) {
		n_keepGsf = false;
		return n_keepGsf;
	      }
	      else if(ihits == nhits  && ichi2 < nchi2) {
		n_keepGsf = false;
		return n_keepGsf;
	      }
	      else {
		if(minBremDphi < 0.05) 
		  secondaries.push_back(igsf);
	      }
	    }
	    if(fracMod > 0.3 && sharedHits > 1 && outCharge != -2 && inCharge != outCharge) {
	      n_keepGsf = false;
	      return n_keepGsf;
	    }
	    else {
	      if(minBremDphi < 0.05) 
		secondaries.push_back(igsf);
	    }
	  } // end elseif
	  else {
	    if(minBremDphi < 0.05) 
	      secondaries.push_back(igsf);
	  }
	}
      }
    }
  }

  return n_keepGsf;
}
float 
PFElecTkProducer::selectSecondaries(const reco::GsfPFRecTrack primGsf,
				    const reco::GsfPFRecTrack secGsf) {
  //   bool isValidSecondary = false;
  

  float minDeta = 1000.; 
  float minDphi = 1000.; 

  // possible other selections
  // secondary p < primary p 
  
  // temporary: discard gsf with pMode() > 49 
  // or KF pt>49  (no pre-ided)
 
  PFRecTrackRef refsecKF =  secGsf.kfPFRecTrackRef();
  if(refsecKF.isNonnull()) {
    TrackRef kfref = refsecKF->trackRef();
    if(kfref->pt() > maxPtConvReco_)
      return minDphi;
  }
  
  reco::GsfTrackRef secGsfTrack = secGsf.gsfTrackRef();
  if(secGsfTrack->ptMode() > maxPtConvReco_)
    return minDphi;
  


  std::vector<reco::PFBrem> primPFBrem = primGsf.PFRecBrem();
  std::vector<reco::PFBrem> secPFBrem = secGsf.PFRecBrem();


  unsigned int cbrem = 0;

  for (unsigned isbrem = 0; isbrem < secPFBrem.size(); isbrem++) {
    if(secPFBrem[isbrem].indTrajPoint() == 99) continue;
    const reco::PFTrajectoryPoint& atSecECAL 
      = secPFBrem[isbrem].extrapolatedPoint( reco::PFTrajectoryPoint::ECALEntrance );
    if( ! atSecECAL.isValid() ) continue;
    float secEta = atSecECAL.positionREP().Eta();
    float secPhi  = atSecECAL.positionREP().Phi();


    for(unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
      if(primPFBrem[ipbrem].indTrajPoint() == 99) continue;
      const reco::PFTrajectoryPoint& atPrimECAL 
	= primPFBrem[ipbrem].extrapolatedPoint( reco::PFTrajectoryPoint::ECALEntrance );
      if( ! atPrimECAL.isValid() ) continue;
      float primEta = atPrimECAL.positionREP().Eta();
      float primPhi = atPrimECAL.positionREP().Phi();
      
      float deta = fabs(primEta - secEta);
      float dphi = fabs(primPhi - secPhi);
      if (dphi>TMath::Pi()) dphi-= TMath::TwoPi();     
      if(fabs(dphi) < minDphi) {	   
	minDeta = deta;
	minDphi = fabs(dphi);
      }
    }
    
    
    cbrem++;
    if(cbrem == 2) 
      break;
  }
  return minDphi;
}
// ------------ method called once each job just before starting event loop  ------------
void 
PFElecTkProducer::beginRun(edm::Run& run,
			   const EventSetup& iSetup)
{
  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

  ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);


  mtsTransform_ = MultiTrajectoryStateTransform(tracker.product(),magneticField.product());
  

  pfTransformer_= new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0,0,0))));
  
  
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  TransientTrackBuilder thebuilder = *(builder.product());
  

  if(useConvBremFinder_) {
    FILE * fileConvBremID = fopen(path_mvaWeightFileConvBrem_.c_str(), "r");
    if (fileConvBremID) {
      fclose(fileConvBremID);
    }
    else {
      string err = "PFElecTkProducer: cannot open weight file '";
      err += path_mvaWeightFileConvBrem_;
      err += "'";
      throw invalid_argument( err );
    }
  }
  convBremFinder_ = new ConvBremPFTrackFinder(thebuilder,mvaConvBremFinderID_,path_mvaWeightFileConvBrem_);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFElecTkProducer::endRun() {
  delete pfTransformer_;
}

//define this as a plug-in
