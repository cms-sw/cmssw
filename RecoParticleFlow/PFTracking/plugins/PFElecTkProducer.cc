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
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"

#include "TMath.h"
using namespace std;
using namespace edm;
using namespace reco;
PFElecTkProducer::PFElecTkProducer(const ParameterSet& iConfig, const convbremhelpers::HeavyObjectCache*):
  conf_(iConfig)
{
  

  gsfTrackLabel_ = consumes<reco::GsfTrackCollection>(iConfig.getParameter<InputTag>
						      ("GsfTrackModuleLabel"));

  pfTrackLabel_ = consumes<reco::PFRecTrackCollection>(iConfig.getParameter<InputTag>
						       ("PFRecTrackLabel"));

  primVtxLabel_ = consumes<reco::VertexCollection>(iConfig.getParameter<InputTag>
						   ("PrimaryVertexLabel"));

  pfEcalClusters_ = consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>
							("PFEcalClusters"));

  pfNuclear_ = consumes<reco::PFDisplacedTrackerVertexCollection>(iConfig.getParameter<InputTag>
								  ("PFNuclear"));
  
  pfConv_ = consumes<reco::PFConversionCollection>(iConfig.getParameter<InputTag>
						   ("PFConversions"));
  
  pfV0_ = consumes<reco::PFV0Collection>(iConfig.getParameter<InputTag>
					 ("PFV0"));

  useNuclear_ = iConfig.getParameter<bool>("useNuclear");
  useConversions_ = iConfig.getParameter<bool>("useConversions");
  useV0_ = iConfig.getParameter<bool>("useV0");
  debugGsfCleaning_ = iConfig.getParameter<bool>("debugGsfCleaning");

  produces<GsfPFRecTrackCollection>();
  produces<GsfPFRecTrackCollection>( "Secondary" ).setBranchAlias( "secondary" );


  trajinev_ = iConfig.getParameter<bool>("TrajInEvents");
  modemomentum_ = iConfig.getParameter<bool>("ModeMomentum");
  applySel_ = iConfig.getParameter<bool>("applyEGSelection");
  applyGsfClean_ = iConfig.getParameter<bool>("applyGsfTrackCleaning");
  applyAngularGsfClean_ = iConfig.getParameter<bool>("applyAlsoGsfAngularCleaning");
  detaCutGsfClean_ =  iConfig.getParameter<double>("maxDEtaGsfAngularCleaning");
  dphiCutGsfClean_ =  iConfig.getParameter<double>("maxDPhiBremTangGsfAngularCleaning");
  useFifthStepForTrackDriven_ = iConfig.getParameter<bool>("useFifthStepForTrackerDrivenGsf");
  useFifthStepForEcalDriven_ = iConfig.getParameter<bool>("useFifthStepForEcalDrivenGsf");
  maxPtConvReco_ = iConfig.getParameter<double>("MaxConvBremRecoPT");
  detaGsfSC_ = iConfig.getParameter<double>("MinDEtaGsfSC");
  dphiGsfSC_ = iConfig.getParameter<double>("MinDPhiGsfSC");
  SCEne_ = iConfig.getParameter<double>("MinSCEnergy");
  
  // set parameter for convBremFinder
  useConvBremFinder_ =     iConfig.getParameter<bool>("useConvBremFinder");
  
  mvaConvBremFinderIDBarrelLowPt_ = iConfig.getParameter<double>("pf_convBremFinderID_mvaCutBarrelLowPt");
  mvaConvBremFinderIDBarrelHighPt_ = iConfig.getParameter<double>("pf_convBremFinderID_mvaCutBarrelHighPt");
  mvaConvBremFinderIDEndcapsLowPt_ = iConfig.getParameter<double>("pf_convBremFinderID_mvaCutEndcapsLowPt");
  mvaConvBremFinderIDEndcapsHighPt_ = iConfig.getParameter<double>("pf_convBremFinderID_mvaCutEndcapsHighPt");
  
}


PFElecTkProducer::~PFElecTkProducer()
{ }


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PFElecTkProducer::produce(Event& iEvent, const EventSetup& iSetup)
{


  //create the empty collections 
  auto_ptr< GsfPFRecTrackCollection > 
    gsfPFRecTrackCollection(new GsfPFRecTrackCollection);

  
  auto_ptr< GsfPFRecTrackCollection > 
    gsfPFRecTrackCollectionSecondary(new GsfPFRecTrackCollection);

  //read collections of tracks
  Handle<GsfTrackCollection> gsftrackscoll;
  iEvent.getByToken(gsfTrackLabel_,gsftrackscoll);

  //read collections of trajectories
  Handle<vector<Trajectory> > TrajectoryCollection;
 
  //read pfrectrack collection
  Handle<PFRecTrackCollection> thePfRecTrackCollection;
  iEvent.getByToken(pfTrackLabel_,thePfRecTrackCollection);
  const PFRecTrackCollection& PfRTkColl = *(thePfRecTrackCollection.product());

  // PFClusters
  Handle<PFClusterCollection> theECPfClustCollection;
  iEvent.getByToken(pfEcalClusters_,theECPfClustCollection);
  const PFClusterCollection& theEcalClusters = *(theECPfClustCollection.product());

  //Primary Vertexes
  Handle<reco::VertexCollection> thePrimaryVertexColl;
  iEvent.getByToken(primVtxLabel_,thePrimaryVertexColl);
  
  // Displaced Vertex
  Handle< reco::PFDisplacedTrackerVertexCollection > pfNuclears; 
  if( useNuclear_ ) 
    iEvent.getByToken(pfNuclear_, pfNuclears);
  
  // Conversions 
  Handle< reco::PFConversionCollection > pfConversions;
  if( useConversions_ ) 
    iEvent.getByToken(pfConv_,pfConversions);

  // V0
  Handle< reco::PFV0Collection > pfV0;
  if( useV0_ ) 
    iEvent.getByToken(pfV0_, pfV0);
    
      
  

  GsfTrackCollection gsftracks = *(gsftrackscoll.product());	
  vector<Trajectory> tjvec(0);
  if (trajinev_){
    iEvent.getByToken(gsfTrackLabel_,TrajectoryCollection); 
    
    tjvec= *(TrajectoryCollection.product());
  }
  
  
  vector<reco::GsfPFRecTrack> selGsfPFRecTracks;
  vector<reco::GsfPFRecTrack> primaryGsfPFRecTracks;
  std::map<unsigned int, std::vector<reco::GsfPFRecTrack> >  GsfPFMap;
  

  for (unsigned int igsf=0; igsf<gsftracks.size();igsf++) {
    
    GsfTrackRef trackRef(gsftrackscoll, igsf);

    int kf_ind=FindPfRef(PfRTkColl,gsftracks[igsf],false);
    
    if (kf_ind>=0) {
      
      PFRecTrackRef kf_ref(thePfRecTrackCollection,
			   kf_ind);

      // remove fifth step tracks
      if( useFifthStepForEcalDriven_ == false
	  || useFifthStepForTrackDriven_ == false) {
	bool isFifthStepTrack = isFifthStep(kf_ref);
	bool isEcalDriven = true;
	bool isTrackerDriven = true;
	
	if (&(*trackRef->seedRef())==0) {
	  isEcalDriven = false;
	  isTrackerDriven = false;
	}
	else {
	  auto const& SeedFromRef= dynamic_cast<ElectronSeed const&>(*(trackRef->extra()->seedRef()) );
	  if(SeedFromRef.caloCluster().isNull())
	    isEcalDriven = false;
	  if(SeedFromRef.ctfTrack().isNull())
	    isTrackerDriven = false;
	}
	//note: the same track could be both ecalDriven and trackerDriven
	if(isFifthStepTrack && 
	   isEcalDriven &&
	   isTrackerDriven == false &&
	   useFifthStepForEcalDriven_ == false) {
	  continue;
	}

	if(isFifthStepTrack && 
	   isTrackerDriven  && 
	   isEcalDriven == false && 
	   useFifthStepForTrackDriven_ == false) {
	  continue;
	}

	if(isFifthStepTrack && 
	   isTrackerDriven && 
	   isEcalDriven && 
	   useFifthStepForTrackDriven_ == false &&
	   useFifthStepForEcalDriven_ == false) {
	  continue;
	}
      }
      
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
  
  
  unsigned int count_primary = 0;
  if(selGsfPFRecTracks.size() > 0) {
    for(unsigned int ipfgsf=0; ipfgsf<selGsfPFRecTracks.size();ipfgsf++) {
      
      vector<unsigned int> secondaries(0);
      secondaries.clear();
      bool keepGsf = true;

      if(applyGsfClean_) {
	keepGsf = resolveGsfTracks(selGsfPFRecTracks,ipfgsf,secondaries,theEcalClusters);
      }
 
      //is primary? 
      if(keepGsf == true) {

	// Find kf tracks from converted brem photons
	if(convBremFinder_->foundConvBremPFRecTrack(thePfRecTrackCollection,thePrimaryVertexColl,
						    pfNuclears,pfConversions,pfV0,
                                                    globalCache(),
						    useNuclear_,useConversions_,useV0_,
						    theEcalClusters,selGsfPFRecTracks[ipfgsf])) {
	  const vector<PFRecTrackRef>& convBremPFRecTracks(convBremFinder_->getConvBremPFRecTracks());
	  for(unsigned int ii = 0; ii<convBremPFRecTracks.size(); ii++) {
	    selGsfPFRecTracks[ipfgsf].addConvBremPFRecTrackRef(convBremPFRecTracks[ii]);
	  }
	}
	
	// save primaries gsf tracks
	//	gsfPFRecTrackCollection->push_back(selGsfPFRecTracks[ipfgsf]);
	primaryGsfPFRecTracks.push_back(selGsfPFRecTracks[ipfgsf]);
	
	
	// NOTE:: THE TRACKID IS USED TO LINK THE PRIMARY GSF TRACK. THIS NEEDS 
	// TO BE CHANGED AS SOON AS IT IS POSSIBLE TO CHANGE DATAFORMATS
	// A MODIFICATION HERE IMPLIES A MODIFICATION IN PFBLOCKALGO.CC/H
	unsigned int primGsfIndex = selGsfPFRecTracks[ipfgsf].trackId();
	vector<reco::GsfPFRecTrack> trueGsfPFRecTracks;
	if(secondaries.size() > 0) {
	  // loop on secondaries gsf tracks (from converted brems)
	  for(unsigned int isecpfgsf=0; isecpfgsf<secondaries.size();isecpfgsf++) {
	    
	    PFRecTrackRef refsecKF =  selGsfPFRecTracks[(secondaries[isecpfgsf])].kfPFRecTrackRef();
	    
	    unsigned int secGsfIndex = selGsfPFRecTracks[(secondaries[isecpfgsf])].trackId();
	    GsfTrackRef secGsfRef = selGsfPFRecTracks[(secondaries[isecpfgsf])].gsfTrackRef();

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

	    if(validgsfbrem) {
	      gsfPFRecTrackCollectionSecondary->push_back(secpftrack_);
	      trueGsfPFRecTracks.push_back(secpftrack_);
	    }	    
	  }
	}
	GsfPFMap.insert(pair<unsigned int,std::vector<reco::GsfPFRecTrack> >(count_primary,trueGsfPFRecTracks));
	trueGsfPFRecTracks.clear();
	count_primary++;
      }
    }
  }
  
  
  const edm::OrphanHandle<GsfPFRecTrackCollection> gsfPfRefProd = 
    iEvent.put(gsfPFRecTrackCollectionSecondary,"Secondary");
  
  
  //now the secondary GsfPFRecTracks are in the event, the Ref can be created
  createGsfPFRecTrackRef(gsfPfRefProd,primaryGsfPFRecTracks,GsfPFMap);
  
  for(unsigned int iGSF = 0; iGSF<primaryGsfPFRecTracks.size();iGSF++){
    gsfPFRecTrackCollection->push_back(primaryGsfPFRecTracks[iGSF]);
  }
  iEvent.put(gsfPFRecTrackCollection);

  selGsfPFRecTracks.clear();
  GsfPFMap.clear();
  primaryGsfPFRecTracks.clear();
}
  
// create the secondary GsfPFRecTracks Ref
void
PFElecTkProducer::createGsfPFRecTrackRef(const edm::OrphanHandle<reco::GsfPFRecTrackCollection>& gsfPfHandle,
					 std::vector<reco::GsfPFRecTrack>& gsfPFRecTrackPrimary,
					 const std::map<unsigned int, std::vector<reco::GsfPFRecTrack> >& MapPrimSec) {
  unsigned int cgsf=0;
  unsigned int csecgsf=0;
  for (std::map<unsigned int, std::vector<reco::GsfPFRecTrack> >::const_iterator igsf = MapPrimSec.begin();
       igsf != MapPrimSec.end(); igsf++,cgsf++) {
    vector<reco::GsfPFRecTrack> SecGsfPF = igsf->second;
    for (unsigned int iSecGsf=0; iSecGsf < SecGsfPF.size(); iSecGsf++) {
      edm::Ref<reco::GsfPFRecTrackCollection> refgprt(gsfPfHandle,csecgsf);
      gsfPFRecTrackPrimary[cgsf].addConvBremGsfPFRecTrackRef(refgprt);
      ++csecgsf;
    }
  }

  return;
}
// ------------- method for find the corresponding kf pfrectrack ---------------------
int
PFElecTkProducer::FindPfRef(const reco::PFRecTrackCollection  & PfRTkColl, 
			    const reco::GsfTrack& gsftk,
			    bool otherColl){


  if (&(*gsftk.seedRef())==0) return -1;
  auto const &  ElSeedFromRef=dynamic_cast<ElectronSeed const&>( *(gsftk.extra()->seedRef()) );
  //CASE 1 ELECTRONSEED DOES NOT HAVE A REF TO THE CKFTRACK
  if (ElSeedFromRef.ctfTrack().isNull()){
    reco::PFRecTrackCollection::const_iterator pft=PfRTkColl.begin();
    reco::PFRecTrackCollection::const_iterator pftend=PfRTkColl.end();
    unsigned int i_pf=0;
    int ibest=-1;
    unsigned int ish_max=0;
    float dr_min=1000;
    //SEARCH THE PFRECTRACK THAT SHARES HITS WITH THE ELECTRON SEED
    // Here the cpu time can be improved. 
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
      if (pft->trackRef()==ElSeedFromRef.ctfTrack()){
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
  case TrackBase::undefAlgorithm:
  case TrackBase::ctf:
  case TrackBase::initialStep:
  case TrackBase::lowPtTripletStep:
  case TrackBase::pixelPairStep:
  case TrackBase::jetCoreRegionalStep:
  case TrackBase::muonSeededStepInOut:
  case TrackBase::muonSeededStepOutIn:
    Algo = 0;
    break;
  case TrackBase::detachedTripletStep:
    Algo = 1;
    break;
  case TrackBase::mixedTripletStep:
    Algo = 2;
    break;
  case TrackBase::pixelLessStep:
    Algo = 3;
    break;
  case TrackBase::tobTecStep:
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
PFElecTkProducer::applySelection(const reco::GsfTrack& gsftk) {
  if (&(*gsftk.seedRef())==0) return false;
  auto const& ElSeedFromRef=dynamic_cast<ElectronSeed const&>( *(gsftk.extra()->seedRef()) );

  bool passCut = false;
  if (ElSeedFromRef.ctfTrack().isNull()){
    if(ElSeedFromRef.caloCluster().isNull()) return passCut;
    auto const* scRef = dynamic_cast<SuperCluster const*>(ElSeedFromRef.caloCluster().get());
    //do this just to know if exist a SC? 
    if(scRef) {
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
				   vector<unsigned int> &secondaries,
				   const reco::PFClusterCollection & theEClus) {
  bool debugCleaning = debugGsfCleaning_;
  bool n_keepGsf = true;

  reco::GsfTrackRef nGsfTrack = GsfPFVec[ngsf].gsfTrackRef();
  
  if (&(*nGsfTrack->seedRef())==0) return false;    
  auto const& nElSeedFromRef=dynamic_cast<ElectronSeed const&>( *(nGsfTrack->extra()->seedRef()) );


  TrajectoryStateOnSurface inTSOS = mtsTransform_.innerStateOnSurface((*nGsfTrack));
  GlobalVector ninnMom;
  float nPin =  nGsfTrack->pMode();
  if(inTSOS.isValid()){
    mtsMode_->momentumFromModeCartesian(inTSOS,ninnMom);
    nPin = ninnMom.mag();
  }

  float neta = nGsfTrack->innerMomentum().eta();
  float nphi = nGsfTrack->innerMomentum().phi();
  


  
  if(debugCleaning)
    cout << " PFElecTkProducer:: considering track " << nGsfTrack->pt() 
	 << " eta,phi " <<  nGsfTrack->eta() << ", " <<  nGsfTrack->phi()  << endl;
  
  
  for (unsigned int igsf=0; igsf< GsfPFVec.size();igsf++) {
    if(igsf != ngsf ) {
      reco::GsfTrackRef iGsfTrack = GsfPFVec[igsf].gsfTrackRef();

      if(debugCleaning)
	cout << " PFElecTkProducer:: and  comparing with track " << iGsfTrack->pt() 
	     << " eta,phi " <<  iGsfTrack->eta() << ", " <<  iGsfTrack->phi()  << endl;
      
      float ieta = iGsfTrack->innerMomentum().eta();
      float iphi = iGsfTrack->innerMomentum().phi();
      float feta = fabs(neta - ieta);
      float fphi = fabs(nphi - iphi);
      if (fphi>TMath::Pi()) fphi-= TMath::TwoPi();     
  

      // apply a superloose preselection only to avoid un-useful cpu time: hard-coded for this reason
      if(feta < 0.5 && fabs(fphi) < 1.0) {
	if(debugCleaning)
	  cout << " Entering angular superloose preselection " << endl;
	
	TrajectoryStateOnSurface i_inTSOS = mtsTransform_.innerStateOnSurface((*iGsfTrack));
	GlobalVector i_innMom;
	float iPin = iGsfTrack->pMode();
	if(i_inTSOS.isValid()){
	  mtsMode_->momentumFromModeCartesian(i_inTSOS,i_innMom);  
	  iPin = i_innMom.mag();
	}

	if (&(*iGsfTrack->seedRef())==0) continue;   
	auto const& iElSeedFromRef=dynamic_cast<ElectronSeed const&>( *(iGsfTrack->extra()->seedRef()) );

	float SCEnergy = -1.;
	// Check if two tracks match the same SC     
	bool areBothGsfEcalDriven = false;;
	bool isSameSC = isSameEgSC(nElSeedFromRef,iElSeedFromRef,areBothGsfEcalDriven,SCEnergy);
	
	// CASE1 both GsfTracks ecalDriven and match the same SC
	if(areBothGsfEcalDriven ) {
	  if(isSameSC) {
	    float nEP = SCEnergy/nPin;
	    float iEP =  SCEnergy/iPin;
	    if(debugCleaning)
	      cout << " Entering SAME supercluster case " 
		   << " nEP " << nEP 
		   << " iEP " << iEP << endl;
	    
	    
	    
	    // if same SC take the closest or if same
	    // distance the best E/p
	    
	    // Innermost using LostHits technology
	    bool isSameLayer = false;
	    bool iGsfInnermostWithLostHits = 
	      isInnerMostWithLostHits(nGsfTrack,iGsfTrack,isSameLayer);
	    
	    
	    if(debugCleaning)
	      cout << " iGsf is InnerMostWithLostHits " << iGsfInnermostWithLostHits
		   << " isSameLayer " << isSameLayer  << endl;
	    
	    if (iGsfInnermostWithLostHits) {
	      n_keepGsf = false;
	      return n_keepGsf;
	    }
	    else if(isSameLayer){
	      if(fabs(iEP-1) < fabs(nEP-1)) {
		n_keepGsf = false;
		return n_keepGsf;
	      }
	      else{
		secondaries.push_back(igsf);
	      }
	    }
	    else {
	      // save secondaries gsf track (put selection)
	    secondaries.push_back(igsf);
	    }
	  } // end same SC case
	}
	else {
	  // enter in the condition where at least one track is trackerDriven
	  float minBremDphi =  minTangDist(GsfPFVec[ngsf],GsfPFVec[igsf]);
	  float nETot = 0.;
	  float iETot = 0.;
	  bool isBothGsfTrackerDriven = false;
	  bool nEcalDriven = false;
	  bool iEcalDriven = false;
	  bool isSameScEgPf = isSharingEcalEnergyWithEgSC(GsfPFVec[ngsf],
							  GsfPFVec[igsf],
							  nElSeedFromRef,
							  iElSeedFromRef,
							  theEClus,
							  isBothGsfTrackerDriven,
							  nEcalDriven,
							  iEcalDriven,
							  nETot,
							  iETot);

	  // check if the first hit of iGsfTrack < nGsfTrack	      
	  bool isSameLayer = false;
	  bool iGsfInnermostWithLostHits = 
	    isInnerMostWithLostHits(nGsfTrack,iGsfTrack,isSameLayer);

	  if(isSameScEgPf) {
	    // CASE 2 : One Gsf has reference to a SC and the other one not or both not
	   	    
	    if(debugCleaning) {
	      cout << " Sharing ECAL energy passed " 
		   << " nEtot " << nETot 
		   << " iEtot " << iETot << endl;
	      if(isBothGsfTrackerDriven) 
		cout << " Both Track are trackerDriven " << endl;
	    }

	    // Innermost using LostHits technology
	    if (iGsfInnermostWithLostHits) {
	      n_keepGsf = false;
	      return n_keepGsf;
	    }
	    else if(isSameLayer){
	      // Thirt Case:  One Gsf has reference to a SC and the other one not or both not
	      // gsf tracks starts from the same layer
	      // check number of sharing modules (at least 50%)
	      // check number of sharing hits (at least 2)
	      // check charge flip inner/outer

	     
		// they share energy
	      if(isBothGsfTrackerDriven == false) {
		// if at least one Gsf track is EcalDriven choose that one.
		if(iEcalDriven) {
		  n_keepGsf = false;
		  return n_keepGsf;
		}
		else {
		  secondaries.push_back(igsf);
		}
	      }
	      else {
		// if both tracks are tracker driven choose that one with the best E/p
		// with ETot = max(En,Ei)

		float ETot = -1;
		if(nETot != iETot) {
		  if(nETot > iETot)
		    ETot = nETot;
		  else 
		    ETot = iETot;
		}
		else {
		  ETot = nETot;
		}
		float nEP = ETot/nPin;
		float iEP = ETot/iPin;
		
		
		if(debugCleaning) 
		  cout << " nETot " << nETot
		       << " iETot " << iETot 
		       << " ETot " << ETot << endl 
		       << " nPin " << nPin
		       << " iPin " << iPin 
		       << " nEP " << nEP 
		       << " iEP " << iEP << endl;
		
	
		if(fabs(iEP-1) < fabs(nEP-1)) {
		  n_keepGsf = false;
		  return n_keepGsf;
		}
		else{
		  secondaries.push_back(igsf);
		}
	      }
	    }
	    else {
	      secondaries.push_back(igsf);
	    }
	  }
	  else if(feta < detaCutGsfClean_ && minBremDphi < dphiCutGsfClean_) {
	    // very close tracks
	    bool secPushedBack = false;
	    if(nEcalDriven == false && nETot == 0.) {
	      n_keepGsf = false;
	      return n_keepGsf;
	    }
	    else if(iEcalDriven == false && iETot == 0.) {
	      secondaries.push_back(igsf);
	      secPushedBack = true;
	    }
	    if(debugCleaning)
	      cout << " Close Tracks " 
		   << " feta " << feta << " fabs(fphi) " << fabs(fphi) 
		   << " minBremDphi " <<  minBremDphi 
		   << " nETot " << nETot 
		   << " iETot " << iETot 
		   << " nLostHits " <<  nGsfTrack->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS) 
		   << " iLostHits " << iGsfTrack->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS) << endl;
	    
	    // apply selection only if one track has lost hits
	    if(applyAngularGsfClean_) {
	      if (iGsfInnermostWithLostHits) {
		n_keepGsf = false;
		return n_keepGsf;
	      }
	      else if(isSameLayer == false) {
		if(secPushedBack == false) 
		  secondaries.push_back(igsf);
	      }
	    }
	  }
	  else if(feta < 0.1 && minBremDphi < 0.2){
	    // failed all the conditions, discard only tracker driven tracks
	    // with no PFClusters linked. 
	    if(debugCleaning)
	      cout << " Close Tracks and failed all the conditions " 
		   << " feta " << feta << " fabs(fphi) " << fabs(fphi) 
		   << " minBremDphi " <<  minBremDphi 
		   << " nETot " << nETot 
		   << " iETot " << iETot 
		   << " nLostHits " <<  nGsfTrack->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS) 
		   << " iLostHits " << iGsfTrack->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS) << endl;
	    
	    if(nEcalDriven == false && nETot == 0.) {
	      n_keepGsf = false;
	      return n_keepGsf;
	    }
	    // Here I do not push back the secondary because considered fakes...
	  }
	}
      }
    }
  }
  
  return n_keepGsf;
}
float 
PFElecTkProducer::minTangDist(const reco::GsfPFRecTrack& primGsf,
			      const reco::GsfPFRecTrack& secGsf) {

  float minDphi = 1000.; 


  std::vector<reco::PFBrem> primPFBrem = primGsf.PFRecBrem();
  std::vector<reco::PFBrem> secPFBrem = secGsf.PFRecBrem();


  unsigned int cbrem = 0;
  for (unsigned isbrem = 0; isbrem < secPFBrem.size(); isbrem++) {
    if(secPFBrem[isbrem].indTrajPoint() == 99) continue;
    const reco::PFTrajectoryPoint& atSecECAL 
      = secPFBrem[isbrem].extrapolatedPoint( reco::PFTrajectoryPoint::ECALEntrance );
    if( ! atSecECAL.isValid() ) continue;
    float secPhi  = atSecECAL.positionREP().Phi();

    unsigned int sbrem = 0;
    for(unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
      if(primPFBrem[ipbrem].indTrajPoint() == 99) continue;
      const reco::PFTrajectoryPoint& atPrimECAL 
	= primPFBrem[ipbrem].extrapolatedPoint( reco::PFTrajectoryPoint::ECALEntrance );
      if( ! atPrimECAL.isValid() ) continue;
      sbrem++;
      if(sbrem <= 3) {
	float primPhi = atPrimECAL.positionREP().Phi();
	
	float dphi = fabs(primPhi - secPhi);
	if (dphi>TMath::Pi()) dphi-= TMath::TwoPi();     
	if(fabs(dphi) < minDphi) {	   
	  minDphi = fabs(dphi);
	}
      }
    }
    
    
    cbrem++;
    if(cbrem == 3) 
      break;
  }
  return minDphi;
}
bool 
PFElecTkProducer::isSameEgSC(const reco::ElectronSeed& nSeed,
			     const reco::ElectronSeed& iSeed,
			     bool& bothGsfEcalDriven,
			     float& SCEnergy) {
  
  bool isSameSC = false;

  if(nSeed.caloCluster().isNonnull() && iSeed.caloCluster().isNonnull()) {
    auto const* nscRef = dynamic_cast<SuperCluster const*>(nSeed.caloCluster().get());
    auto const* iscRef = dynamic_cast<SuperCluster const*>(iSeed.caloCluster().get());

    if(nscRef && iscRef) {
      bothGsfEcalDriven = true;
      if(nscRef == iscRef) {
	isSameSC = true;
	// retrieve the supercluster energy
	SCEnergy = nscRef->energy();
      }
    }
  }
  return isSameSC;
}
bool 
PFElecTkProducer::isSharingEcalEnergyWithEgSC(const reco::GsfPFRecTrack& nGsfPFRecTrack,
					      const reco::GsfPFRecTrack& iGsfPFRecTrack,
					      const reco::ElectronSeed& nSeed,
					      const reco::ElectronSeed& iSeed,
					      const reco::PFClusterCollection& theEClus,
					      bool& bothGsfTrackerDriven,
					      bool& nEcalDriven,
					      bool& iEcalDriven,
					      float& nEnergy,
					      float& iEnergy) {
  
  bool isSharingEnergy = false;

  //which is EcalDriven?
  bool oneEcalDriven = true;
  SuperCluster const* scRef = nullptr;
  GsfPFRecTrack gsfPfTrack;

  if(nSeed.caloCluster().isNonnull()) {
    scRef = dynamic_cast<SuperCluster const*>(nSeed.caloCluster().get());
    assert(scRef);
    nEnergy = scRef->energy();
    nEcalDriven = true;
    gsfPfTrack = iGsfPFRecTrack;
  }
  else if(iSeed.caloCluster().isNonnull()){
    scRef = dynamic_cast<SuperCluster const*>(iSeed.caloCluster().get());
    assert(scRef);
    iEnergy = scRef->energy();
    iEcalDriven = true;
    gsfPfTrack = nGsfPFRecTrack;
  }
  else{
     oneEcalDriven = false;
  }
  
  if(oneEcalDriven) {
    //run a basic reconstruction for the particle flow

    vector<PFCluster> vecPFClusters;
    vecPFClusters.clear();

    for (PFClusterCollection::const_iterator clus = theEClus.begin();
	 clus != theEClus.end();
	 clus++ ) {
      PFCluster clust = *clus;
      clust.calculatePositionREP();

      float deta = fabs(scRef->position().eta() - clust.position().eta());
      float dphi = fabs(scRef->position().phi() - clust.position().phi());
      if (dphi>TMath::Pi()) dphi-= TMath::TwoPi();
 
      // Angle preselection between the supercluster and pfclusters
      // this is needed just to save some cpu-time for this is hard-coded     
      if(deta < 0.5 && fabs(dphi) < 1.0) {
	bool foundLink = false;
	double distGsf = gsfPfTrack.extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax ).isValid() ?
	  LinkByRecHit::testTrackAndClusterByRecHit(gsfPfTrack , clust ) : -1.;
	// check if it touch the GsfTrack
	if(distGsf > 0.) {
	  if(nEcalDriven) 
	    iEnergy += clust.energy();
	  else
	    nEnergy += clust.energy();
	  vecPFClusters.push_back(clust);
	  foundLink = true;
	}
	// check if it touch the Brem-tangents
	if(foundLink == false) {
	  vector<PFBrem> primPFBrem = gsfPfTrack.PFRecBrem();
	  for(unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
	    if(primPFBrem[ipbrem].indTrajPoint() == 99) continue;
	    const reco::PFRecTrack& pfBremTrack = primPFBrem[ipbrem];
	    double dist = pfBremTrack.extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax ).isValid() ?
	      LinkByRecHit::testTrackAndClusterByRecHit(pfBremTrack , clust, true ) : -1.;
	    if(dist > 0.) {
	      if(nEcalDriven) 
		iEnergy += clust.energy();
	      else
		nEnergy += clust.energy();
	      vecPFClusters.push_back(clust);
	      foundLink = true;
	    }
	  }	
	}  
      } // END if anble preselection
    } // PFClusters Loop
    if(vecPFClusters.size() > 0 ) {
      for(unsigned int pf = 0; pf < vecPFClusters.size(); pf++) {
	bool isCommon = ClusterClusterMapping::overlap(vecPFClusters[pf],*scRef);
	if(isCommon) {
	  isSharingEnergy = true;
	}
	break;
      }
    }
  }
  else {
    // both tracks are trackerDriven, try ECAL energy matching also in this case.

    bothGsfTrackerDriven = true;
    vector<PFCluster> nPFCluster;
    vector<PFCluster> iPFCluster;

    nPFCluster.clear();
    iPFCluster.clear();

    for (PFClusterCollection::const_iterator clus = theEClus.begin();
	 clus != theEClus.end();
	 clus++ ) {
      PFCluster clust = *clus;
      clust.calculatePositionREP();
      
      float ndeta = fabs(nGsfPFRecTrack.gsfTrackRef()->eta() - clust.position().eta());
      float ndphi = fabs(nGsfPFRecTrack.gsfTrackRef()->phi() - clust.position().phi());
      if (ndphi>TMath::Pi()) ndphi-= TMath::TwoPi();
      // Apply loose preselection with the track
      // just to save cpu time, for this hard-coded
      if(ndeta < 0.5 && fabs(ndphi) < 1.0) {
	bool foundNLink = false;
	
	double distGsf = nGsfPFRecTrack.extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax ).isValid() ?
	  LinkByRecHit::testTrackAndClusterByRecHit(nGsfPFRecTrack , clust ) : -1.;
	if(distGsf > 0.) {
	  nPFCluster.push_back(clust);
	  nEnergy += clust.energy();
	  foundNLink = true;
	}
	if(foundNLink == false) {
	  const vector<PFBrem>& primPFBrem = nGsfPFRecTrack.PFRecBrem();
	  for(unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
	    if(primPFBrem[ipbrem].indTrajPoint() == 99) continue;
	    const reco::PFRecTrack& pfBremTrack = primPFBrem[ipbrem];
	    if(foundNLink == false) {
	      double dist = pfBremTrack.extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax ).isValid() ?
		LinkByRecHit::testTrackAndClusterByRecHit(pfBremTrack , clust, true ) : -1.;
	      if(dist > 0.) {
		nPFCluster.push_back(clust);
		nEnergy += clust.energy();
		foundNLink = true;
	      }
	    }
	  }
	}
      }

      float ideta = fabs(iGsfPFRecTrack.gsfTrackRef()->eta() - clust.position().eta());
      float idphi = fabs(iGsfPFRecTrack.gsfTrackRef()->phi() - clust.position().phi());
      if (idphi>TMath::Pi()) idphi-= TMath::TwoPi();
      // Apply loose preselection with the track
      // just to save cpu time, for this hard-coded
      if(ideta < 0.5 && fabs(idphi) < 1.0) {
	bool foundILink = false;
	double dist = iGsfPFRecTrack.extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax ).isValid() ?
	  LinkByRecHit::testTrackAndClusterByRecHit(iGsfPFRecTrack , clust ) : -1.;
	if(dist > 0.) {
	  iPFCluster.push_back(clust);
	  iEnergy += clust.energy();
	  foundILink = true;
	}
	if(foundILink == false) {
	  vector<PFBrem> primPFBrem = iGsfPFRecTrack.PFRecBrem();
	  for(unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
	    if(primPFBrem[ipbrem].indTrajPoint() == 99) continue;
	    const reco::PFRecTrack& pfBremTrack = primPFBrem[ipbrem];
	    if(foundILink == false) {
	      double dist = LinkByRecHit::testTrackAndClusterByRecHit(pfBremTrack , clust, true );
	      if(dist > 0.) {
		iPFCluster.push_back(clust);
		iEnergy += clust.energy();
		foundILink = true;
	      }
	    }
	  }
	}
      }
    }


    if(nPFCluster.size() > 0 && iPFCluster.size() > 0) {
      for(unsigned int npf = 0; npf < nPFCluster.size(); npf++) {
	for(unsigned int ipf = 0; ipf < iPFCluster.size(); ipf++) {
	  bool isCommon = ClusterClusterMapping::overlap(nPFCluster[npf],iPFCluster[ipf]);
	  if(isCommon) {
	    isSharingEnergy = true;
	    break;
	  }
	}
	if(isSharingEnergy)
	  break;
      }
    }
  }

  return isSharingEnergy;
}
bool PFElecTkProducer::isInnerMost(const reco::GsfTrackRef& nGsfTrack,
				   const reco::GsfTrackRef& iGsfTrack,
				   bool& sameLayer) {
  
  // copied by the class RecoEgamma/EgammaElectronAlgos/src/EgAmbiguityTools.cc
  // obsolete but the code is kept: now using lost hits method

  const reco::HitPattern &gsfHitPattern1 = nGsfTrack->hitPattern();
  const reco::HitPattern &gsfHitPattern2 = iGsfTrack->hitPattern();
  
  // retrieve first valid hit
  int gsfHitCounter1 = 0 ;
  trackingRecHit_iterator elHitsIt1 ;
  for
    ( elHitsIt1 = nGsfTrack->recHitsBegin() ;
     elHitsIt1 != nGsfTrack->recHitsEnd() ;
     elHitsIt1++, gsfHitCounter1++ )
    { if (((**elHitsIt1).isValid())) break ; }
  
  int gsfHitCounter2 = 0 ;
  trackingRecHit_iterator elHitsIt2 ;
  for
    ( elHitsIt2 = iGsfTrack->recHitsBegin() ;
     elHitsIt2 != iGsfTrack->recHitsEnd() ;
     elHitsIt2++, gsfHitCounter2++ )
    { if (((**elHitsIt2).isValid())) break ; }
  
  uint32_t gsfHit1 = gsfHitPattern1.getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter1) ;
  uint32_t gsfHit2 = gsfHitPattern2.getHitPattern(HitPattern::TRACK_HITS, gsfHitCounter2) ;
  
  
  if (gsfHitPattern1.getSubStructure(gsfHit1)!=gsfHitPattern2.getSubStructure(gsfHit2))
   { 
     return (gsfHitPattern2.getSubStructure(gsfHit2)<gsfHitPattern1.getSubStructure(gsfHit1)); 
   }
  else if (gsfHitPattern1.getLayer(gsfHit1)!=gsfHitPattern2.getLayer(gsfHit2))
    { 
      return (gsfHitPattern2.getLayer(gsfHit2)<gsfHitPattern1.getLayer(gsfHit1)); 
    }
  else
   { 
     sameLayer = true;
     return  false; 
   }
}
bool PFElecTkProducer::isInnerMostWithLostHits(const reco::GsfTrackRef& nGsfTrack,
					       const reco::GsfTrackRef& iGsfTrack,
					       bool& sameLayer) {
  
  // define closest using the lost hits on the expectedhitsineer
  unsigned int nLostHits = nGsfTrack->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS);
  unsigned int iLostHits = iGsfTrack->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS);
  
  if (nLostHits!=iLostHits) {
    return (nLostHits > iLostHits);
  } 
  else {
    sameLayer = true;
    return  false; 
  }
}



// ------------ method called once each job just before starting event loop  ------------
void 
PFElecTkProducer::beginRun(const edm::Run& run,
			   const EventSetup& iSetup)
{
  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

  ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  mtsTransform_ = MultiTrajectoryStateTransform(tracker.product(),magneticField.product());  

  pfTransformer_.reset( new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0,0,0)))) );  
  
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  TransientTrackBuilder thebuilder = *(builder.product());
  
  convBremFinder_.reset( new ConvBremPFTrackFinder(thebuilder,
                                                   mvaConvBremFinderIDBarrelLowPt_,
                                                   mvaConvBremFinderIDBarrelHighPt_,
                                                   mvaConvBremFinderIDEndcapsLowPt_, 
                                                   mvaConvBremFinderIDEndcapsHighPt_) );
 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFElecTkProducer::endRun(const edm::Run& run,
			 const EventSetup& iSetup) {
  pfTransformer_.reset();
  convBremFinder_.reset();
}

//define this as a plug-in
