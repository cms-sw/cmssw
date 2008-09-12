#include <iostream>
//
#include "RecoParticleFlow/PFTracking/interface/PFConversionsProducer.h"
// 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "RecoEgamma/EgammaTools/interface/ConversionLikelihoodCalculator.h"
//
#include "TrackingTools/PatternTools/interface/Trajectory.h"
//
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <vector>

using namespace std;
PFConversionsProducer::PFConversionsProducer( const edm::ParameterSet& pset ) : pfTransformer_(0) {
  // input collection names
  conversionCollectionProducer_ = pset.getParameter<std::string>("conversionProducer");
  conversionCollection_ = pset.getParameter<std::string>("conversionCollection");
  debug_ = pset.getParameter<bool>("debug");

  OtherConvLabels_ =pset.getParameter<std::vector< edm::InputTag > >("OtherConversionCollection");
  OtherOutInLabels_ =pset.getParameter<std::vector< edm::InputTag > >("OtherOutInCollection");
  OtherInOutLabels_ =pset.getParameter<std::vector< edm::InputTag > >("OtherInOutCollection");
  // output collection names
  PFConversionCollection_     = pset.getParameter<std::string>("PFConversionCollection");
  PFConversionRecTracks_      = pset.getParameter<std::string>("PFRecTracksFromConversions");  


 // Register the product
  produces<reco::PFConversionCollection>(PFConversionCollection_);
  produces<reco::PFRecTrackCollection>(PFConversionRecTracks_);

}



PFConversionsProducer::~PFConversionsProducer() {
  delete pfTransformer_;
}


void PFConversionsProducer::beginJob( const edm::EventSetup& setup)
{

  nEvt_=0;
  edm::ESHandle<MagneticField> magneticField;
  setup.get<IdealMagneticFieldRecord>().get(magneticField);
  pfTransformer_= new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0,0,0))));

  //  pfTransformer_->OnlyProp();
  return ;
}






void PFConversionsProducer::produce( edm::Event& e, const edm::EventSetup& )
{
  
  
  using namespace edm;
  if (debug_) std::cout <<" PFConversionsProducer Produce event: "<<e.id().event() <<" in run "<<e.id().run()<< std::endl;
 
  cout<<"EVENT "<<e.id()<<endl;
  nEvt_++;  
  
  ///// Get the externally reconstructed  conversions
  Handle<reco::ConversionCollection> conversionHandle; 
  e.getByLabel(conversionCollectionProducer_, conversionCollection_ , conversionHandle);
  if (! conversionHandle.isValid()) {
    edm::LogError("PFConversionsProducer") << "Error! Can't get the product "<<conversionCollection_.c_str();
    return;
  }
  const reco::ConversionCollection conversionCollection = *(conversionHandle.product());
  //  std::cout  << "PFConversionsProducer  input conversions collection size " << conversionCollection.size() << "\n";


   //read collections of trajectories
  Handle<std::vector<Trajectory> > outInTrajectoryHandle;
  e.getByLabel("ckfOutInTracksFromConversions",outInTrajectoryHandle); 
  //  

  Handle<std::vector<Trajectory> > inOutTrajectoryHandle; 
  e.getByLabel("ckfInOutTracksFromConversions",inOutTrajectoryHandle); 


  // read collections of tracks
  Handle<reco::TrackCollection> outInTrkHandle; 
  e.getByLabel("ckfOutInTracksFromConversions",outInTrkHandle); 

  Handle<reco::TrackCollection> inOutTrkHandle; 
  e.getByLabel("ckfInOutTracksFromConversions",inOutTrkHandle); 




  // PFConversion output collection
  reco::PFConversionCollection outputConversionCollection;
  std::auto_ptr<reco::PFConversionCollection> outputConversionCollection_p(new reco::PFConversionCollection);
  // PFRecTracks output collection
  reco::PFRecTrackCollection pfConversionRecTrackCollection;
  std::auto_ptr<reco::PFRecTrackCollection> pfConversionRecTrackCollection_p(new reco::PFRecTrackCollection);

  
  reco::PFRecTrackRefProd pfTrackRefProd = e.getRefBeforePut<reco::PFRecTrackCollection>(PFConversionRecTracks_);
  
  
  int iPfTk=0;
  std::vector<reco::ConversionRef> tmp;
  std::map<reco::CaloClusterPtr, std::vector<reco::ConversionRef> > aMap;
  
  

  for( unsigned int icp = 0;  icp < conversionHandle->size(); icp++) {
    reco::ConversionRef cpRef(reco::ConversionRef(conversionHandle,icp));
    std::vector<reco::TrackRef> tracks = cpRef->tracks();
 
    if ( tracks.size() < 2 ) continue;
    reco::CaloClusterPtr aClu = cpRef->caloCluster()[0];
     
    tmp.clear();
    for( unsigned int jcp = icp;  jcp < conversionHandle->size(); jcp++) {
      reco::ConversionRef cp2Ref(reco::ConversionRef(conversionHandle,jcp));
      std::vector<reco::TrackRef> tracks2 = cp2Ref->tracks();
      if ( tracks.size() < 2 ) continue;

      
      if ( cpRef->caloCluster()[0] == cp2Ref ->caloCluster()[0] ) {
	if (debug_) std::cout << " PFConversionProducer Pushing back SC Energy " << aClu->energy() << " eta " << aClu->eta() << " phi " << aClu->phi() << " E/P " << cp2Ref->EoverP() << std::endl;     	
	tmp.push_back(cp2Ref);    
      }



    }
 
    aMap.insert(make_pair(  aClu, tmp ));   
  }

    
  
  for (  std::map<reco::CaloClusterPtr, std::vector<reco::ConversionRef> >::iterator iMap = aMap.begin(); iMap!= aMap.end(); ++iMap) {
    std::vector<reco::ConversionRef> conversions = iMap->second;
    
    
    float epMin=999;
    unsigned int iBestConv=0;
    for( unsigned int icp = 0;  icp < conversions.size(); icp++) {
      reco::ConversionRef cpRef = conversions[icp];
      std::vector<reco::TrackRef> tracks = cpRef->tracks();
      if (debug_) std::cout << " PFConversionProducer This conversion candidate has track size " << tracks.size() << " E/P " << cpRef->EoverP() << std::endl;
      if (debug_) std::cout << " PFConversionProducer SC Energy " << cpRef->caloCluster()[0]->energy() << " eta " << cpRef->caloCluster()[0]->eta() << " phi " << cpRef->caloCluster()[0]->phi() << std::endl;
      
      float px=0;
      float py=0;
      float pz=0;
      for (unsigned int i=0; i<tracks.size(); i++) {
	px+=tracks[i]->innerMomentum().x();
	py+=tracks[i]->innerMomentum().y();
	pz+=tracks[i]->innerMomentum().z();
      }
      float p=sqrt(px*px+py*py+pz*pz);
      float ep=fabs(1.-cpRef->caloCluster()[0]->energy()/p);
      if (debug_) std::cout << "icp " << icp  << " 1-E/P = " << ep << " E/P " << cpRef->caloCluster()[0]->energy()/p << std::endl; 
      if ( ep<epMin) {
	epMin=ep;
	iBestConv=icp;
      }
      
    }
    
  
    if (debug_) std::cout<< " Best conv " << iBestConv << std::endl;
    reco::ConversionRef cpRef = conversions[iBestConv];
    std::vector<reco::TrackRef> tracks = conversions[iBestConv]->tracks();
    cout<<"SQDA"<<endl;
    fillPFConversions ( cpRef, outInTrkHandle, inOutTrkHandle, outInTrajectoryHandle, inOutTrajectoryHandle, iPfTk,  pfTrackRefProd, outputConversionCollection,  pfConversionRecTrackCollection);
    
    
  }  /// loop over photons
 
  ///MICHELE  
  ///other conversion collections added in order of purity
   
  if ((OtherConvLabels_.size()==OtherOutInLabels_.size()) &&(OtherConvLabels_.size()==OtherInOutLabels_.size())){
    for (uint icol=0; icol<  OtherConvLabels_.size();icol++){
      Handle<reco::ConversionCollection> newColl;
      e.getByLabel(OtherConvLabels_[icol],newColl);
      
      //read collections of trajectories
      Handle<std::vector<Trajectory> > outInTraj;
      e.getByLabel(OtherOutInLabels_[icol],outInTraj); 
      //  
      
      Handle<std::vector<Trajectory> > inOutTraj; 
      e.getByLabel(OtherInOutLabels_[icol],inOutTraj); 

      
      // read collections of tracks
      Handle<reco::TrackCollection> outInTrk; 
      e.getByLabel(OtherOutInLabels_[icol],outInTrk); 
      
      Handle<reco::TrackCollection> inOutTrk; 
      e.getByLabel(OtherInOutLabels_[icol],inOutTrk); 
  
      ///vector of bool of the same size of new conversion collection
    
      uint AlreadySaved=  outputConversionCollection.size();
 
      cout<<"TRACCE GIA' PRESE "<<AlreadySaved<<endl;  
      
      cout<<"COLL "<<OtherConvLabels_[icol]<<" SIZ "<<newColl->size()
	  <<" "<<outInTrk->size()<<" "<<inOutTrk->size()<<endl;
      for( unsigned int icp = 0;  icp < newColl->size(); icp++) {
	reco::ConversionRef cpRef(reco::ConversionRef(newColl,icp));
	std::vector<reco::TrackRef> tracks = cpRef->tracks();
	
	if ( tracks.size() < 2 ) continue;
	
	if (isNotUsed(cpRef,outputConversionCollection)){
	  cout<<"QQ "<<endl;

	  fillPFConversions ( cpRef, outInTrk, inOutTrk, outInTraj, inOutTraj, 
			      iPfTk,  pfTrackRefProd, outputConversionCollection,  
			      pfConversionRecTrackCollection);
	}
	
	

      }//end loop on the conversion
    }
  }

  
  // put the products in the event
  if (debug_) std::cout << " PFConversionProducer putting PFConversions in the event " << outputConversionCollection.size() <<  std::endl; 
  outputConversionCollection_p->assign(outputConversionCollection.begin(),outputConversionCollection.end());
  e.put( outputConversionCollection_p, PFConversionCollection_ );

  if (debug_)  std::cout << " PFConversionProducer putting pfRecTracks in the event " << pfConversionRecTrackCollection.size() <<  std::endl; 
  pfConversionRecTrackCollection_p->assign(pfConversionRecTrackCollection.begin(),pfConversionRecTrackCollection.end());
  e.put( pfConversionRecTrackCollection_p, PFConversionRecTracks_ );

  


}



void PFConversionsProducer:: fillPFConversions ( reco::ConversionRef& cpRef, 
						 const edm::Handle<reco::TrackCollection> & outInTrkHandle,
						 const edm::Handle<reco::TrackCollection> & inOutTrkHandle, 
                                                 const edm::Handle<std::vector<Trajectory> > &   outInTrajectoryHandle, 
						 const edm::Handle<std::vector<Trajectory> > &   inOutTrajectoryHandle,
                                                 int iPfTk,
						 reco::PFRecTrackRefProd& pfTrackRefProd,
						 reco::PFConversionCollection& outputConversionCollection,
						 reco::PFRecTrackCollection& pfConversionRecTrackCollection ) {


  std::vector<Trajectory> tjOIvec= *(outInTrajectoryHandle.product());
  std::vector<Trajectory> tjIOvec= *(inOutTrajectoryHandle.product());


    std::vector<reco::TrackRef> tracks = cpRef->tracks();
    
    /////////////////////// Transform trajectories from conversion tracks in to PFRecTracks
    std::vector<reco::PFRecTrackRef> pfRecTracksRef;
    for (unsigned int i=0; i<tracks.size(); i++) {
      
      //  if (debug_) std::cout << " PFConversionProducer Track charge " <<  tracks[i]->charge() << " pt " << tracks[i]->pt() << " eta " << tracks[i]->eta() << " phi " << tracks[i]->phi() << std::endl;        
      
      
      int nFound=0;
      int iOutInSize=0;
      for( reco::TrackCollection::const_iterator  iTk =  (*outInTrkHandle).begin(); iTk !=  (*outInTrkHandle).end(); iTk++) {
	
	Trajectory traj=tjOIvec[iOutInSize];
	iOutInSize++;
	
	if ( &(*iTk) != &(*tracks[i]) ) continue; 
	nFound++;
	if (debug_) std::cout << " Found the corresponding trajectory " << std::endl;
	
	reco::PFRecTrack pftrack( double(tracks[i]->charge()), reco::PFRecTrack::KF_ELCAND, i, tracks[i] );
	
	
	bool valid = pfTransformer_->addPoints( pftrack, *tracks[i], traj);
	
	if(valid) {
	  pfRecTracksRef.push_back(reco::PFRecTrackRef( pfTrackRefProd, iPfTk ));
	  iPfTk++;	
	  pfConversionRecTrackCollection.push_back(pftrack);
	}
	
      }
      
      if (nFound==0) {
	
	
	int iInOutSize=0;
	for( reco::TrackCollection::const_iterator  iTk =  (*inOutTrkHandle).begin(); iTk !=  (*inOutTrkHandle).end(); iTk++) {
	  
	  Trajectory traj=tjIOvec[iInOutSize];
	  iInOutSize++;
	  
	  if ( &(*iTk) != &(*tracks[i]) ) continue; 
	  if (debug_) std::cout << " Found the correspnding trajectory " << std::endl;	  
	  
	  reco::PFRecTrack pftrack( double(tracks[i]->charge()), reco::PFRecTrack::KF, i, tracks[i] );
	  
	  //	Trajectory FakeTraj;
	  bool valid = pfTransformer_->addPoints( pftrack, *tracks[i], traj);
	  
	  if(valid) {
	    pfRecTracksRef.push_back(reco::PFRecTrackRef( pfTrackRefProd, iPfTk ));
	    iPfTk++;	
	    pfConversionRecTrackCollection.push_back(pftrack);
	  }
	  
	}
      }
    }
    
    
    reco::PFConversion  pfConversion(cpRef, pfRecTracksRef);
    outputConversionCollection.push_back(pfConversion);
    



}



void PFConversionsProducer::endJob()
{


  
   edm::LogInfo("PFConversionProducer") << "Analyzed " << nEvt_  << "\n";
   // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
   std::cout  << "PFConversionProducer::endJob Analyzed " << nEvt_ << " events " << "\n";
   
   return ;
}
 
bool PFConversionsProducer::isNotUsed(reco::ConversionRef newPf,reco::PFConversionCollection PFC){
  std::vector<reco::TrackRef> tracks = newPf->tracks();
  if (tracks.size()!=2) return false;
  for (uint ip=0; ip<PFC.size();ip++){
    std::vector<reco::TrackRef> oldTracks = PFC[ip].originalConversion()->tracks();
    for (uint it=0; it<2; it++){
      for (uint it2=0; it2<oldTracks.size(); it2++){
	if (SameTrack(tracks[it],oldTracks[it2])) return false;
      }
    }
    
  }
  return true;
}

bool PFConversionsProducer::SameTrack(reco::TrackRef t1, reco::TrackRef t2){
  float irec=0;
  float isha=0;
  trackingRecHit_iterator i1b= t1->recHitsBegin();
  trackingRecHit_iterator i1e= t1->recHitsEnd();
  for(;i1b!=i1e;++i1b){
    if (!((*i1b)->isValid())) continue;
    irec++;
    trackingRecHit_iterator i2b= t2->recHitsBegin();
    trackingRecHit_iterator i2e= t2->recHitsEnd();
    for(;i2b!=i2e;++i2b){
      if (!((*i2b)->isValid())) continue;
      if ((*i1b)->sharesInput(&(**i2b), TrackingRecHit::all )) isha++;
    }
  }
  cout<<"REC "<<irec<<" "<<isha<<" "<<isha/irec<<endl;
  return ((isha/irec)>0.5);
}
