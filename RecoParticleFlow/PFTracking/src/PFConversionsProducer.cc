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
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
//
#include "TrackingTools/PatternTools/interface/Trajectory.h"
//
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <vector>


PFConversionsProducer::PFConversionsProducer( const edm::ParameterSet& pset ) : pfTransformer_(0) {
  // input collection names
  conversionCollectionProducer_ = pset.getParameter<std::string>("conversionProducer");
  conversionCollection_ = pset.getParameter<std::string>("conversionCollection");
  debug_ = pset.getParameter<bool>("debug");


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
 

  nEvt_++;  
  
  ///// Get the externally reconstructed  conversions
  Handle<reco::PhotonCollection> conversionHandle; 
  e.getByLabel(conversionCollectionProducer_, conversionCollection_ , conversionHandle);
  if (! conversionHandle.isValid()) {
    edm::LogError("PFConversionsProducer") << "Error! Can't get the product "<<conversionCollection_.c_str();
    return;
  }
  const reco::PhotonCollection conversionCollection = *(conversionHandle.product());
  //  std::cout  << "PFConversionsProducer  input conversions collection size " << conversionCollection.size() << "\n";


   //read collections of trajectories
  Handle<std::vector<Trajectory> > outInTrajectoryCollection;
  e.getByLabel("ckfOutInTracksFromConversions",outInTrajectoryCollection); 
  std::vector<Trajectory> tjOIvec= *(outInTrajectoryCollection.product());

  Handle<std::vector<Trajectory> > inOutTrajectoryCollection; 
  e.getByLabel("ckfInOutTracksFromConversions",inOutTrajectoryCollection); 
  std::vector<Trajectory> tjIOvec= *(inOutTrajectoryCollection.product());

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
  for( reco::PhotonCollection::const_iterator  iPho = conversionCollection.begin(); iPho != conversionCollection.end(); iPho++) {
    if ( !(*iPho).isConverted() ) continue;
    std::vector<reco::ConversionRef> conversions = (*iPho).conversions();



    float epMin=999;
    unsigned int iBestConv=0;
    for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {
      
      std::vector<reco::TrackRef> tracks = conversions[iConv]->tracks();
      if (debug_) std::cout << " PFConversionProducer This conversion candidate has track size " << tracks.size() << std::endl;
      if (debug_) std::cout << " PFConversionProducer SC Energy " << (*iPho).superCluster()->energy() << " eta " << (*iPho).superCluster()->eta() << " phi " << (*iPho).superCluster()->phi() << std::endl;
      
      float px=0;
      float py=0;
      float pz=0;
      for (unsigned int i=0; i<tracks.size(); i++) {
	px+=tracks[i]->innerMomentum().x();
	py+=tracks[i]->innerMomentum().y();
	pz+=tracks[i]->innerMomentum().z();
      }
      float p=sqrt(px*px+py*py+pz*pz);
      float ep=fabs(1.-(*iPho).superCluster()->energy()/p);
       if (debug_) std::cout << "iConv " << iConv  << " 1-E/P = " << ep << std::endl; 
      if ( ep<epMin) {
	epMin=ep;
	iBestConv=iConv;
      }
      
    }

    if (debug_) std::cout<< " Best conv " << iBestConv << std::endl;
    std::vector<reco::TrackRef> tracks = conversions[iBestConv]->tracks();
    if ( tracks.size() < 2 ) continue;

    std::vector<reco::PFRecTrackRef> pfRecTracksRef;
    
    for (unsigned int i=0; i<tracks.size(); i++) {
      
      if (debug_) std::cout << " PFConversionProducer Track charge " <<  tracks[i]->charge() << " pt " << tracks[i]->pt() << " eta " << tracks[i]->eta() << " phi " << tracks[i]->phi() << std::endl;        
      
      
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
    

    reco::PFConversion  pfConversion(conversions[iBestConv], pfRecTracksRef);
    outputConversionCollection.push_back(pfConversion);

    //   } // loop over conversions 

  }  /// loop over photons



  // put the products in the event
  if (debug_) std::cout << " PFConversionProducer putting PFConversions in the event " << outputConversionCollection.size() <<  std::endl; 
  outputConversionCollection_p->assign(outputConversionCollection.begin(),outputConversionCollection.end());
  e.put( outputConversionCollection_p, PFConversionCollection_ );

  if (debug_)  std::cout << " PFConversionProducer putting pfRecTracks in the event " << pfConversionRecTrackCollection.size() <<  std::endl; 
  pfConversionRecTrackCollection_p->assign(pfConversionRecTrackCollection.begin(),pfConversionRecTrackCollection.end());
  e.put( pfConversionRecTrackCollection_p, PFConversionRecTracks_ );

  


}




void PFConversionsProducer::endJob()
{


  
   edm::LogInfo("PFConversionProducer") << "Analyzed " << nEvt_  << "\n";
   // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
   std::cout  << "PFConversionProducer::endJob Analyzed " << nEvt_ << " events " << "\n";
   
   return ;
}
 


