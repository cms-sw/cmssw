#include <iostream>
//
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
//
#include "DQMOffline/EGamma/interface/PhotonAnalyzer.h"
//
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
//
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
// 

 

using namespace std;

 
PhotonAnalyzer::PhotonAnalyzer( const edm::ParameterSet& pset )
  {

    fName_     = pset.getUntrackedParameter<std::string>("Name");
    verbosity_ = pset.getUntrackedParameter<int>("Verbosity");

    
    photonCollectionProducer_ = pset.getParameter<std::string>("phoProducer");
    photonCollection_ = pset.getParameter<std::string>("photonCollection");
    parameters = pset;
   

}



PhotonAnalyzer::~PhotonAnalyzer() {




}


void PhotonAnalyzer::beginJob( const edm::EventSetup& setup)
{


  nEvt_=0;
  
    dbe_ = 0;
    dbe_ = edm::Service<DQMStore>().operator->();
  


 if (dbe_) {
    if (verbosity_ > 0 ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }
  if (dbe_) {
    if (verbosity_ > 0 ) dbe_->showDirStructure();
  }



  if (dbe_) {  
    //// All MC photons
    // SC from reco photons
    h_scE_=0;
    h_scEt_=0;
    h_scEta_=0;
    h_scPhi_=0;
    //
    h_phoE_=0;
    h_phoEta_=0;
    h_phoPhi_=0;
    //

    dbe_->setCurrentFolder("DQMOffline/Egamma/PhotonAnalyzer");

    //// Reconstructed Converted photons
    h_scE_ = dbe_->book1D("scE","SC Energy ",100,0., 200.);
    h_scEt_ = dbe_->book1D("scEt","SC Et ",100,0., 200.);
    h_scEta_ = dbe_->book1D("scEta","SC Eta ",40,-3., 3.);
    h_scPhi_ = dbe_->book1D("scPhi","SC Phi ",40, -3.14, 3.14);
    //
    h_phoE_ = dbe_->book1D("phoE","Photon Energy ",100,0., 200.);
    h_phoEta_ = dbe_->book1D("phoEta","Photon Eta ",40,-3., 3.);
    h_phoPhi_ = dbe_->book1D("phoPhi","Photon  Phi ",40,  -3.14, 3.14);
    
     // SC from reco photons
     dbe_->tag( h_scE_->getFullname(),10);
     dbe_->tag( h_scEt_->getFullname(),11);
     dbe_->tag( h_scEta_->getFullname(),12);
     dbe_->tag( h_scPhi_->getFullname(),13);
     //
     dbe_->tag( h_phoE_->getFullname(),14);
     dbe_->tag( h_phoEta_->getFullname(),15);
     dbe_->tag( h_phoPhi_->getFullname(),16);
     //

  }

  
  return ;
}


float PhotonAnalyzer::etaTransformation(  float EtaParticle , float Zvertex)  {

//---Definitions
	const float PI    = 3.1415927;
	const float TWOPI = 2.0*PI;

//---Definitions for ECAL
	const float R_ECAL           = 136.5;
	const float Z_Endcap         = 328.0;
	const float etaBarrelEndcap  = 1.479; 
   
//---ETA correction

	float Theta = 0.0  ; 
        float ZEcal = R_ECAL*sinh(EtaParticle)+Zvertex;

	if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
	if(Theta<0.0) Theta = Theta+PI ;
	float ETA = - log(tan(0.5*Theta));
         
	if( fabs(ETA) > etaBarrelEndcap )
	  {
	   float Zend = Z_Endcap ;
	   if(EtaParticle<0.0 )  Zend = -Zend ;
	   float Zlen = Zend - Zvertex ;
	   float RR = Zlen/sinh(EtaParticle); 
	   Theta = atan(RR/Zend);
	   if(Theta<0.0) Theta = Theta+PI ;
 	   ETA = - log(tan(0.5*Theta));		      
	  } 
//---Return the result
        return ETA;
//---end
}






void PhotonAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& )
{
  
  
  using namespace edm;
  const float etaPhiDistance=0.01;
  // Fiducial region
  const float TRK_BARL =0.9;
  const float BARL = 1.4442; // DAQ TDR p.290
  const float END_LO = 1.566;
  const float END_HI = 2.5;
 // Electron mass
  const Float_t mElec= 0.000511;


  nEvt_++;  
  LogInfo("PhotonAnalyzer") << "PhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";
  //  LogDebug("PhotonAnalyzer") << "PhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "PhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
 
  
  ///// Get the recontructed  conversions
  Handle<reco::PhotonCollection> photonHandle; 
  e.getByLabel(photonCollectionProducer_, photonCollection_ , photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
  std::cout  << "PhotonAnalyzer  Photons with conversions collection size " << photonCollection.size() << "\n";



    //std::cout   << " PhotonAnalyzer  Starting loop over photon candidates " << "\n";
    for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {
            
      //      std::cout  << " PhotonAnalyzer Reco SC energy " << (*iPho).superCluster()->energy() <<  "\n";

      float phiClu=(*iPho).superCluster()->phi();
      float etaClu=(*iPho).superCluster()->eta();

      
      h_scE_->Fill( (*iPho).superCluster()->energy() );
      h_scEt_->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->position().eta()) );
      h_scEta_->Fill( (*iPho).superCluster()->position().eta() );
      h_scPhi_->Fill( (*iPho).superCluster()->position().phi() );


      h_phoE_->Fill( (*iPho).energy() );
      h_phoEta_->Fill( (*iPho).eta() );
      h_phoPhi_->Fill( (*iPho).phi() );
      
      
    }/// End loop over Reco  particles
    

  

}




void PhotonAnalyzer::endJob()
{



  dbe_->showDirStructure();
  bool outputMEsInRootFile = parameters.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
  
  edm::LogInfo("PhotonAnalyzer") << "Analyzed " << nEvt_  << "\n";
   // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
  std::cout  << "PhotonAnalyzer::endJob Analyzed " << nEvt_ << " events " << "\n";
   
   return ;
}
 


