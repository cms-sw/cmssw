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
   
 

    parameters_ = pset;
   

}



PhotonAnalyzer::~PhotonAnalyzer() {




}


void PhotonAnalyzer::beginJob( const edm::EventSetup& setup)
{


  nEvt_=0;
  nEntry_=0;
  
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



  double eMin = parameters_.getParameter<double>("eMin");
  double eMax = parameters_.getParameter<double>("eMax");
  int eBin = parameters_.getParameter<int>("eBin");

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int etBin = parameters_.getParameter<int>("etBin");

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int etaBin = parameters_.getParameter<int>("etaBin");

  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");
  

  double r9Min = parameters_.getParameter<double>("r9Min"); 
  double r9Max = parameters_.getParameter<double>("r9Max"); 
  int r9Bin = parameters_.getParameter<int>("r9Bin");

  if (dbe_) {  
    //// All MC photons
    // SC from reco photons

    dbe_->setCurrentFolder("DQMOffline/Egamma/PhotonAnalyzer");

    //// Reconstructed Converted photons
    std::string histname = "scE";
    h_scE_.push_back(dbe_->book1D(histname+"all","SC Energy: all Ecal  ",eBin,eMin, eMax));
    h_scE_.push_back(dbe_->book1D(histname+"barrel","SC Energy: Barrel ",eBin,eMin, eMax));
    h_scE_.push_back(dbe_->book1D(histname+"endcap","SC Energy: Endcap ",eBin,eMin, eMax));

    histname = "scEt";
    h_scEt_.push_back( dbe_->book1D(histname+"all","SC Et: all Ecal ",etBin,etMin, etMax) );
    h_scEt_.push_back( dbe_->book1D(histname+"barrel","SC Et: Barrel",etBin,etMin, etMax) );
    h_scEt_.push_back( dbe_->book1D(histname+"endcap","SC Et: Endcap",etBin,etMin, etMax) );

    histname = "r9";
    h_r9_.push_back( dbe_->book1D(histname+"all",   "r9: all Ecal",r9Bin,r9Min, r9Max) );
    h_r9_.push_back( dbe_->book1D(histname+"barrel","r9: all Ecal",r9Bin,r9Min, r9Max) );
    h_r9_.push_back( dbe_->book1D(histname+"endcap","r9: all Ecal",r9Bin,r9Min, r9Max) );

    h_scEta_ = dbe_->book1D("scEta","SC Eta ",etaBin,etaMin, etaMax);
    h_scPhi_ = dbe_->book1D("scPhi","SC Phi ",phiBin,phiMin,phiMax);
    h_scEtaPhi_ = dbe_->book2D("scEtaPhi","SC Phi vs Eta ",etaBin, etaMin, etaMax,phiBin,phiMin,phiMax);
    //
    histname = "phoE";
    h_phoE_.push_back(dbe_->book1D(histname+"all","Photon Energy: all ecal ", eBin,eMin, eMax));
    h_phoE_.push_back(dbe_->book1D(histname+"barrel","Photon Energy: barrel ",eBin,eMin, eMax));
    h_phoE_.push_back(dbe_->book1D(histname+"endcap","Photon Energy: endcap ",eBin,eMin, eMax));
    histname = "phoEt";
    h_phoEt_.push_back(dbe_->book1D(histname+"all","Photon Transverse Energy: all ecal ", etBin,etMin, etMax));
    h_phoEt_.push_back(dbe_->book1D(histname+"barrel","Photon Transverse Energy: barrel ",etBin,etMin, etMax));
    h_phoEt_.push_back(dbe_->book1D(histname+"endcap","Photon Transverse Energy: endcap ",etBin,etMin, etMax));

    h_phoEta_ = dbe_->book1D("phoEta","Photon Eta ",etaBin,etaMin, etaMax);
    h_phoPhi_ = dbe_->book1D("phoPhi","Photon  Phi ",phiBin,phiMin,phiMax);
    
     // SC from reco photons
    /*
    dbe_->tag( h_scE_->getFullname(),10);
    dbe_->tag( h_scEt_->getFullname(),11);
    dbe_->tag( h_scEta_->getFullname(),12);
    dbe_->tag( h_scPhi_->getFullname(),13);
    
    dbe_->tag( h_phoE_->getFullname(),14);
    dbe_->tag( h_phoEta_->getFullname(),15);
    dbe_->tag( h_phoPhi_->getFullname(),16);
    
    */

  }

  
  return ;
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
            
      std::cout  << " PhotonAnalyzer Reco SC energy " << (*iPho).superCluster()->energy() <<  "\n";

      nEntry_++;
      float phiClu=(*iPho).superCluster()->phi();
      float etaClu=(*iPho).superCluster()->eta();
      float etaPho=(*iPho).phi();
      float phiPho=(*iPho).eta();

      bool  scIsInBarrel=false;
      bool  scIsInEndcap=false;
      bool  phoIsInBarrel=false;
      bool  phoIsInEndcap=false;
   
      if ( fabs(etaClu) <  1.479 ) 
	scIsInBarrel=true;
      else
	scIsInEndcap=true;

      if ( fabs(etaPho) <  1.479 ) 
	phoIsInBarrel=true;
      else
	phoIsInEndcap=true;



      h_scEta_->Fill( (*iPho).superCluster()->position().eta() );
      h_scPhi_->Fill( (*iPho).superCluster()->position().phi() );
      h_scEtaPhi_->Fill( (*iPho).superCluster()->position().eta(),(*iPho).superCluster()->position().phi() );

   
      h_scE_[0]->Fill( (*iPho).superCluster()->energy() );
      h_scEt_[0]->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->position().eta()) );
      h_r9_[0]->Fill( (*iPho).r9() );
      
      if ( scIsInBarrel ) {
	h_scE_[1]->Fill( (*iPho).superCluster()->energy() );
	h_scEt_[1]->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->position().eta()) );
      }
      if ( scIsInEndcap ) {
	h_scE_[2]->Fill( (*iPho).superCluster()->energy() );
	h_scEt_[2]->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->position().eta()) );
      }


      h_phoEta_->Fill( (*iPho).eta() );
      h_phoPhi_->Fill( (*iPho).phi() );

      h_phoE_[0]->Fill( (*iPho).energy() );
      h_phoEt_[0]->Fill( (*iPho).energy()/ cosh( (*iPho).eta()) );
      h_r9_[0]->Fill( (*iPho).r9());
      
      if ( phoIsInBarrel ) {
	h_phoE_[1]->Fill( (*iPho).energy() );
	h_phoEt_[1]->Fill( (*iPho).energy()/ cosh( (*iPho).eta()) );
	h_r9_[1]->Fill( (*iPho).r9());
      }

      if ( phoIsInEndcap ) {
	h_phoE_[2]->Fill( (*iPho).energy() );
	h_phoEt_[2]->Fill( (*iPho).energy()/ cosh( (*iPho).eta()) );
	h_r9_[2]->Fill( (*iPho).r9());
      }


      
    }/// End loop over Reco  particles
    

  

}




void PhotonAnalyzer::endJob()
{



  dbe_->showDirStructure();
  bool outputMEsInRootFile = parameters_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
  
  edm::LogInfo("PhotonAnalyzer") << "Analyzed " << nEvt_  << "\n";
   // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
  std::cout  << "PhotonAnalyzer::endJob Analyzed " << nEvt_ << " events " << "\n";
  std::cout << " Total number of photons " << nEntry_ << std::endl;
   
   return ;
}
 


