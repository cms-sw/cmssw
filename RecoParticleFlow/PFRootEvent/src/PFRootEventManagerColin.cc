#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManagerColin.h"
#include "RecoParticleFlow/PFRootEvent/interface/IO.h"

#include "TTree.h"
#include "TFile.h"

#include <iostream>

using namespace std;

PFRootEventManagerColin::PFRootEventManagerColin(const char* file)
  : PFRootEventManager(file) {
  
  tauEvent_ = 0;
  neutralEvent_ = 0;
  outTreeMy_ = 0;   
  
  //   readOptions(file, false, false);
  
  // book histos here
  //   neutralEvent_ = new NeutralEvent();  


  //   tauEvent_ = new TauEvent();  
  //   outTree_ = new TTree("Tau","");
  //   outTree_->Branch("event","TauEvent", &tauEvent_,32000,2);

  readSpecificOptions(file);

}

PFRootEventManagerColin::~PFRootEventManagerColin() {

  //   delete event_;
  //   delete outTree_;
}


void PFRootEventManagerColin::readSpecificOptions(const char* file) {



  cout<<"calling PFRootEventManagerColin::readSpecificOptions"<<endl; 
  //   PFRootEventManager::readOptions(file, refresh, reconnect);

  
  options_->GetOpt("colin", "mode", mode_);

  if(outTreeMy_) delete outTreeMy_;
  
  outFile_->cd();
  switch(mode_) {
  case Neutral:
    cout<<"colin: Neutral mode"<<endl;
    neutralEvent_ = new NeutralEvent();  
    outTreeMy_ = new TTree("Neutral","");
    outTreeMy_->Branch("event","NeutralEvent", &neutralEvent_,32000,2);
    gDirectory->ls();
    break;
  case HIGH_E_TAUS:
    cout<<"colin: highETaus mode"<<endl;
    tauEvent_ = new TauEvent();  
    outTreeMy_ = new TTree("Tau","");
    outTreeMy_->Branch("event","TauEvent", &tauEvent_,32000,2);
    gDirectory->ls();
    break;
  default:
    cerr<<"colin: undefined mode"<<endl;
    exit(1);
  }
}





bool PFRootEventManagerColin::processEntry(int entry) {

  tauEvent_->reset();

  if( ! PFRootEventManager::processEntry(entry) ) {
    // cerr<<"event was not accepted"<<endl;
    // print();
    tauEvent_->rCode = 10;
    return false; // event not accepted
  }

  bool rvalue = false;
  switch(mode_) {
  case Neutral:
    // cout<<"colin: process Neutral"<<endl;
    rvalue = processNeutral();
    break;
  case HIGH_E_TAUS:
    // cout<<"colin: process highETaus"<<endl;
    rvalue = processHIGH_E_TAUS();
    break;
  default:
    cerr<<"colin: undefined mode"<<endl;
    assert(0);
  }
  outTreeMy_->Fill();


  outTreeMy_->Fill();


  outTreeMy_->Fill();


  return rvalue;
}




bool PFRootEventManagerColin::processNeutral() {
  //   else {
  //     cerr<<"event accepted"<<endl;
  //   }
  
  //   if( ! ( (*clustersECAL_).size() <= 1 && 
  //      (*clustersHCAL_).size() <= 1 ) ) {
  //     cerr<<"wrong number of ECAL or HCAL clusters :"
  //    <<(*clustersECAL_).size()<<","<<(*clustersHCAL_).size()<<endl;
  //     return false; 
  //   }
  // 1 HCAL cluster
  
  neutralEvent_->reset();
  
  // particle

  const HepMC::GenEvent* myGenEvent = MCTruth_.GetEvent();
  if(!myGenEvent) {
    assert(0);
  }

  if( myGenEvent->particles_size() != 1 ) {
    cerr<<"wrong number of particles:"
        <<myGenEvent->particles_size()<<endl;
    return 0;
  }

  // take first particle  
  const HepMC::GenParticle* particle = *(myGenEvent->particles_begin() );
  
  // and check that it's a K0L 
  if( particle->pdg_id() != 130 ) {
    cerr<<"not a K0L : "<<particle->pdg_id()<<endl;
    return false;
  }

  neutralEvent_->eNeutral = particle->momentum().e();

  double eta =  particle->momentum().eta();
  double phi =  particle->momentum().phi();
  neutralEvent_->etaNeutral = eta;
  

  neutralEvent_->nECAL = (*clustersECAL_).size();

  // look for the closest ECAL cluster from the particle.

  double minDist2 = 9999999;
  // int iClosest = -1;
  for( unsigned i=0; i<(*clustersECAL_).size(); ++i) {
    double deta = (*clustersECAL_)[i].position().Eta() - eta;
    double dphi = (*clustersECAL_)[i].position().Phi() - phi;
    double dist2 = deta*deta + dphi*dphi;
    if(dist2 < minDist2) {
      minDist2 = dist2;
      neutralEvent_->eECAL = (*clustersECAL_)[i].energy();
    }
  }


  neutralEvent_->nHCAL = (*clustersHCAL_).size();
  if( (*clustersHCAL_).size() == 1 ) {
    neutralEvent_->eHCAL = (*clustersHCAL_)[0].energy();
  }

  
  outTreeMy_->Fill();
  
  if(  neutralEvent_->nECAL<1 && neutralEvent_->eNeutral<1 )
    return true;
  else return false;
}


bool PFRootEventManagerColin::processHIGH_E_TAUS() {


  // true info 
  // 1 charged hadron, 2 photons
  // save charged part mom, save sum E photons
  

  int iHadron  = -1;
  int iPi0 = -1;
  unsigned nStableChargedHadrons=0;
  unsigned nPi0=0;
  for(unsigned i=0; i<trueParticles_.size(); i++) {
    
    const reco::PFSimParticle& part = trueParticles_[i];
    
    int pdgCode = part.pdgCode();
    double charge = part.charge();

    if( std::abs(pdgCode) > 100 &&
        charge !=0 && 
        part.daughterIds().empty() ) {
      nStableChargedHadrons++;
      iHadron = i;
    }
    else if( std::abs(pdgCode)==111) {
      nPi0++;
      iPi0 = i; 
    } 

    // cout<<i<<" "<<part<<endl;
  }


  // one has to select 1 charged and 2 photons 
  // to use this function.

  // even after filtering events with one stable charged particle,
  // this particle can be a lepton (eg leptonic pion decay)
  if( nStableChargedHadrons==0 ) {
    tauEvent_->rCode = 4; 
    return false;
  }
  assert( nStableChargedHadrons==1 );


  
  double pHadron = trueParticles_[iHadron].extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach ).momentum().P(); 
  tauEvent_->pHadron = pHadron;

  //  tauEvent_->eEcalHadron = trueParticles_[iHadron].ecalEnergy();
  
  if(nPi0 == 1) {
    math::XYZTLorentzVector pi0mom =  trueParticles_[iPi0].extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach ).momentum();
    tauEvent_->eNeutral = pi0mom.E();
    tauEvent_->etaNeutral = pi0mom.Eta();
  }
  else {
    tauEvent_->eNeutral = 0;
  }

  //   if( tauEvent_->eNeutral > 0.1* tauEvent_->pHadron ) {
  //     print();
  //   }


  // check that there is 
  // only one track
  // 0 or 1 ecal cluster
  // 0 or 1 hcal cluster

  if( recTracks_.size() != 1 ) {
    //     cout<<"more than 1 track"<<endl;
    tauEvent_->rCode = 1;
    return false;
  }
  if( clustersECAL_->size() > 1 ) {
    //     cout<<"more than 1 ecal cluster"<<endl;
    tauEvent_->rCode = 2;
    // return false;
  }
  if( clustersHCAL_->size() > 1 ) {
    //     cout<<"more than 1 hcal cluster"<<endl;
    tauEvent_->rCode = 3;
    return false;
  }
  // save track mom + neutral info.

  tauEvent_->pTrack = recTracks_[0].extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach ).momentum().P();
  tauEvent_->ptTrack = recTracks_[0].extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach ).momentum().Pt();
  tauEvent_->etaTrack = recTracks_[0].extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach ).momentum().Eta();

  // access blocks

  // take the track block

  // look for the closest associated ecal and hcal clusters

  // fill the tree

    


  for(unsigned i=0; i<pfBlocks_->size(); i++) {
    const reco::PFBlock& block = (*pfBlocks_)[i];
    
    const edm::OwnVector< reco::PFBlockElement >& 
      elements = block.elements();
    
    // look for the track
    int iTrack = -1;
    unsigned nTracks = 0;
    for(unsigned ie=0; ie<elements.size(); ie++) {
      if(elements[ie].type() == reco::PFBlockElement::TRACK  ) {
        iTrack = ie;
        nTracks++;
      }
    }
    
    if(nTracks!=1) continue; // no track, or too many tracks in the block
    
    std::multimap<double, unsigned> sortedElems;
    block.associatedElements( iTrack, 
                              block.linkData(),
                              sortedElems );
    
    tauEvent_->nECAL=0;
    tauEvent_->nHCAL=0;
    
    typedef std::multimap<double, unsigned>::iterator IE;
    for(IE ie = sortedElems.begin(); ie != sortedElems.end(); ++ie ) {
      
      
      double chi2 = ie->first;
      unsigned index = ie->second;
      
      reco::PFBlockElement::Type type = elements[index].type();
      
      reco::PFClusterRef clusterRef = elements[index].clusterRef();


      

      if( type == reco::PFBlockElement::ECAL ) {
        if(!tauEvent_->nECAL ) { // closest ecal
          assert( !clusterRef.isNull() );
          tauEvent_->eECAL = clusterRef->energy();
          tauEvent_->etaECAL = clusterRef->position().Eta();
          tauEvent_->chi2ECAL = chi2;
          tauEvent_->nECAL++;
        }
      }


      else if( type == reco::PFBlockElement::HCAL ) {
        if(!tauEvent_->nHCAL ) { // closest hcal
          assert( !clusterRef.isNull() );
          tauEvent_->eHCAL = clusterRef->energy();
          tauEvent_->etaHCAL = clusterRef->position().Eta();
          tauEvent_->nHCAL++;
        }
      } 
    } // eles associated to the track
  } // blocks




  return false;    
}




void PFRootEventManagerColin::write() {
  // write histos here
  outFile_->cd();
  outTreeMy_->Write();

  PFRootEventManager::write();
}

