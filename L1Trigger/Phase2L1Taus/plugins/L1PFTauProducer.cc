// -*- C++ -*-
//
// Package:    L1PFTauProducer
// Class:      L1PFTauProducer
// 
/**\class L1PFTauProducer L1PFTauProducer.cc L1Trigger/Phase2L1Taus/plugins/L1PFTauProducer.cc

 Description: Level 1 L1PFTaus for the Demonstrator

 Implementation:
     [Notes on implementation]
*/


#include "L1Trigger/Phase2L1Taus/interface/L1PFTauProducer.hh"

L1PFTauProducer::L1PFTauProducer(const edm::ParameterSet& cfg) :
  debug(                cfg.getUntrackedParameter<bool>("debug", false)),
  L1PFToken_(           consumes< vector<l1t::PFCandidate> >(cfg.getParameter<edm::InputTag>("L1PFObjects"))),
  L1NeutralToken_(      consumes< vector<l1t::PFCandidate> >(cfg.getParameter<edm::InputTag>("L1Neutrals")) )
{
  produces< L1PFTauCollection >( "L1PFTaus" ).setBranchAlias("L1PFTaus");
}

void L1PFTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::unique_ptr<L1PFTauCollection> newL1PFTauCollection(new L1PFTauCollection);

  edm::Handle<  l1t::PFCandidateCollection > l1PFCandidates;
  iEvent.getByToken( L1PFToken_, l1PFCandidates);
  l1t::PFCandidateCollection pfChargedHadrons;
  l1t::PFCandidateCollection pfEGammas;
  l1t::PFCandidateCollection pfNeutrals;
  l1t::PFCandidateCollection l1PFCandidates_sort;

  for(auto l1PFCand : *l1PFCandidates)
    l1PFCandidates_sort.push_back(l1PFCand);

  std::sort(l1PFCandidates_sort.begin(), l1PFCandidates_sort.end(), [](l1t::PFCandidate i,l1t::PFCandidate j){return(i.pt() > j.pt());});   

  // get all PF Charged Hadrons
  for(auto l1PFCand : l1PFCandidates_sort){
    // this saves some low pt 3 prong taus, but should be optimized with respect to 1 prong pi0
    if(l1PFCand.id() == l1t::PFCandidate::ChargedHadron || (l1PFCand.pt()<5 && l1PFCand.id() == l1t::PFCandidate::Electron)){
      pfChargedHadrons.push_back(l1PFCand);
      continue;
    }
    // get all PF EM candidates
    if((l1PFCand.id() == l1t::PFCandidate::Electron) || (l1PFCand.id() == l1t::PFCandidate::Photon)){
      pfEGammas.push_back(l1PFCand);
    }
  }
  
  // create all Tau Candidates based on detector region
  // 12 in phi x 10 taus 
  // nTaus in phi, nTaus in Eta
  tauCandidates.clear();
  createTaus(tauCandidates);
  
  // loop through PF Candidates and add PF Cands to each Tau
  for(auto pfChargedHadron : pfChargedHadrons){
    for(auto &tauCandidate : tauCandidates){
      if(tauCandidate.addPFChargedHadron(pfChargedHadron)){
	break;
      }
    }
  }
  
  for(auto pfEGamma : pfEGammas){
    for(auto &tauCandidate : tauCandidates){
      if(tauCandidate.addEG(pfEGamma))
	continue;
    }
  }
  
     // Run Tau Algo over all Taus. 
  for(auto &tauCandidate : tauCandidates){
    tauCandidate.process();
  }
  
  // Sort by PT
  unsigned int nCands = 12;
  tau_cand_sort(tauCandidates, newL1PFTauCollection, nCands);

  //for(int i = 0; i < newL1PFTauCollection->size(); i++)
  //{
  //std::cout<<"Tau "<<i<<newL1PFTauCollection->at(i)<<std::endl;
  //}

  iEvent.put( std::move(newL1PFTauCollection) , "L1PFTaus" );
}
  
// create taus based on grid structure
void L1PFTauProducer::createTaus(tauMapperCollection &inputCollection){
  inputCollection.clear();
  float left_edge_center_eta = (-1)*tracker_eta + tau_size_eta/2 ;
 
  for(float iTau_eta = left_edge_center_eta; iTau_eta < tracker_eta; iTau_eta = iTau_eta + tau_size_eta ){
    for(float iTau_phi = -3.14159; iTau_phi < 3.14159; iTau_phi = iTau_phi + tau_size_phi ){
      TauMapper tempTau; 
      tempTau.l1PFTau.setHWEta(iTau_eta);
      tempTau.l1PFTau.setHWPhi(iTau_phi);
      tempTau.ClearSeedHadron();
      inputCollection.push_back(tempTau);
    }
  }

}


void L1PFTauProducer::tau_cand_sort(tauMapperCollection tauCandidates, std::unique_ptr<L1PFTauCollection> &newL1PFTauCollection, unsigned int nCands){
  std::sort(tauCandidates.begin(), tauCandidates.end(), [](TauMapper i,TauMapper j){return(i.l1PFTau.pt() > j.l1PFTau.pt());});   
  
  for(unsigned int i = 0; i < nCands && i < tauCandidates.size(); i++){
    newL1PFTauCollection->push_back(tauCandidates.at(i).l1PFTau);
  }

}


/////////////
// DESTRUCTOR
L1PFTauProducer::~L1PFTauProducer()
{
}  


//////////
// END JOB
void L1PFTauProducer::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}

////////////
// BEGIN JOB
void L1PFTauProducer::beginRun(const edm::Run& run, const edm::EventSetup& iSetup )
{
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1PFTauProducer);

//  LocalWords:  PFChargedCandidates
