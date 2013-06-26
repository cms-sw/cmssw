#include "RecoParticleFlow/PFRootEvent/interface/FWLiteJetProducer.h"
#include "RecoParticleFlow/PFRootEvent/interface/ProtoJet.h"
#include "RecoParticleFlow/PFRootEvent/interface/JetRecoTypes.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "RecoParticleFlow/PFRootEvent/interface/JetMaker.h"
#include "RecoParticleFlow/PFRootEvent/interface/JetAlgoHelper.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include <iostream>
#include <vector>

using namespace reco;
using namespace std;
using namespace JetReco;

//--------------test Mac/afs------------------

FWLiteJetProducer::FWLiteJetProducer(){ 

  mEtInputCut_=0.5;     
  mEInputCut_=0.;
  seedThreshold_=1.0;
  coneRadius_=0.5;
  coneAreaFraction_=1.0;
  maxPairSize_=2;
  maxIterations_=100;
  overlapThreshold_=0.75;
  ptMin_=10.;
  rparam_=1.;
  algoIC_=0;
  algoMC_=0;
}




//-----------------------------------------------------------
FWLiteJetProducer::~FWLiteJetProducer() {       
  delete algoIC_;
  delete algoMC_;
}
//-----------------------------------------------------------

void FWLiteJetProducer::updateParameter(){
  
  if (algoIC_) delete algoIC_;
  if (algoMC_) delete algoMC_;
  algoIC_= new CMSIterativeConeAlgorithm(seedThreshold_,coneRadius_ );
  algoMC_= new CMSMidpointAlgorithm(seedThreshold_, coneRadius_,coneAreaFraction_, 
                                    maxPairSize_, maxIterations_, overlapThreshold_, 0) ; 
  algoF_.setPtMin(ptMin_);
  algoF_.setRParam(rparam_);
  print();
}


//-----------------------------------------------------------
void FWLiteJetProducer::print() {       

  cout <<"--- FWLiteJetProducer:Print(): ---" << endl;

  cout <<"Cut: mEtInputCut " << mEtInputCut_  <<endl;
  cout <<"Cut: mEInputCut " <<  mEInputCut_<<endl; 
  cout <<"IC/MC: seedThreshold " << seedThreshold_  <<endl;
  cout <<"IC/MC: coneRadius " << coneRadius_ <<endl;
  cout <<"MC: coneAreaFraction " << coneAreaFraction_ <<endl; 
  cout <<"MC: maxPairSize " <<maxPairSize_ <<endl;
  cout <<"MC: maxIterations " << maxIterations_ <<endl;
  cout <<"MC: overlapThreshold " <<  overlapThreshold_<<endl;
  cout <<"FJ: PtMin " << ptMin_ <<endl;
  cout <<"FJ: Rparam " <<  rparam_<<endl;
  cout <<"----------------------------------" << endl;

}



//-----------------------------------------------------------
void FWLiteJetProducer::applyCuts(const reco::CandidatePtrVector& Candidates, 
				  JetReco::InputCollection* input){
  //!!!!
  //edm::OrphanHandle< reco::CandidateCollection >  CandidateHandle(&(Candidates), edm::ProductID(20001) );

  input->reserve ( Candidates.size());  
  //cout<<" Candidate " << CandidateHandle->size()<<Candidates.size() << endl;
  for (unsigned i = 0; i <Candidates.size() ; i++) {
    const reco::Candidate* constituent = Candidates[i].get();        

    if ((mEtInputCut_ <= 0 || constituent->et() > mEtInputCut_) &&
        (mEInputCut_ <= 0 || constituent->energy() > mEInputCut_)) {                    
      input->push_back (InputItem(constituent,i));                  
    }
  }
}

//-----------------------------------------------------------
void FWLiteJetProducer::makeIterativeConeJets(const InputCollection& fInput, OutputCollection* fOutput){        
  if (fInput.empty ()) {
    std::cout << "empty input for jet algorithm: bypassing..." << std::endl;
  }
  else {                                      
    algoIC_->run(fInput, & (*fOutput));
  }     
}       

//-----------------------------------------------------------
void FWLiteJetProducer::makeFastJets(const InputCollection& fInput, OutputCollection* fOutput){
  // FastJetFWLiteWrapper algo;
  if (fInput.empty ()) {
    std::cout << "empty input for jet algorithm: bypassing..." << std::endl;
  }
  else {                                      
    algoF_.run(fInput, &(*fOutput));
  } 
}

//-----------------------------------------------------------
void FWLiteJetProducer::makeMidpointJets(const InputCollection& fInput, OutputCollection* fOutput){
  //  CMSMidpointAlgorithm algo;
  if (fInput.empty ()) {
    std::cout << "empty input for jet algorithm: bypassing..." << std::endl;
  }
  else {                                      
    algoMC_->run(fInput, &(*fOutput));
  } 
}



//-----------------------------------------------------------
