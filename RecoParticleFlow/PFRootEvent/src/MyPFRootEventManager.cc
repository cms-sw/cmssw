#include "RecoParticleFlow/PFRootEvent/interface/MyPFRootEventManager.h"

#include <TFile.h>

#include <iostream>

using namespace std;


MyPFRootEventManager::MyPFRootEventManager(const char* file)
  : PFRootEventManager(file) {
  
  // book histos here

  // you can add your own options to the option file,
  // following the model of PFRootEventManager::readOptions
}


MyPFRootEventManager::~MyPFRootEventManager() {
  // delete histos here
}


bool MyPFRootEventManager::processEntry(int entry) {
  if( ! PFRootEventManager::processEntry(entry) )
    return false; // event not accepted

  // fill histos here

  cout<<"true particles: "<<endl;

  for(unsigned i=0; i<trueParticles_.size(); i++) {
    const reco::PFSimParticle& ptc = trueParticles_[i];
      
    cout<<ptc<<endl;
  }

  
  //   cout<<"particle flow blocks : "<<endl;

  //   for(unsigned i=0; i<allPFBs_.size(); i++) {
  //     const PFBlock& block = allPFBs_[i];
      
  //     cout<<block<<endl;
  //   }

  // clusters can be accessed here, or through the pflow blocks
  for(unsigned i=0; i<clustersECAL_->size(); i++) {
    ;
  }   



  return false;
}




void MyPFRootEventManager::write() {
  // write histos here
  outFile_->cd();
}

