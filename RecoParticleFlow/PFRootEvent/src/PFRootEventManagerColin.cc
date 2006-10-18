#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManagerColin.h"
#include "TTree.h"
#include "TFile.h"

#include <iostream>

using namespace std;

PFRootEventManagerColin::PFRootEventManagerColin(const char* file)
  : PFRootEventManager(file) {
  
  // book histos here
  event_ = new EventColin();
  
  outTree_ = new TTree("Eff","");
  outTree_->Branch("event","EventColin", &event_,32000,2);
}

PFRootEventManagerColin::~PFRootEventManagerColin() {
  // delete event_;
}


bool PFRootEventManagerColin::processEntry(int entry) {
  PFRootEventManager::processEntry(entry);

  // fill histos here

  event_->reset();

//   EventColin::Particle ptc;
//   ptc.eta = 1;
//   ptc.phi = 2;
//   ptc.e = 10;

  // if(trueParticles_.size() != 1 ) return false;

  for(unsigned i=0; i<trueParticles_.size(); i++) {
    const reco::PFParticle& ptc = trueParticles_[i];
      
    // cout<<ptc<<endl;

    const reco::PFTrajectoryPoint& tpatecal 
      = ptc.trajectoryPoint(1);

    // cout<<tpatecal<<endl;

    EventColin::Particle outptc;
    outptc.eta = tpatecal.positionXYZ().Eta();
    outptc.phi = tpatecal.positionXYZ().Phi();    
    outptc.e = tpatecal.momentum().E();

    // cout<<"energy "<<outptc.e<<endl;
    event_->addParticle(outptc);
  }

  for(unsigned i=0; i<clustersECAL_->size(); i++) {
    EventColin::Cluster cluster;
    cluster.eta = (*clustersECAL_)[i].positionXYZ().Eta();
    cluster.phi = (*clustersECAL_)[i].positionXYZ().Phi();
    cluster.e = (*clustersECAL_)[i].energy();
    cluster.layer = (*clustersECAL_)[i].layer();
    cluster.type = (*clustersECAL_)[i].type();
    event_->addCluster(cluster);
  }   

  for(unsigned i=0; i<clustersIslandBarrel_.size(); i++) {
    EventColin::Cluster cluster;
    cluster.eta =  clustersIslandBarrel_[i].eta();
    cluster.phi = clustersIslandBarrel_[i].phi();
    cluster.e = clustersIslandBarrel_[i].energy();
    cluster.layer = -1;
    cluster.type = 4;
    event_->addClusterIsland(cluster);
  }

  outTree_->Fill();
  return false;
}

void PFRootEventManagerColin::write() {
  // write histos here
  outFile_->cd();
  outTree_->Write();
}

