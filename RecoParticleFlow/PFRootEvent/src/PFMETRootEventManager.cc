#include "RecoParticleFlow/PFRootEvent/interface/PFMETRootEventManager.h"
#include "RecoParticleFlow/Benchmark/interface/PFMETBenchmark.h"

#include <TFile.h>

#include <iostream>

using namespace std;


PFMETRootEventManager::PFMETRootEventManager(const char* file)
  : PFRootEventManager(file) {
  
  // book histos here

  // you can add your own options to the option file,
  // following the model of PFRootEventManager::readOptions
}


PFMETRootEventManager::~PFMETRootEventManager() {
  // delete histos here
}

bool PFMETRootEventManager::processEntry(int entry) {
  if( ! PFRootEventManager::processEntry(entry) )
    return false; // event not accepted
  // fill histos here
  return false;
}

float PFMETRootEventManager::DeltaMET(int entry) 
{
  if( ! PFRootEventManager::processEntry(entry) )
    return false; // event not accepted
  benchmark_.calculateQuantities( pfMetsCMSSW_, genParticlesCMSSW_, caloMetsCMSSW_ );
  return benchmark_.getDeltaPFMET();
}

float PFMETRootEventManager::DeltaPhi(int entry) 
{
  if( ! PFRootEventManager::processEntry(entry) )
    return false; // event not accepted
  benchmark_.calculateQuantities( pfMetsCMSSW_, genParticlesCMSSW_, caloMetsCMSSW_ );
  return benchmark_.getDeltaPFPhi();
}

void PFMETRootEventManager::write() {
  // write histos here
  outFile_->cd();
}

