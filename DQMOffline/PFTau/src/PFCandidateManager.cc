#include "DQMOffline/PFTau/interface/PFCandidateManager.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// #include "DQMServices/Core/interface/MonitorElement.h"
// #include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;



PFCandidateManager::~PFCandidateManager() {}


void PFCandidateManager::setDirectory(TDirectory* dir) {
  
  Benchmark::setDirectory(dir);

  candBench_.setDirectory(dir);
  pfCandBench_.setDirectory(dir);
  matchCandBench_.setDirectory(dir);

} 

void PFCandidateManager::setParameters( float dRMax,
					bool matchCharge, 
					Benchmark::Mode mode) {
  dRMax_ = dRMax;
  matchCharge_ = matchCharge;
  mode_ = mode;
  
  candBench_.setParameters(mode);
  pfCandBench_.setParameters(mode);
  matchCandBench_.setParameters(mode);
  
}

void PFCandidateManager::setup() {
  candBench_.setup();
  pfCandBench_.setup();
  matchCandBench_.setup();
}



