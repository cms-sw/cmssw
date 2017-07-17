#include "DQMOffline/PFTau/interface/PFCandidateManager.h"


#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"


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

void PFCandidateManager::setup(DQMStore::IBooker& b) {
  candBench_.setup(b);
  pfCandBench_.setup(b);
  matchCandBench_.setup(b);
}
