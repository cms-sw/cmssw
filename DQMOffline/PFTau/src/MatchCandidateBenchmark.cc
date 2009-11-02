#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"

#include "DataFormats/Candidate/interface/Candidate.h"


// #include "DQMServices/Core/interface/MonitorElement.h"
// #include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;



MatchCandidateBenchmark::~MatchCandidateBenchmark() {}


void MatchCandidateBenchmark::setup( bool setupMother) {

  delta_pt_ = book1D("delta_pt_", "delta pt_;#Deltap_{T} (GeV)", 100, -50, 50);
}



void MatchCandidateBenchmark::fillOne(const reco::Candidate& cand,
				      const reco::Candidate& matchedCand) {

  delta_pt_->Fill( cand.pt() - matchedCand.pt() );

}
