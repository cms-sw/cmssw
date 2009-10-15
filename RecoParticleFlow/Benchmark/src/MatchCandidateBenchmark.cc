#include "RecoParticleFlow/Benchmark/interface/MatchCandidateBenchmark.h"

#include "DataFormats/Candidate/interface/Candidate.h"


#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>


using namespace std;


MatchCandidateBenchmark::MatchCandidateBenchmark() {}

MatchCandidateBenchmark::~MatchCandidateBenchmark() {}


void MatchCandidateBenchmark::setup() {

  CandidateBenchmark::setup()
  delta_pt_ = book1D("delta_pt_", "delta pt_;#Deltap_{T} (GeV)", 100, -50, 50);
}


void MatchCandidateBenchmark::fill(const Collection& candCollection,
				   const Collection& matchCandCollection) {
  

  vector<int> matchIndices = match( candCollection, 
				    matchCandCollection );

  for (unsigned int i = 0; i < candCollection.size(); i++) {
    const reco::Candidate& cand = candCollection[i];

    int iMatch = matchIndices[i];

    assert(iMatch<matchCandCollection.size());
 
    // filling the histograms in CandidateBenchmark only in case 
    // of a matching. Is this a good solution? 
    if( iMatch!=-1 ) {
      CandidateBenchmark::fill(cand);
      fill( cand, matchCandCollection[ iMatch ] );
    }
  }
}


void MatchCandidateBenchmark::fill(const reco::Candidate& cand,
				   const reco::Candidate& matchedCand) {

  delta_pt_->Fill( cand.pt() - matchedCand.pt() );

}
