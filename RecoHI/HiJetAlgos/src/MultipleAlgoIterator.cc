#include "RecoHI/HiJetAlgos/interface/MultipleAlgoIterator.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;

void MultipleAlgoIterator::offsetCorrectJets()
{

  LogDebug("PileUpSubtractor")<<"The subtractor correcting jets...\n";

  jetOffset_.clear();

  using namespace reco;
  
  (*fjInputs_) = fjOriginalInputs_;
  subtractPedestal(*fjInputs_);
  const fastjet::JetDefinition& def = fjClusterSeq_->jet_def();
  if ( !doAreaFastjet_ && !doRhoFastjet_) {
    fastjet::ClusterSequence newseq( *fjInputs_, def );
    (*fjClusterSeq_) = newseq;
  } else {
    fastjet::ClusterSequenceArea newseq( *fjInputs_, def , *fjActiveArea_ );
    (*fjClusterSeq_) = newseq;
  }
  
  (*fjJets_) = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));
  
  jetOffset_.reserve(fjJets_->size());
  
  vector<fastjet::PseudoJet>::iterator pseudojetTMP = fjJets_->begin (),
    jetsEnd = fjJets_->end();
  for (; pseudojetTMP != jetsEnd; ++pseudojetTMP) {
    
    int ijet = pseudojetTMP - fjJets_->begin();
    jetOffset_[ijet] = 0;
    
    std::vector<fastjet::PseudoJet> towers =
      sorted_by_pt(fjClusterSeq_->constituents(*pseudojetTMP));
    
    double newjetet = 0.;
    for(vector<fastjet::PseudoJet>::const_iterator ito = towers.begin(),
	  towEnd = towers.end();
	ito != towEnd;
	++ito)
      {
	const reco::CandidatePtr& originalTower = (*inputs_)[ito->user_index()];
	int it = ieta( originalTower );
	double Original_Et = originalTower->et();
	double etnew = Original_Et - (*emean_.find(it)).second - (*esigma_.find(it)).second;
	if(etnew < 0.) etnew = 0;
	newjetet = newjetet + etnew;
	jetOffset_[ijet] += Original_Et - etnew;
      }
  }
}

