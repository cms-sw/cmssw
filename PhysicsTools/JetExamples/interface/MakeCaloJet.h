#ifndef JetAlgorithms_MakeCaloJet_h
#define JetAlgorithms_MakeCaloJet_h

// MakeCaloJet.h
// Initial Version form Fernando Varela Rodriguez
// History: R. Harris, Oct 19, 2005, modified to work with real CaloTowers from Jeremy Mans

#include <vector>
class ProtoJet;
#include "PhysicsTools/Candidate/interface/CandidateFwd.h"

void MakeCaloJet(const reco::CandidateCollection &ctc, const std::vector<ProtoJet>& protoJets, 
		 reco::CandidateCollection &caloJets);

#endif
