#ifndef CandAlgos_CandSelectorBase_h
#define CandAlgos_CandSelectorBase_h
// Candidate Selector base producer module
// $Id: CandSelectorBase.h,v 1.4 2006/02/02 09:55:22 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SelectorProducer.h"
#include "DataFormats/Common/interface/ClonePolicy.h"

typedef SelectorProducer<reco::CandidateCollection, 
			 CandSelector, 
			 edm::ClonePolicy<reco::Candidate> > CandSelectorBase;

#endif
