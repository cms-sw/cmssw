#ifndef CandAlgos_CandSelectorBase_h
#define CandAlgos_CandSelectorBase_h
// Candidate Selector base producer module
// $Id: CandSelectorBase.h,v 1.5 2006/02/21 10:37:28 llista Exp $
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SelectorProducer.h"

typedef SelectorProducer<reco::CandidateCollection, 
			 CandSelector, 
			 edm::ClonePolicy<reco::Candidate> > CandSelectorBase;

#endif
