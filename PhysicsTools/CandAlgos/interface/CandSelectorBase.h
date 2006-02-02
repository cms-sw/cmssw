#ifndef CandAlgos_CandSelectorBase_h
#define CandAlgos_CandSelectorBase_h
// Candidate Selector base producer module
// $Id: CandSelectorBase.h,v 1.3 2005/10/25 09:08:31 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SelectorProducer.h"
#include "DataFormats/Common/interface/ClonePolicy.h"

typedef SelectorProducer<aod::CandidateCollection, 
			 CandSelector, 
			 edm::ClonePolicy<aod::Candidate> > CandSelectorBase;

#endif
