#ifndef CandAlgos_CandSelectorBase_h
#define CandAlgos_CandSelectorBase_h
/** \class CandSelectorBase
 *
 * Base class for all candidate selectors.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: CandReducer.h,v 1.2 2006/03/03 10:20:44 llista Exp $
 *
 */#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SelectorProducer.h"

typedef SelectorProducer<reco::CandidateCollection, 
			 CandSelector, 
			 edm::ClonePolicy<reco::Candidate> > CandSelectorBase;

#endif
