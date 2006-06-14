#ifndef CandAlgos_CandSelectorBase_h
#define CandAlgos_CandSelectorBase_h
/** \class CandSelectorBase
 *
 * Base class for all candidate selectors.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.7 $
 *
 * $Id: CandSelectorBase.h,v 1.7 2006/03/03 10:48:20 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SelectorProducer.h"

typedef SelectorProducer<reco::CandidateCollection, CandSelector> CandSelectorBase;

#endif
