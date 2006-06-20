#ifndef CandAlgos_CandSelectorBase_h
#define CandAlgos_CandSelectorBase_h
/** \class CandSelectorBase
 *
 * Base class for all candidate selectors.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.8 $
 *
 * $Id: CandSelectorBase.h,v 1.8 2006/06/14 11:54:29 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SelectorProducer.h"

typedef SelectorProducerBase<reco::CandidateCollection, CandSelector> CandSelectorBase;

#endif
