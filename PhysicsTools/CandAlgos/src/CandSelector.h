#ifndef CandAlgos_CandSelector_h
#define CandAlgos_CandSelector_h
/** \class cand::modules::CandSelector
 *
 * Selects candidates from a collection and saves 
 * their clones in a new collection. The selection can 
 * be specified by the user as a string.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.8 $
 *
 * $Id: CandSelector.h,v 1.8 2006/06/20 09:58:01 llista Exp $
 *
 */
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/CandAlgos/src/SingleCandidateSelector.h"

namespace cand {
  namespace modules {
    typedef 
      ObjectSelector<
        SingleElementCollectionSelector<
          reco::CandidateCollection, 
          SingleCandidateSelector
        > 
      > CandSelector;
  }
}

#endif
