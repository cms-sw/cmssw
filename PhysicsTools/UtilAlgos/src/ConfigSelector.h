#ifndef UtilAlgos_ConfigSelector_h
#define UtilAlgos_ConfigSelector_h
/** \class ConfigSelector
 *
 * Selects objects from a collection and saves 
 * their clones in a new collection. The selection can 
 * be specified by the user as a string.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.9 $
 *
 * $Id: CandSelector.h,v 1.9 2006/07/25 17:26:44 llista Exp $
 *
 */
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/CandAlgos/src/SingleCandidateSelector.h"

typedef 
      ObjectSelector<
        SingleElementCollectionSelector<
          reco::CandidateCollection, 
          SingleObjectSelector
        > 
      > ConfigSelector;
  }
}

#endif
