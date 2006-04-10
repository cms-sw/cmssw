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
 * \version $Revision: 1.6 $
 *
 * $Id: CandSelector.h,v 1.6 2006/04/04 11:12:48 llista Exp $
 *
 */
#include "PhysicsTools/CandAlgos/interface/CandSelectorBase.h"

namespace cand {
  namespace modules {
    
    class CandSelector : public CandSelectorBase {
    public:
      /// constructor from parameter set
      explicit CandSelector( const edm::ParameterSet& );
      /// destructor
      ~CandSelector();
    };

  }
}

#endif
