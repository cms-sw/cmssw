#ifndef CandAlgos_CandSelector_h
#define CandAlgos_CandSelector_h
/** \class candmodules::CandSelector
 *
 * Selects candidates from a collection and saves 
 * their clones in a new collection. The selection can 
 * be specified by the user as a string.
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: CandReducer.h,v 1.2 2006/03/03 10:20:44 llista Exp $
 *
 */
#include "PhysicsTools/CandAlgos/interface/CandSelectorBase.h"

namespace candmodules {

  class CandSelector : public CandSelectorBase {
  public:
    /// constructor from parameter set
    explicit CandSelector( const edm::ParameterSet& );
    /// destructor
    ~CandSelector();
  };

}

#endif
