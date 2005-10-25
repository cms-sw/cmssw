#ifndef CandAlgos_CandSelector_h
#define CandAlgos_CandSelector_h
// Ported from original implementation by Chris Jones
// $Id: CandSelector.h,v 1.3 2005/10/24 09:42:46 llista Exp $
//
#include "PhysicsTools/CandAlgos/interface/CandSelectorBase.h"

namespace candmodules {

  class CandSelector : public CandSelectorBase {
  public:
    explicit CandSelector( const edm::ParameterSet& );
    ~CandSelector();
  };

}

#endif /* CANDCOMBINER_CANDSELECTOR_H */
