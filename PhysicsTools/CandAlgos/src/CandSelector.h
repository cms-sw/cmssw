#ifndef CANDCOMBINER_CANDSELECTOR_H
#define CANDCOMBINER_CANDSELECTOR_H
// Ported from original implementation by Chris Jones
// $Id: CandSelector.h,v 1.1 2005/10/24 06:08:18 llista Exp $
//
#include "PhysicsTools/CandAlgos/interface/CandSelectorBase.h"

class CandSelector : public CandSelectorBase {
public:
  explicit CandSelector( const edm::ParameterSet& );
  ~CandSelector();
};


#endif /* CANDCOMBINER_CANDSELECTOR_H */
