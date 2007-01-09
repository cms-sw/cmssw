#ifndef BTauReco_TaggingVariableFwd_h
#define BTauReco_TaggingVariableFwd_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class TaggingVariableList;
  typedef std::vector<TaggingVariableList>                TaggingVariableListCollection;
  typedef edm::Ref<TaggingVariableListCollection>         TaggingVariableListRef;
  typedef edm::RefProd<TaggingVariableListCollection>     TaggingVariableListRefProd;
  typedef edm::RefVector<TaggingVariableListCollection>   TaggingVariableListRefVector;
}

#endif // BTauReco_TaggingVariableListFwd_h
