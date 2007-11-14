
#include <vector>
#include <boost/cstdint.hpp> 
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace 
{
  namespace 
  {
    L1TriggerScalers l1TriggerScalersRef;
    LumiScalers lumiScalersRef;

    edm::Wrapper<L1TriggerScalers> w_l1TriggerScalers;
    edm::Wrapper<LumiScalers> w_lumiScalers;

    edm::RefProd<L1TriggerScalers> l1TriggerScalersRef ;
    edm::RefProd<L1GctInternEmCandCollection> lumiScalersRef ;
  }
}
