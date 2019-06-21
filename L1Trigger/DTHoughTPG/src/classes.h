#include "DataFormats/Common/interface/Wrapper.h"

#include "L1Trigger/DTHoughTPG/interface/DTHough.h"

#include <vector>
#include <map>
#include <utility>

namespace
{
  namespace
  {
    DTHough< RefDTDigi_t > HT_DG;
    std::vector< DTHough< RefDTDigi_t > > vHT_DG;
    edm::Wrapper< std::vector< DTHough< RefDTDigi_t > > > wvHT_DG;
  }
}


