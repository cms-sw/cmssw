#include "EventFilter/HcalRawToDigi/interface/AMC13Header.h"

namespace hcal {
  const uint64_t* AMC13Header::AMCPayload(int i) const {
    const uint64_t* ptr=(&cdfHeader)+14;
    for (int j=0; j<i; j++)
      ptr+=AMCSize(j);
    return ptr;
  }
}
